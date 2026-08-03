import * as chrono from 'chrono-node';
import Fuse from 'fuse.js';
import { z } from 'zod';
import type { UsageMetadata } from '@langchain/core/messages';
import { llm, LLMMessage } from '../core/model-abstraction/llm.provider';
import { TOOL_MODEL_PRESETS } from '../config';
import { estimateCostUsd } from '../config/token-cost';
import { buildRAGContextWithConfidence } from '../rag/pipeline';
import { classifyResponseConfidence, CONFIDENCE_THRESHOLD } from './response-confidence';
import {
  getHistory, appendMessage, getCRMState, setCRMState,
  getPendingIntent, setPendingIntent, clearPendingIntent,
  getLeadCaptureState, setLeadCaptureState, type LeadCaptureState,
  getBookingState, type BookingState,
  type PendingIntent,
} from '../memory/conversation.memory';
import { moderateContent, detectPromptInjection } from '../core/guardrails/moderation';
import { maskPIIForLLM } from '../core/guardrails/pii-filter';
import { checkTenantRateLimit, checkTenantTokenQuota } from '../core/guardrails/rate-limiter';
import { resolveTenantConfig } from '../services/context.builder';
import { cleanArg } from '../utils/clean-arg';
import { extractCapturedData } from '../utils/extract-captured-data';
import { tryBookingConfirmationShortcut } from './booking-confirmation-shortcut';
import { backendClient, type CRMSearchResult } from '../services/backend.client';
import { buildCRMQueryPrompt } from '../prompts/system.prompts';
import { LeadFieldExtractionSchema } from './widget-lead.schema';
import { checkFastPath } from './fast-path';
import { looksLikeProfileQuestion } from './website-profile-fast-path';
import { getToolsForSurface } from '../tools/registry';
import { runToolLoop, ToolCallLog } from '../tools/runner';
import { logger } from '../utils/logger';

export interface AgentInput {
  tenantId: string;
  sessionId: string;
  message: string;
  systemPrompt: string;
  /** Optional: override resolved tenant branding (used when backend is unavailable) */
  companyName?: string;
  agentName?: string;
  language?: string;
  /** Present only for the public website widget's own conversations — gates
   * maybeCaptureWidgetLead() below, so an internal staff member casually
   * chatting with their own tenant's AI assistant never accidentally
   * creates a spurious Lead. */
  visitorId?: string;
  pageUrl?: string;
}

export interface AgentOutput {
  response: string;
  escalate: boolean;
  capturedData: Record<string, string>;
}

// These phrases in the AI *response* unambiguously mean the bot is escalating the session.
// Intentionally narrow — product names like "Safety Complaint Course" or "Emergency Services"
// must NOT trigger false escalation when the bot lists CRM data.
const AI_ESCALATION_PHRASES = [
  'connect you with', 'human agent', 'escalat', 'not able to help',
];

// User explicitly requesting a human (checked against user message only, not AI response).
const USER_ESCALATION_PHRASES = [
  'speak to a human', 'talk to a human', 'need a human', 'real person',
  'human support', 'i want to sue', 'legal action',
];

const LLM_TIMEOUT_MS = 50000;
// The tool loop has its own internal budget (35s, see runner.ts) plus a
// final unbound call that can itself retry primary->fallback (up to 2 more
// 20s-capped calls) — this outer bound stays above that (so the loop's own
// graceful-degradation path always gets a chance to run) while staying
// safely under the backend's own axios proxy timeout to /api/chat (raised to
// 100s for this path — see ai.routes.ts).
const TOOL_LOOP_TIMEOUT_MS = 90000;
// maybeCaptureWidgetLead() runs fire-and-forget (see its call sites below) —
// this timeout is defense-in-depth only, since nothing awaits this call
// anymore, but it still stops a hung request from lingering indefinitely.
const LEAD_EXTRACTION_TIMEOUT_MS = 15000;

function shouldEscalate(response: string, userMessage: string): boolean {
  const responseLower = response.toLowerCase();
  const userLower     = userMessage.toLowerCase();
  return (
    AI_ESCALATION_PHRASES.some((s) => responseLower.includes(s)) ||
    USER_ESCALATION_PHRASES.some((s) => userLower.includes(s))
  );
}

/** The public website widget's own lead-capture-to-CRM flow. Only ever runs
 * when `visitorId` is present (see AgentInput's own comment on why). Merges
 * the cheap regex first-pass (capturedNow) into the persisted
 * LeadCaptureState, then — only if a name or contact method is still
 * missing — calls generateStructured() to fill the gaps (regex alone is
 * fundamentally weak at names/companies across real phrasing variety).
 * Once firstName + (email|phone) are known and no Lead has been created yet
 * for this session, calls the backend's real, already-built
 * captureLeadFromExternalSource() (via createLeadFromWidget()). Never
 * throws — a failure here degrades to "ask again next turn", never crashes
 * the chat response the visitor is waiting on. */
async function maybeCaptureWidgetLead(
  input: AgentInput,
  capturedNow: Record<string, string>,
): Promise<void> {
  if (!input.visitorId) return;
  try {
    const state: LeadCaptureState = (await getLeadCaptureState(input.sessionId)) ?? {};
    if (state.leadCreated) return;

    if (capturedNow.email && !state.email) state.email = capturedNow.email;
    if (capturedNow.phone && !state.phone) state.phone = capturedNow.phone;
    if (capturedNow.name && !state.firstName) {
      const parts = capturedNow.name.split(/\s+/).filter(Boolean);
      state.firstName = parts[0];
      if (parts.length > 1) state.lastName = parts.slice(1).join(' ');
    }

    // Only call the LLM when something's still missing — skip entirely once
    // this session already has a usable name + contact method, to avoid
    // burning tokens on every turn of an already-qualified conversation.
    if (!state.firstName || (!state.email && !state.phone)) {
      try {
        const extraction = await withTimeout(
          llm.generateStructured(
            [
              {
                role: 'system',
                content: 'Extract the visitor\'s first name, last name, email, phone, company, and the service/reason they\'re contacting about, from their message below, if mentioned. Use null for anything not present — do not guess or fabricate.',
              },
              { role: 'user', content: input.message },
            ],
            LeadFieldExtractionSchema,
          ),
          LEAD_EXTRACTION_TIMEOUT_MS,
          'Widget lead-field extraction',
        );
        // cleanArg() guards against Groq's occasional literal-string "null"
        // (instead of real JSON null) for a field it doesn't know — without
        // this, z.string().nullable() lets "null" through as a truthy
        // string, producing a Lead with firstName/lastName literally set to
        // the text "null" (confirmed live: exactly this happened for a real
        // Doctor-tenant conversation before this fix).
        const cFirst = cleanArg(extraction.firstName ?? undefined);
        const cLast  = cleanArg(extraction.lastName ?? undefined);
        const cEmail = cleanArg(extraction.email ?? undefined);
        const cPhone = cleanArg(extraction.phone ?? undefined);
        const cCompany = cleanArg(extraction.company ?? undefined);
        const cService = cleanArg(extraction.service ?? undefined);
        if (cFirst && !state.firstName)   state.firstName = cFirst;
        if (cLast && !state.lastName)     state.lastName  = cLast;
        if (cEmail && !state.email)       state.email     = cEmail;
        if (cPhone && !state.phone)       state.phone     = cPhone;
        if (cCompany && !state.company)   state.company   = cCompany;
        if (cService && !state.service)   state.service   = cService;
      } catch (err) {
        logger.warn('Widget lead-field structured extraction failed', {
          sessionId: input.sessionId, error: (err as Error).message,
        });
      }
    }

    await setLeadCaptureState(input.sessionId, state);

    if (state.firstName && (state.email || state.phone)) {
      const result = await backendClient.createLeadFromWidget({
        tenantId:  input.tenantId,
        sessionId: input.sessionId,
        visitorId: input.visitorId,
        sourceUrl: input.pageUrl,
        firstName: state.firstName,
        lastName:  state.lastName,
        email:     state.email,
        phone:     state.phone,
        company:   state.company,
        service:   state.service,
      });
      if (result.success && result.leadId) {
        state.leadCreated = true;
        state.leadId = result.leadId;
        await setLeadCaptureState(input.sessionId, state);
      }
    }
  } catch (err) {
    logger.warn('maybeCaptureWidgetLead failed', { sessionId: input.sessionId, error: (err as Error).message });
  }
}

/* ── Lead-capture signal detection (don't go to CRM mode for these) ─ */
const LEAD_CAPTURE_SIGNALS = [
  'i want to buy', 'i want to book', 'i want to enquire', 'i want to know more',
  'i am interested', "i'm interested", 'book an appointment', 'schedule a call',
  'hello', 'hi there', 'good morning', 'good evening', 'good afternoon',
  'my name is', 'i am from', 'please contact me', 'call me',
];

/* ── CRM intent detection ─────────────────────────────────────────── */
const CRM_QUERY_TRIGGERS = [
  'how many', 'how much', 'show me', 'show all', 'list all', 'list my', 'count',
  'any ', 'is there', 'are there', 'is it', 'find', 'get me', 'what are', 'display',
  'fetch', 'give me', 'tell me about', 'recent', 'latest', 'last', 'what is the',
  'what\'s the', 'price of', 'cost of', 'do you have', 'do we have', 'i need',
  'show', 'check', 'which', 'where', 'when', 'total number', 'how much is',
];

const CRM_DATA_KEYWORDS = [
  'invoice', 'account', 'deal', 'lead', 'contact', 'customer', 'opportunity',
  'campaign', 'product', 'meeting', 'order', 'case', 'task', 'call', 'note',
  'quote', 'ticket', 'record', 'data', 'crm', 'vendor', 'item', 'price', 'cost',
  'amount', 'total', 'discount', 'quantity', 'tax', 'rate', 'stock', 'inventory',
  'history', 'deal', 'pipeline', 'stage', 'status', 'report',
];

// A cheap, narrowing-only signal that a turn is booking-shaped — used to
// skip binding catalog/RAG tools on that turn. Plain substring matching, so
// it can false-positive on words like "notebook"/"textbook"; the worst case
// is one turn where booking tools are bound instead of the ones actually
// needed, never a broken booking flow, so this is accepted as-is.
const BOOKING_ONLY_SIGNALS = [
  'appointment', 'book', 'schedule', 'slot', 'available time', 'meeting',
];

function isBookingOnlyMessage(message: string): boolean {
  const lower = message.toLowerCase();
  return BOOKING_ONLY_SIGNALS.some((s) => lower.includes(s));
}

// Channel names the user might type in a message
const CHANNEL_ALIASES: Record<string, string[]> = {
  hubspot:    ['hubspot', 'hub spot'],
  salesforce: ['salesforce', 'sales force'],
  zoho:       ['zoho'],
  mysql:      ['mysql', 'my sql'],
  postgresql: ['postgresql', 'postgres'],
  mongodb:    ['mongodb', 'mongo'],
  rest:       ['rest api', 'rest connector'],
};

function detectCRMIntent(
  message: string,
  crmModules: Record<string, Array<{ module: string; count: number }>>,
  hasConnectors: boolean,
  inlineRecordContext?: string,
  preferredChannel?: string
): { isCRMQuery: boolean; channel?: string; module?: string; explicitChannel?: string } {
  const lower = message.toLowerCase().trim();
  const hasConnectedCRM = hasConnectors && Object.keys(crmModules).length > 0;

  // Never override explicit lead capture signals
  if (LEAD_CAPTURE_SIGNALS.some((s) => lower.startsWith(s) || lower === s)) {
    return { isCRMQuery: false };
  }

  // Detect if user explicitly named a connector (e.g. "zoho have how many contacts?")
  // When found, check that channel's modules FIRST so "zoho contacts" doesn't match HubSpot
  let explicitChannel: string | undefined;
  for (const [ch, aliases] of Object.entries(CHANNEL_ALIASES)) {
    if (aliases.some((alias) => lower.includes(alias))) {
      explicitChannel = ch;
      break;
    }
  }

  // 1. Match against actual module names — explicit channel checked first, then preferred (from prev turn)
  const channelOrder = explicitChannel
    ? [explicitChannel, ...Object.keys(crmModules).filter((ch) => ch !== explicitChannel)]
    : preferredChannel
      ? [preferredChannel, ...Object.keys(crmModules).filter((ch) => ch !== preferredChannel)]
      : Object.keys(crmModules).sort((a, b) => {
          // Primary connector (most records) first — prevents Salesforce winning over Zoho when both have same module
          const sumA = (crmModules[a] || []).reduce((s, m) => s + m.count, 0);
          const sumB = (crmModules[b] || []).reduce((s, m) => s + m.count, 0);
          return sumB - sumA;
        });

  for (const channel of channelOrder) {
    const modules = crmModules[channel] || [];
    for (const { module } of modules) {
      const modLower = module.toLowerCase();
      // Match plural form (Products) or singular stem (product→Products)
      const modStem = modLower.replace(/ies$/, 'y').replace(/s$/, '');
      if (lower.includes(modLower) || (modStem.length > 3 && lower.includes(modStem))) {
        return { isCRMQuery: true, channel, module };
      }
      // CamelCase expansion: "DealHistory" → ["deal","history"] — ALL tokens must appear in message
      // Fixes "deal history" not matching "DealHistory" when direct string include fails
      const camelTokens = module
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .toLowerCase()
        .split(/\s+/)
        .filter((t) => t.length > 2);
      if (camelTokens.length >= 2 && camelTokens.every((t) => lower.includes(t))) {
        return { isCRMQuery: true, channel, module };
      }
    }
  }

  // 2. If we have inline records already injected in context, check if message words
  //    appear in the record names (e.g. "2gb ram" appears in a record displayName).
  //    This lets "2gb ram is there?" match without knowing the module name.
  if (hasConnectedCRM && inlineRecordContext) {
    const words = lower.replace(/[?.,!]/g, '').split(' ').filter((w) => w.length > 2);
    for (const [channel, modules] of Object.entries(crmModules)) {
      for (const { module } of modules) {
        const modulePattern = module.toLowerCase().replace(/[^a-z0-9 ]/g, '');
        if (words.some((w) => inlineRecordContext.toLowerCase().includes(w) && modulePattern)) {
          return { isCRMQuery: true, channel, module };
        }
      }
    }
  }

  // 3. Trigger word match — if connected CRM exists, a trigger alone is enough
  const hasTrigger = CRM_QUERY_TRIGGERS.some((t) => lower.includes(t));
  if (hasTrigger) {
    if (hasConnectedCRM) {
      // Try to find a specific module via keyword — respect explicit channel ordering
      const matchedKw = CRM_DATA_KEYWORDS.find((k) => lower.includes(k));
      if (matchedKw) {
        for (const channel of channelOrder) {
          const modules = crmModules[channel] || [];
          const mod = modules.find(
            (m) => m.module.toLowerCase().includes(matchedKw) || matchedKw.includes(m.module.toLowerCase())
          );
          if (mod) return { isCRMQuery: true, channel, module: mod.module };
        }
      }
      // If user named a connector explicitly, pass it through even without a module match
      if (explicitChannel) return { isCRMQuery: true, channel: explicitChannel };
      return { isCRMQuery: true };
    }
  }

  // 4. Short factual questions when CRM is connected ("what is this price?", "how much?")
  const isShortQuestion = lower.split(' ').length <= 8;
  const isFactualPattern = /^(what|how|is|are|do|does|can|which|where|when|who|why)\b/.test(lower);
  if (hasConnectedCRM && isShortQuestion && isFactualPattern) {
    return { isCRMQuery: true, explicitChannel };
  }

  return { isCRMQuery: false, explicitChannel };
}

/* ── Format actual CRM records for LLM injection ───────────────── */

// Exact-name internal fields to always skip
const SKIP_FIELD_RE = /^(id|_id|externalid|tenantid|attributes|systemmodstamp|isdeleted|lastsyncdat|createdat|updatedat|hs_createdate|hs_lastmodifieddate|hs_object_source_label|hs_object_source_detail|hs_object_type|hs_all_owner_ids|hs_user_ids_of_all_owners|hubspot_owner_assigneddate|hs_object_id|sfdc_id)$/i;

// Prefix/substring patterns for internal metadata — HubSpot v2 stage-tracking, Salesforce system fields
const SKIP_META_RE = /^(hs_v2_|hs_date_entered_|hs_time_in_|sfdc_internal_|lastactivity)/i;

const LONG_VALUE_RE = /^[A-Za-z0-9+/]{100,}={0,2}$/; // base64 / very long blobs

function pickTopFields(data: Record<string, unknown>, max = 10): Array<[string, unknown]> {
  return Object.entries(data)
    .filter(([k, v]) => {
      if (SKIP_FIELD_RE.test(k)) return false;
      if (SKIP_META_RE.test(k)) return false;
      const s = String(v ?? '').trim();
      if (!s || s === 'null' || s === 'undefined') return false;
      if (LONG_VALUE_RE.test(s)) return false;
      if (s.length > 200) return false;
      return true;
    })
    .sort(([ka, va], [kb, vb]) => fieldPriority(ka, va) - fieldPriority(kb, vb))
    .slice(0, max);
}

/* Converts camelCase module names to readable display names: DealHistory → Deal History */
function moduleDisplayName(module: string): string {
  return module.replace(/([a-z])([A-Z])/g, '$1 $2');
}

function formatRecordsForLLM(
  records: Array<{ externalId: string; displayName: string; data: Record<string, unknown>; syncedAt: string }>,
  module: string,
  channel: string,
  totalCount?: number,
  isFiltered = false   // true when records are already server-side filtered
): string {
  const modDisplay = moduleDisplayName(module);
  if (!records.length) return `No ${modDisplay} records found in ${channel.toUpperCase()}.`;

  const SHOW_MAX   = 50;
  const FIELDS_MAX = isFiltered ? 15 : 10; // show more fields per record for filtered results
  const total      = totalCount ?? records.length;
  const showing    = Math.min(records.length, SHOW_MAX);
  const header     = total > showing
    ? `=== CRM DATA: ${modDisplay} from ${channel.toUpperCase()} (${total} records total, showing first ${showing}) ===`
    : `=== CRM DATA: ${modDisplay} from ${channel.toUpperCase()} (${total} records) ===`;

  const lines = [header];
  let stripped = 0;
  for (const r of records.slice(0, SHOW_MAX)) {
    const fields = pickTopFields(r.data, FIELDS_MAX);
    const fieldStr = fields.map(([k, v]) => `${k}: ${v}`).join(', ');
    const recordText = `${r.displayName || r.externalId} ${fieldStr}`;
    const injectionCheck = detectPromptInjection(recordText);
    if (!injectionCheck.safe) {
      logger.warn('CRM record stripped from LLM context: prompt injection pattern detected', {
        externalId: r.externalId,
        reason: injectionCheck.reason,
      });
      stripped++;
      continue;
    }
    lines.push(`• ${r.displayName || r.externalId}${fieldStr ? ' | ' + fieldStr : ''}`);
  }
  if (stripped > 0) {
    lines.push(`[NOTE: ${stripped} record(s) omitted due to security policy.]`);
  }
  lines.push('=================================');
  // When records are pre-filtered server-side, instruct the LLM not to re-apply the filter
  if (isFiltered) {
    lines.push(`[IMPORTANT: All ${records.length} records above are pre-verified matches. Present ALL of them — do NOT skip or re-filter any record.]`);
  }
  return lines.join('\n');
}

/* ── Find best matching CRM module when search term is empty ─────
   "show me all deals" → searchTerm strips everything → use this
   to match "deals" against actual module names in the tenant DB. */
function findBestModule(
  message: string,
  crmModules: Record<string, Array<{ module: string; count: number }>>,
  preferChannel?: string
): { channel: string; module: string } | null {
  const words = message
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 2);

  let best: { channel: string; module: string; score: number } | null = null;

  // Check preferred channel first so ties resolve in its favour
  const channelOrder = preferChannel
    ? [preferChannel, ...Object.keys(crmModules).filter((ch) => ch !== preferChannel)]
    : Object.keys(crmModules);

  for (const channel of channelOrder) {
    const modules = crmModules[channel] || [];
    for (const { module } of modules) {
      const modTokens = module
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .toLowerCase()
        .split(/[\s_\-]+/)
        .filter((t) => t.length > 1);

      let score = 0;
      for (const w of words) {
        for (const t of modTokens) {
          if (t.startsWith(w) || w.startsWith(t)) score++;
        }
      }
      if (score > 0 && (!best || score > best.score)) {
        best = { channel, module, score };
      }
    }
  }

  return best;
}

/* ── Server-side numeric filter for price/amount comparisons ─────────────────
   LLMs are unreliable at math. When the user asks "price less than 1000",
   we apply the filter in JavaScript BEFORE sending records to the LLM so the
   result is always 100% accurate. The LLM only formats the pre-filtered list. */

type FilterOp = 'lt' | 'lte' | 'gt' | 'gte' | 'between';

interface NumericFilter {
  operator: FilterOp;
  value: number;
  value2?: number; // used by 'between'
}

// Field names that typically hold a monetary / numeric amount
const PRICE_FIELD_RE = /^(price|unit.?price|amount|cost|rate|fee|value|total|revenue|salary|charge|budget|deal.?amount|invoice.?amount)/i;

// Primary amount fields — checked FIRST to avoid secondary estimation fields (Expected_Revenue, Budget)
// with value 0 incorrectly failing a numeric filter (e.g. Chanay Amount=55000 but Expected_Revenue=0)
const PRIMARY_AMOUNT_RE = /^(amount|deal_amount|invoice_amount|grand_total|total_amount|list_price|unit_price|sale_price)$/i;

function extractNumericFilter(message: string): NumericFilter | null {
  // Strip currency symbols and commas so "₹1,000" → "1000"
  const m = message.toLowerCase().replace(/[₹$,]/g, '');

  // between X and Y
  const bm = m.match(/between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)/);
  if (bm) return { operator: 'between', value: parseFloat(bm[1]), value2: parseFloat(bm[2]) };

  // less than / below / under / < / lessthan
  const lt = m.match(/(?:less\s*than|lessthan|below|under|<)\s*(\d+(?:\.\d+)?)/);
  if (lt) return { operator: 'lt', value: parseFloat(lt[1]) };

  // more than / above / over / > / greaterthan
  const gt = m.match(/(?:more\s*than|morethan|greaterthan|greater\s*than|above|over|>)\s*(\d+(?:\.\d+)?)/);
  if (gt) return { operator: 'gt', value: parseFloat(gt[1]) };

  // up to / upto / at most / max
  const lte = m.match(/(?:up\s*to|upto|at\s*most|max(?:imum)?)\s*(\d+(?:\.\d+)?)/);
  if (lte) return { operator: 'lte', value: parseFloat(lte[1]) };

  // at least / atleast / min / minimum
  const gte = m.match(/(?:at\s*least|atleast|min(?:imum)?)\s*(\d+(?:\.\d+)?)/);
  if (gte) return { operator: 'gte', value: parseFloat(gte[1]) };

  return null;
}

/* ── Detect aggregate intent (sum/average) — computed server-side to avoid LLM math errors ── */
function detectAggregateIntent(message: string): 'sum' | 'average' | null {
  const m = message.toLowerCase();
  // Exclude count-only queries ("total number of deals" is a count, not a sum of amounts)
  if (/\b(total|sum)\b.*\b(number|count|many)\b/.test(m)) return null;
  if (/\b(total|sum|add\s*up|combined|altogether)\b/.test(m) &&
      /\b(amount|price|cost|revenue|value|worth)\b/.test(m)) return 'sum';
  if (/\b(average|avg|mean)\b/.test(m) &&
      /\b(amount|price|cost|revenue|value)\b/.test(m)) return 'average';
  return null;
}

type CRMRecord = { externalId: string; displayName: string; data: Record<string, unknown>; syncedAt: string };

// Matches field names that CONTAIN amount/price/value/total keywords anywhere — catches Amount_INR, Deal_Value, GrandTotal, etc.
const AMOUNT_CONTAINS_RE = /amount|price|deal.?value|grand.?total/i;

function parseAmount(val: unknown): number | null {
  const s = String(val ?? '').replace(/[^0-9.]/g, '');
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

function getNumericFieldValue(data: Record<string, unknown>): number | null {
  // Pass 1: exact primary amount fields (Amount, Total_Amount, etc.) — most reliable for CRM deals
  for (const [key, val] of Object.entries(data)) {
    if (!PRIMARY_AMOUNT_RE.test(key)) continue;
    const n = parseAmount(val);
    if (n !== null && n >= 0) return n;
  }
  // Pass 2: field name STARTS WITH price/amount/cost/value/total/etc. — skips zeros (likely unset)
  for (const [key, val] of Object.entries(data)) {
    if (!PRICE_FIELD_RE.test(key)) continue;
    const n = parseAmount(val);
    if (n !== null && n > 0) return n;
  }
  // Pass 3: field name CONTAINS "amount" or "price" anywhere — catches Amount_INR, UnitPrice, DealAmount, etc.
  for (const [key, val] of Object.entries(data)) {
    if (!AMOUNT_CONTAINS_RE.test(key)) continue;
    if (PRIMARY_AMOUNT_RE.test(key) || PRICE_FIELD_RE.test(key)) continue; // already checked
    const n = parseAmount(val);
    if (n !== null && n > 0) return n;
  }
  return null;
}

function applyNumericFilter(records: CRMRecord[], filter: NumericFilter): CRMRecord[] {
  return records.filter((r) => {
    const v = getNumericFieldValue(r.data);
    if (v === null) return false;
    switch (filter.operator) {
      case 'lt':      return v <  filter.value;
      case 'lte':     return v <= filter.value;
      case 'gt':      return v >  filter.value;
      case 'gte':     return v >= filter.value;
      case 'between': return v >= filter.value && v <= (filter.value2 ?? filter.value);
    }
  });
}

function filterLabel(f: NumericFilter): string {
  switch (f.operator) {
    case 'lt':      return `price < ${f.value}`;
    case 'lte':     return `price ≤ ${f.value}`;
    case 'gt':      return `price > ${f.value}`;
    case 'gte':     return `price ≥ ${f.value}`;
    case 'between': return `price between ${f.value} and ${f.value2}`;
  }
}

/* ── Server-side string filter for status/stage queries ──────────────────────
   "active accounts" / "deals in proposal stage" / "paid invoices"
   Applied before LLM — reliable text match, no hallucination risk. */
const STAGE_FIELD_RE  = /^(stage|deal_stage|phase|pipeline_stage)/i;
const STATUS_FIELD_RE = /^(status|state|lead_status|account_type|call_status|invoice_status|campaign_status)/i;

interface StringFilter {
  fieldPattern: RegExp;
  value: string;
}

function extractStringFilter(message: string): StringFilter | null {
  const m = message.toLowerCase();
  // Stage values common to Zoho Deals
  const stageMatch = m.match(/\b(proposal|negotiation|closed\s*won|closed\s*lost|qualification|prospecting|won|lost|review)\b/);
  if (stageMatch) return { fieldPattern: STAGE_FIELD_RE, value: stageMatch[1].replace(/\s+/g, ' ') };
  // Generic status values
  const statusMatch = m.match(/\b(active|inactive|pending|completed|done|open|closed|sent|draft|paid|unpaid|overdue|new|converted)\b/);
  if (statusMatch) return { fieldPattern: STATUS_FIELD_RE, value: statusMatch[1] };
  return null;
}

function applyStringFilter(records: CRMRecord[], f: StringFilter): CRMRecord[] {
  return records.filter((r) => {
    for (const [key, val] of Object.entries(r.data)) {
      if (!f.fieldPattern.test(key)) continue;
      if (String(val ?? '').toLowerCase().includes(f.value)) return true;
    }
    return false;
  });
}

/* ── LLM-powered NLU: structured intent + entity extraction ─────────── */

const ChatIntentSchema = z.object({
  intent: z.enum(['schedule_meeting', 'reschedule_meeting', 'cancel_meeting', 'send_email', 'crm_query', 'general_chat']),
  confidence: z.number().min(0).max(1),
  entities: z.object({
    person:       z.string().nullable(),
    rawDateTime:  z.string().nullable(),
    meetingType:  z.string().nullable(),
    notes:        z.string().nullable(),
  }).passthrough(), // allow extra LLM fields (e.g. "duration") without failing
  missingRequired:       z.array(z.string()),
  clarificationQuestion: z.string().nullable(),
}).passthrough();

type ChatIntent = z.infer<typeof ChatIntentSchema>;

// Only call extractChatIntent when the message looks automation-related.
// This avoids the extra LLM round-trip for pure CRM queries and general chat.
const MIGHT_BE_AUTOMATION_RE = /\b(book|schedule|reschedule|meeting|appointment|booking|call|cancel|email|send|arrange|setup|plan|create|rescheduling|postpone|move|shift)\b/i;

async function extractChatIntent(
  message: string,
  history: Array<{ role: string; content: string }>,
  tenantName: string,
): Promise<ChatIntent | null> {
  try {
    const model = (llm.getModel() as any).withStructuredOutput(ChatIntentSchema);

    const systemPrompt = `You are an intent extractor for ${tenantName}'s business AI assistant.
Extract the user's intent and entities from their message. The user may type naturally, with typos, abbreviations, or broken English.

Intent definitions:
- schedule_meeting: user wants to CREATE a new meeting, appointment, booking, or call
- reschedule_meeting: user wants to CHANGE the time or date of an existing meeting
- cancel_meeting: user wants to CANCEL an existing meeting
- send_email: user wants to send a follow-up or promotional email to a contact
- crm_query: user is asking about EXISTING CRM data (contacts, deals, invoices, reports, tasks, activities)
- general_chat: anything else — greeting, small talk, questions about the business

Rules:
- For schedule/reschedule: extract person name and datetime EXACTLY as the user typed them (do NOT normalize or interpret them).
- If the intent is schedule_meeting but NO time is mentioned: set missingRequired = ["time"] and clarificationQuestion = "What time should I schedule the meeting?".
- If the intent is schedule_meeting but NO person is mentioned: that is OK — set person to null, missingRequired stays empty (meeting can be created without a contact).
- If both date AND time are missing: set missingRequired = ["time"] and ask for it.
- "reschedule", "postpone", "move the meeting", "change appointment time" → reschedule_meeting.
- "send email", "send followup", "email to X" → send_email.
- "show my appointments", "list tasks", "any meetings today?" → crm_query (NOT schedule_meeting).`;

    const historyBlock = history.slice(-4).map((h) => `${h.role}: ${h.content}`).join('\n');
    const userPrompt = historyBlock
      ? `Recent conversation:\n${historyBlock}\n\nCurrent message: ${message}`
      : message;

    const result = await model.invoke([
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: userPrompt },
    ]);
    return result as ChatIntent;
  } catch (err) {
    const errMsg = (err as Error).message ?? '';
    // Groq strict mode rejects extra fields and returns the raw generated content in
    // failed_generation. Try to parse it as JSON. If it starts with '<' (HTML/XML),
    // skip parsing — it's a server error page, not a partial intent.
    const fgMatch = errMsg.match(/"failed_generation"\s*:\s*"([\s\S]+?)(?="\s*[,}])/);
    if (fgMatch) {
      const rawContent = fgMatch[1].replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\\\/g, '\\').trim();
      if (rawContent.startsWith('{')) {
        try {
          const parsed = JSON.parse(rawContent);
          return ChatIntentSchema.parse(parsed) as ChatIntent;
        } catch { /* fall through */ }
      }
    }
    logger.warn('extractChatIntent failed — falling through to LLM path', { error: errMsg.slice(0, 200) });
    return null;
  }
}

// Extracts name + optional inline email from customer replies.
// "name : suganth A and email: suganth2501@gmail.com" → { name: "suganth A", email: "suganth2501@gmail.com" }
// "reply with name : Sugan" → { name: "Sugan", email: null }
function parseCustomerReply(msg: string): { name: string; email: string | null } {
  const emailMatch = msg.match(/\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b/);
  const email = emailMatch ? emailMatch[1] : null;

  let withoutEmail = email ? msg.replace(email, '').trim() : msg;
  const name = withoutEmail
    .replace(/^(reply\s+with\s+(name\s*[:\-—]?\s*)?|name\s*[:\-—]\s*|it'?s\s+|i\s+am\s+|i'm\s+|this\s+is\s+|for\s+|the\s+person\s+is\s+|contact\s+is\s+|customer\s+is\s+)/i, '')
    .replace(/\s*(and\s+)?email\s*[:\-—]?\s*$/i, '')
    .replace(/[,;:]+$/, '')
    .trim();

  return { name, email };
}

// Normalize phone to E.164 for Twilio — handles Indian local format (07xxxxxxxxx → +917xxxxxxxxx)
function normalizePhone(raw: string): string {
  const digits = raw.replace(/[\s\-().+]/g, '');
  if (raw.startsWith('+')) return raw.trim();                           // already E.164
  if (digits.length === 12 && digits.startsWith('91')) return `+${digits}`; // 917010935239
  if (digits.length === 11 && digits.startsWith('0'))  return `+91${digits.slice(1)}`; // 07010935239
  if (digits.length === 10)                            return `+91${digits}`; // 7010935239
  return raw.trim(); // unknown — pass through unchanged
}

function mergePendingIntent(pending: PendingIntent, reply: ChatIntent | null): PendingIntent {
  const merged = { ...pending, entities: { ...pending.entities } };
  const replyEntities = reply?.entities;

  // Fill in what was missing
  if (!merged.entities.person && replyEntities?.person)           merged.entities.person = replyEntities.person;
  if (!merged.entities.rawDateTime && replyEntities?.rawDateTime) merged.entities.rawDateTime = replyEntities.rawDateTime;
  if (!merged.entities.meetingType && replyEntities?.meetingType) merged.entities.meetingType = replyEntities.meetingType;

  // Rebuild missingRequired based on what's still absent
  merged.missingRequired = merged.missingRequired.filter((field) => {
    if (field === 'person'      && merged.entities.person)      return false;
    if (field === 'time'        && merged.entities.rawDateTime) return false;
    if (field === 'date'        && merged.entities.rawDateTime) return false;
    return true;
  });
  merged.clarificationQuestion = merged.missingRequired.length > 0
    ? pending.clarificationQuestion
    : null;
  return merged;
}

/* ── Build Q&A context block for LLM injection ───────────────────── */
function formatQnAContext(pairs: Array<{ question: string; answer: string; category: string }>): string {
  if (!pairs.length) return '';
  const lines = ['=== CUSTOM Q&A (TRAINED ANSWERS — ALWAYS USE THESE FIRST) ==='];
  for (const p of pairs) {
    lines.push(`Q: ${p.question}`);
    lines.push(`A: ${p.answer}`);
    lines.push('');
  }
  lines.push('When the user asks something matching one of the above questions, use that answer exactly.');
  lines.push('=================================');
  return lines.join('\n');
}

/** Lets the LLM answer naturally-phrased variants of "tell me about this
 * website"/location/hours/services/staff questions that website-profile-
 * fast-path.ts's literal pattern-match didn't catch — with zero tool call
 * involved, same "always-on injected context" shape as formatQnAContext
 * above. Only built when a profile actually exists (a never-crawled tenant
 * costs nothing extra). */
function formatWebsiteProfileContext(profile: import('../services/backend.client').WebsiteProfileSummary): string {
  const lines = ['=== BUSINESS PROFILE (from this tenant\'s own crawled website) ==='];
  if (profile.summary) lines.push(`Summary: ${profile.summary}`);
  if (profile.services?.length) lines.push(`Services: ${profile.services.join(', ')}`);
  if (profile.contact?.address) lines.push(`Address: ${profile.contact.address}`);
  if (profile.contact?.phone) lines.push(`Phone: ${profile.contact.phone}`);
  if (profile.contact?.email) lines.push(`Email: ${profile.contact.email}`);
  if (profile.hours) lines.push(`Hours: ${profile.hours}`);
  if (profile.staff?.length) {
    lines.push(`Team: ${profile.staff.map((s) => (s.title ? `${s.name} (${s.title})` : s.name)).join(', ')}`);
  }
  if (lines.length === 1) return ''; // profile doc exists but every field is empty
  lines.push('Use the above real facts when the visitor asks about the business, its services, location, hours, or team. Do not fabricate anything not listed above.');
  lines.push('=================================');
  return lines.join('\n');
}

/** Tells the LLM directly what's already known about this visitor's lead
 * capture, instead of making it re-derive that from re-reading the raw
 * transcript every turn — same "always-on injected context" shape as
 * formatWebsiteProfileContext/formatQnAContext above. Only built once at
 * least one field is already known (never injects an empty block on a
 * fresh conversation, so this costs nothing for the common early-turn
 * case). */
function formatLeadCaptureProgressContext(state: LeadCaptureState, booking?: BookingState | null): string {
  const known: string[] = [];
  if (state.firstName) known.push(`name=${state.firstName}${state.lastName ? ' ' + state.lastName : ''}`);
  if (state.email)     known.push(`email=${state.email}`);
  if (state.phone)     known.push(`phone=${state.phone}`);
  if (state.company)   known.push(`company=${state.company}`);
  if (state.service)   known.push(`service=${state.service}`);
  if (!known.length) return '';

  const missing: string[] = [];
  if (!state.firstName)             missing.push('name');
  if (!state.email && !state.phone) missing.push('email or phone');
  if (!state.service)               missing.push('service/reason for contact');

  const lines = ['=== INTERNAL NOTE TO YOU — LEAD CAPTURE PROGRESS (never show this note, its labels, or its contents to the visitor) ==='];
  lines.push(`ALREADY KNOWN: ${known.join(', ')}`);
  if (missing.length) lines.push(`STILL NEEDED: ${missing.join(', ')}`);
  if (booking?.meetingCreated) lines.push('A meeting is already booked for this visitor — do not offer to book again.');
  lines.push("Use this privately to decide what to ask next — ask only for what's marked STILL NEEDED, one thing at a time, never re-ask for something already known. Do NOT quote, paraphrase, format as a list, or otherwise reveal this note itself in your reply — just write a normal, natural sentence to the visitor.");
  lines.push('=================================');
  return lines.join('\n');
}

/* ── Extract the entity name from a user message for DB search ──────
   "2gb ram is there?"        → "2gb ram"
   "what is price of 2gb ram" → "2gb ram"
   "show all deals"           → "deals"
   Removes common question / filler words, keeps product/entity nouns. */
const SEARCH_NOISE = new Set([
  'is', 'are', 'there', 'what', 'price', 'of', 'do', 'you', 'have', 'show',
  'me', 'the', 'a', 'an', 'how', 'many', 'list', 'all', 'any', 'for', 'my',
  'our', 'tell', 'get', 'find', 'search', 'this', 'that', 'it', 'i', 'want',
  'need', 'can', 'check', 'about', 'give', 'with', 'which', 'when', 'where',
  'who', 'please', 'hi', 'hello', 'does', 'was', 'were', 'has', 'had', 'will',
  'would', 'could', 'should', 'not', 'in', 'and', 'or', 'from', 'to', 'by',
  'at', 'on', 'its', 'their', 'very', 'just', 'also', 'up', 'out', 'so',
  // Pronouns/articles missing from the original list
  'we', 'us', 'they', 'he', 'she', 'him', 'her', 'them', 'be', 'am', 'if',
  'go', 'got', 'get', 'set', 'put', 'let', 'may', 'few', 'too', 'via',
  // Comparison/filter operator words — user types "price less than 1000" not entity names
  'less', 'than', 'more', 'over', 'under', 'above', 'below', 'between', 'within',
  'lessthan', 'greaterthan', 'morethan', 'upto', 'atleast', 'atmost',
  // Deal/Opportunity stage values — these are filter targets, NOT entity names to search
  'proposal', 'negotiation', 'qualification', 'prospecting', 'analysis',
  'perception', 'identified', 'decision', 'makers',
  // Status/outcome values — filter targets only
  'won', 'lost', 'open', 'closed', 'sent', 'draft', 'paid', 'unpaid',
  'overdue', 'converted', 'active', 'inactive', 'pending', 'completed', 'done',
  // CRM entity-type nouns
  'product', 'products', 'item', 'items', 'invoice', 'invoices', 'invoiced',
  'deal', 'deals', 'contact', 'contacts', 'account', 'accounts', 'lead', 'leads',
  'record', 'records', 'order', 'orders', 'quote', 'quotes', 'task', 'tasks',
  'note', 'notes', 'report', 'reports', 'activity', 'activities',
  'meeting', 'meetings', 'call', 'calls', 'campaign', 'campaigns', 'vendor', 'vendors',
  'history', 'opportunity', 'opportunities',
  'data', 'info', 'information', 'detail', 'details',
  'available', 'stock', 'listed', 'named', 'called', 'type', 'crm',
  // CRM field-descriptor words — these describe fields/metrics, never entity names
  'title', 'field', 'value', 'label', 'column', 'property',
  'name', 'email', 'phone', 'company', 'address', 'status', 'stage',
  'source', 'owner', 'assigned', 'tag', 'tags', 'category', 'id', 'no',
  // Aggregate / financial descriptor words
  'total', 'amount', 'sum', 'average', 'avg', 'count', 'revenue', 'price',
  'cost', 'worth', 'combined', 'altogether', 'overall',
  // Date/time descriptor words — too generic to use as a Meilisearch entity search term
  'date', 'due', 'last', 'first', 'time', 'today', 'yesterday', 'week',
  'month', 'year', 'deadline', 'target', 'start', 'end', 'latest', 'recent',
  // Connector / context words
  'zoho', 'salesforce', 'hubspot', 'mongodb', 'mysql', 'postgresql',
  'customer', 'customers', 'user', 'users', 'person', 'people', 'client', 'clients',
  // Generic filler words users add when querying ("any related data is there")
  'related', 'relevant', 'associated', 'similar', 'specific', 'particular',
  'showing', 'show', 'see', 'view', 'know', 'current', 'existing', 'following',
  'here', 'now', 'yes', 'no', 'ok', 'okay', 'sure', 'please', 'thanks',
  'new', 'old', 'same', 'other', 'another', 'only', 'already', 'still',
  'correct', 'right', 'wrong', 'true', 'false', 'able', 'able',
]);

function extractSearchTerm(message: string): string {
  // 1. Email address — extract before any cleanup (dots in email would be stripped otherwise)
  const emailMatch = message.match(/[\w.+%-]+@[\w.-]+\.\w{2,}/);
  if (emailMatch) return emailMatch[0].trim();

  // 2. Phone number pattern — extract before cleanup
  const phoneMatch = message.match(/[+]?\d[\d\s\-().]{7,}/);
  if (phoneMatch) return phoneMatch[0].trim();

  // 3. General keyword extraction
  const words = message
    .toLowerCase()
    .replace(/[?!,]/g, '')          // strip sentence punctuation (NOT period — decimals, etc.)
    .replace(/\.(?=\s|$)/g, '')     // strip sentence-ending periods only
    .replace(/:/g, '')              // strip colons (e.g. "email id: value")
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !SEARCH_NOISE.has(w) && !/^\d+$/.test(w));
  return words.join(' ').trim().slice(0, 60);
}

/* ── Format DB search results as a compact context block ─────────────────
   Classify each field by its VALUE pattern, not its name.
   This works for ANY connector — Zoho, HubSpot, Salesforce, MySQL, PostgreSQL,
   MongoDB — regardless of what column/field names the data source uses.

   Priority:
     1 = email address value   (x@y.com)
     2 = phone number value    (555-555-5555 / +91 9876...)
     3 = important date field  (due_date, close_date, deadline, target_date)
     4 = price / numeric value where field name hints at money
     5 = short text (< 80 chars) — likely name / title / status
     6 = everything else
   ─────────────────────────────────────────────────────────────────────────── */
const EMAIL_RE  = /^[\w.+%-]+@[\w.-]+\.\w{2,}$/;
const PHONE_RE  = /^[+\d][\d\s\-().]{6,18}$/;
const MONEY_KEY = /price|total|amount|cost|rate|fee|revenue|value|charge|salary|mrp|discount/i;

// Fields whose names signal "important date" — due dates, deadlines, close dates
// These float above generic short text so they don't get cut by max-field limits
const KEY_DATE_KEY = /due|target|deadline|close_date|end_date|expir|deliver|complet|finish|start_date|duedate|closingdate/i;

function fieldPriority(key: string, value: unknown): number {
  const v = String(value).trim();
  if (EMAIL_RE.test(v))                                    return 1; // email address
  if (PHONE_RE.test(v) && /\d{6,}/.test(v))               return 2; // phone number
  if (KEY_DATE_KEY.test(key))                              return 3; // important dates (due, close, deadline)
  if (/^\d+(\.\d{1,2})?$/.test(v) && MONEY_KEY.test(key)) return 4; // price/amount
  if (v.length < 80)                                       return 5; // short label/text
  return 6;                                                           // long text / blob
}

function formatSearchResults(results: CRMSearchResult[], query: string): string {
  if (!results.length) return '';
  const lines = [`=== CRM SEARCH: "${query}" ===`];
  for (const r of results) {
    const isRecentlyViewed = r.module === 'RecentlyViewed' || r.module === 'Recently Viewed';
    lines.push(`[${r.channel.toUpperCase()} / ${r.module}] ${r.displayName}  (MODULE: ${r.module})`);
    const entries = Object.entries(r.data)
      .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== '')
      .sort(([ka, va], [kb, vb]) => fieldPriority(ka, va) - fieldPriority(kb, vb))
      .slice(0, 12);
    for (const [k, v] of entries) {
      lines.push(`  ${k}: ${v}`);
    }
    // Explicitly tell LLM when a record lacks contact fields
    if (isRecentlyViewed && entries.every(([k]) => !/email|phone|mobile/i.test(k))) {
      lines.push('  (Note: This is a recently-viewed activity record — email and phone are not stored here. Look for the contact in Contacts/Leads modules.)');
    }
  }
  lines.push('Use the data above to answer the question. Email/phone/price fields are listed first per record.');
  return lines.join('\n');
}

/* ── Extract email/phone from a CRM record data object ─────────────── */
function extractContactInfo(data: Record<string, unknown>): { email?: string; phone?: string } {
  let email: string | undefined;
  let phone: string | undefined;
  for (const [key, val] of Object.entries(data)) {
    const v = String(val ?? '').trim();
    if (!v || v === 'null' || v === 'undefined') continue;
    const kl = key.toLowerCase();
    if (!email && /^[\w.+%-]+@[\w.-]+\.\w{2,}$/.test(v)) { email = v; continue; }
    if (!phone && (kl.includes('phone') || kl.includes('mobile') || kl.includes('cell')) && /^[+\d][\d\s\-().]{6,}$/.test(v)) { phone = v; }
  }
  return { email, phone };
}

/* ── chrono-node date/time parser ──────────────────────────────────── */
function parseDateTimeWithChrono(rawDateTime: string, referenceDate = new Date()): { startDate: Date; endDate: Date } {
  const results = chrono.parse(rawDateTime, referenceDate, { forwardDate: true });

  if (results.length === 0) {
    const start = new Date(referenceDate);
    start.setDate(start.getDate() + 1);
    start.setHours(10, 0, 0, 0);
    const end = new Date(start);
    end.setHours(11, 0, 0, 0);
    return { startDate: start, endDate: end };
  }

  const parsed    = results[0];
  const startDate = parsed.start.date();
  startDate.setSeconds(0, 0);

  let endDate: Date;
  if (parsed.end) {
    endDate = parsed.end.date();
    endDate.setSeconds(0, 0);
  } else {
    const durMatch = rawDateTime.match(/(\d+(?:\.\d+)?)\s*(hr|hour|min|minute)/i);
    endDate = new Date(startDate);
    if (durMatch) {
      const val    = parseFloat(durMatch[1]);
      const isHour = /hr|hour/i.test(durMatch[2]);
      endDate.setMinutes(endDate.getMinutes() + (isHour ? val * 60 : val));
    } else {
      endDate.setHours(endDate.getHours() + 1);
    }
  }

  return { startDate, endDate };
}

/* ── Fuse.js fuzzy contact name matching ────────────────────────────── */
async function fuzzyMatchContact(rawName: string, tenantId: string): Promise<CRMSearchResult | null> {
  if (!rawName) return null;
  const candidates = await backendClient.searchCRMRecords(tenantId, rawName, 10);
  if (candidates.length === 0) return null;
  if (candidates.length === 1) return candidates[0];

  const fuse = new Fuse(candidates, {
    keys: ['displayName'],
    threshold: 0.4,
    includeScore: true,
  });
  const matches = fuse.search(rawName);
  return matches.length > 0 ? matches[0].item : candidates[0];
}

/* ── Format a date range for display in chat / email ─────────────────── */
function formatDateTimeRange(start: Date, end: Date): string {
  const dateStr  = start.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  const startStr = start.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
  const endStr   = end.toLocaleTimeString('en-US',   { hour: 'numeric', minute: '2-digit', hour12: true });
  return `${dateStr}, ${startStr} – ${endStr}`;
}

// The "LEAD CAPTURE PROGRESS" block injected into the system prompt (see
// formatLeadCaptureProgressContext) is meant for the model's own reference
// only — but small models (confirmed live, repeatedly, with the currently
// configured Groq model) sometimes echo its structure back to the visitor
// regardless of how bluntly the prompt tells them not to. Matches this
// file's own established philosophy of not trusting the LLM to reliably
// self-police (see the confidence gate below) — a deterministic strip is
// the real guarantee, the prompt instruction is just the first, cheaper
// line of defense. Strips from the first leaked marker line onward (that
// content is always trailing scaffolding, never the start of a reply).
const LEAKED_PROGRESS_MARKER_RE = /\n{0,2}(?:={3,}\s*)?(?:INTERNAL NOTE|LEAD CAPTURE PROGRESS)[\s\S]*$/i;
function stripLeakedProgressNotes(text: string): string {
  const cleaned = text.replace(LEAKED_PROGRESS_MARKER_RE, '').trim();
  return cleaned || text.trim();
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
    ),
  ]);
}

export async function runBaseAgent(input: AgentInput): Promise<AgentOutput> {
  try {
  // Public website-widget visitors (identified by visitorId) must never receive
  // live CRM-connector data (customer/deal/invoice records) — that stays exclusive
  // to the internal, staff-authenticated assistant. See the "Universal CRM search"
  // block and system-prompt assembly below, both gated on this flag.
  const isPublicVisitor = !!input.visitorId;
  const stageStart = Date.now();
  let guardrailsMs = 0;
  let ragMs = 0;

  /* ── Rate limit ── */
  const rateCheck = await checkTenantRateLimit(input.tenantId);
  if (!rateCheck.allowed) {
    return {
      response: "I'm a little busy right now — please try again in a moment.",
      escalate: false,
      capturedData: {},
    };
  }

  /* ── Prompt injection guard ── */
  const injectionCheck = detectPromptInjection(input.message);
  if (!injectionCheck.safe) {
    logger.warn('Prompt injection blocked', { tenantId: input.tenantId, reason: injectionCheck.reason });
    backendClient.writeLog({
      tenantId: input.tenantId, sessionId: input.sessionId,
      level: 'warn', event: 'guardrail.prompt_injection',
      message: 'Prompt injection attempt blocked',
      metadata: { reason: injectionCheck.reason, sessionId: input.sessionId },
    });
    backendClient.logSecurityEvent({
      event: 'ai.prompt_blocked', tenantId: input.tenantId,
      detail: { reason: injectionCheck.reason, sessionId: input.sessionId },
    });
    return {
      response: 'I can only assist with questions related to our services. How can I help you today?',
      escalate: false,
      capturedData: {},
    };
  }

  // guardrailsMs now covers only rate-limit + prompt-injection (both cheap,
  // synchronous-ish checks that should still gate everything else before
  // spending anything). Moderation moves below, into the same Promise.all
  // as RAG/tenant-context — it has no data dependency on either direction,
  // and gating the whole fetch behind it was pure wasted latency.
  guardrailsMs = Date.now() - stageStart;

  /* ── PII masking ── */
  const cleanMessage = maskPIIForLLM(input.message);

  /* ── Quick pre-screen: detect obviously generic messages before any heavy fetches ── */
  const FAST_PATH_TRIVIAL = /^(hi|hello|hey|bye|ok|okay|thanks|thank you|ty|sure|got it|how are you|what is your name|who are you|good morning|good evening|good night|night|later|goodbye|cheers|appreciate it|cool|great|perfect|noted|understood|alright|cya)\s*[!?.]*$/i;
  const isObviouslyGeneric = FAST_PATH_TRIVIAL.test(cleanMessage.trim()) || cleanMessage.trim().length <= 3;

  /* ── Fetch tenant context + run moderation concurrently — skip RAG/CRM for obviously generic messages ── */
  const ragStart = Date.now();
  const modStart = Date.now();
  let moderationMs = 0;
  const [modResult, ragResult, tenantConfig, history, prevCRMState, leadCaptureState, bookingState] = await Promise.all([
    moderateContent(input.message).finally(() => { moderationMs = Date.now() - modStart; }),
    isObviouslyGeneric
      ? Promise.resolve({ context: '', topScore: 0 })
      : buildRAGContextWithConfidence(cleanMessage, input.tenantId).catch(() => ({ context: '', topScore: 0 })).finally(() => { ragMs = Date.now() - ragStart; }),
    resolveTenantConfig(input.tenantId, {
      companyName: input.companyName,
      agentName:   input.agentName,
      language:    input.language,
    }),
    getHistory(input.sessionId),
    getCRMState(input.sessionId),
    getLeadCaptureState(input.sessionId),
    getBookingState(input.sessionId),
  ]);
  const ragContext = ragResult.context;
  const ragTopScore = ragResult.topScore;

  /* ── Content moderation result — checked here, after the concurrent fetch
     above, but still fully before anything is sent to an LLM. Disclosed
     tradeoff: the rare unsafe-message case now also pays for the RAG/tenant/
     history/CRM-state fetches above (previously skipped by returning early)
     — a few wasted calls in exchange for lower latency on every normal
     message. ── */
  if (modResult.usedFallback) {
    logger.warn('Moderation running in local-fallback mode', { tenantId: input.tenantId });
    backendClient.writeLog({
      tenantId: input.tenantId, sessionId: input.sessionId,
      level: 'warn', event: 'guardrail.moderation_fallback',
      message: 'OpenAI moderation unavailable — used local fallback check',
      metadata: { flagged: modResult.flagged, sessionId: input.sessionId },
    });
  }
  if (!modResult.safe) {
    logger.warn('Content flagged', { tenantId: input.tenantId, categories: modResult.categories });
    backendClient.writeLog({
      tenantId: input.tenantId, sessionId: input.sessionId,
      level: 'warn', event: 'guardrail.content_moderation',
      message: 'Message flagged by content moderation',
      metadata: { categories: modResult.categories, sessionId: input.sessionId },
    });
    return {
      response: "I'm unable to help with that. Is there something else I can assist you with?",
      escalate: false,
      capturedData: {},
    };
  }

  /* ── Deterministic booking-confirmation shortcut — checked BEFORE
     checkFastPath() on purpose: fast-path's own ACKNOWLEDGMENTS set already
     exact-matches bare replies like "ok"/"sure"/"perfect", which would
     otherwise swallow a visitor confirming a single offered slot before this
     ever runs. Public widget only, and a no-op for the overwhelming majority
     of turns (returns immediately whenever no slots are currently offered).
     See booking-confirmation-shortcut.ts for why this exists — book_meeting
     is nearly always the fragile SECOND tool call in a conversation. ── */
  if (isPublicVisitor) {
    const shortcut = await tryBookingConfirmationShortcut(
      cleanMessage,
      { tenantId: input.tenantId, sessionId: input.sessionId, visitorId: input.visitorId, pageUrl: input.pageUrl },
      bookingState ?? {},
      leadCaptureState ?? {},
    );
    if (shortcut.handled && shortcut.response) {
      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message,     timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: shortcut.response,  timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: shortcut.response });
      logger.info('Deterministic booking-confirmation shortcut handled this turn (no LLM used)', { tenantId: input.tenantId, sessionId: input.sessionId });
      return { response: shortcut.response, escalate: false, capturedData: extractCapturedData(input.message) };
    }
  }

  /* ── Fast-path: instant responses for greetings / off-topic — zero LLM cost.
     Crawled FAQPage entries are merged into the same qnaPairs array fed to
     the existing Fuse.js matcher — a crawled FAQ gets answered with zero new
     matching code, reusing 100% existing, proven infrastructure. ── */
  const mergedQnaPairs = tenantConfig.websiteProfile?.faqs?.length
    ? [...tenantConfig.qnaPairs, ...tenantConfig.websiteProfile.faqs.map((f) => ({ ...f, category: 'crawled-faq' }))]
    : tenantConfig.qnaPairs;
  const fast = checkFastPath(
    cleanMessage,
    tenantConfig.agentName,
    tenantConfig.companyName,
    tenantConfig.hasConnectors && !isPublicVisitor,
    mergedQnaPairs,
    tenantConfig.websiteProfile,
  );
  if (fast.handled && fast.response) {
    let fastResponse = fast.response;
    // Trailing nudge — shown once per session, only after an informational
    // (qna/profile) answer, not a greeting/farewell/off-topic reply, which
    // already has its own closing framing. Public widget only.
    if (isPublicVisitor && (fast.category === 'qna' || fast.category === 'profile')) {
      const leadState = leadCaptureState ?? {};
      if (!leadState.nudgeShown) {
        fastResponse += ' Would you like to book a time with our team, or is there anything else I can help with?';
        await setLeadCaptureState(input.sessionId, { ...leadState, nudgeShown: true });
      }
    }
    // Persist to Redis + MongoDB so chat history is complete
    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message,   timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: fastResponse,   timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: fastResponse });
    logger.info('Fast-path response (no LLM used)', { tenantId: input.tenantId, sessionId: input.sessionId, message: cleanMessage.slice(0, 60) });
    return { response: fastResponse, escalate: false, capturedData: extractCapturedData(input.message) };
  }

  /* ── Tenant token-quota gate — public widget only. Blocks the AI/LLM
     conversation once the tenant's monthly budget is exhausted, but the
     visitor can still leave contact details and become a real Lead — only
     the LLM reply stops, never lead capture. The internal, staff-
     authenticated assistant is never subject to this (isPublicVisitor is
     false there). Placed AFTER fast-path so free greeting/off-topic
     replies keep working even at 100% quota. ── */
  if (isPublicVisitor) {
    const quota = await checkTenantTokenQuota(input.tenantId, tenantConfig.monthlyTokenLimit);
    if (quota.blocked) {
      const capturedNow = extractCapturedData(input.message);
      void maybeCaptureWidgetLead(input, capturedNow);
      const reply = "Thanks for reaching out! Our AI assistant has reached its usage limit for now — please leave your name and email or phone number and our team will get back to you personally.";
      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: reply,          timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: reply });
      logger.info('Tenant AI token quota exhausted — LLM reply blocked, lead capture still attempted', { tenantId: input.tenantId, sessionId: input.sessionId });
      return { response: reply, escalate: false, capturedData: capturedNow };
    }
  }

  /* ── Early state handlers ─────────────────────────────────────────────
     Must run BEFORE the NLU LLM call so that "yes", "no", and customer-name
     replies are handled instantly without wasting a Groq round-trip. ── */

  const AFFIRM_RE = /^(yes|yeah|yep|sure|ok|okay|confirm|go ahead|proceed|do it|approved)[.!?]?\s*$/i;
  const DENY_RE   = /^(no|nope|cancel|stop|nevermind|never mind|don't|dont)[.!?]?\s*$/i;

  // Never for public widget visitors — these confirm/continue real CRM mutations
  // (reschedule/create a meeting, send email/SMS) started by the handlers below,
  // all of which are themselves disabled for public visitors (see isPublicVisitor
  // choke point further down). Guarding the read here too means a pre-existing
  // pendingIntent (e.g. still-live Redis state from before this fix shipped)
  // can never be acted on for a public session either.
  const earlyPending = isPublicVisitor ? null : await getPendingIntent(input.sessionId);

  /* ── Yes/No for reschedule confirmation ── */
  if (earlyPending?.pendingAction === 'await_reschedule_confirm') {
    if (AFFIRM_RE.test(cleanMessage.trim())) {
      await clearPendingIntent(input.sessionId);
      const startDate    = new Date(earlyPending.newStartIso!);
      const endDate      = new Date(earlyPending.newEndIso!);
      const dateRangeStr = formatDateTimeRange(startDate, endDate);

      await backendClient.updateActivity(input.tenantId, earlyPending.existingActivityId!, {
        startDate: earlyPending.newStartIso!,
        endDate:   earlyPending.newEndIso!,
        title:     earlyPending.existingTitle,
      });

      const custName = earlyPending.custName ?? '';
      let reschedCustomer: CRMSearchResult | null = null;
      if (custName) reschedCustomer = await fuzzyMatchContact(custName, input.tenantId);
      const { email: reschedEmail, phone: reschedPhoneRaw } = reschedCustomer ? extractContactInfo(reschedCustomer.data) : {};
      const reschedPhone = reschedPhoneRaw ? normalizePhone(reschedPhoneRaw) : undefined;

      let rEmailSent = false, rSmsSent = false;
      const rEmailSubject = `Meeting Rescheduled: ${dateRangeStr}`;
      const rEmailBody    = `<p>Hi <strong>${reschedCustomer?.displayName ?? 'there'}</strong>,</p><p>Your meeting has been rescheduled to <strong>${dateRangeStr}</strong>.</p><p>Best regards,<br/><strong>${tenantConfig.companyName}</strong></p>`;
      const rSmsText      = `Hi ${reschedCustomer?.displayName ?? 'there'}, your meeting with ${tenantConfig.companyName} has been rescheduled to ${dateRangeStr}.`;
      if (reschedEmail) {
        const r = await backendClient.sendEmail({
          tenantId: input.tenantId, toEmail: reschedEmail,
          toName:   reschedCustomer?.displayName ?? custName,
          subject:  rEmailSubject,
          body:     rEmailBody,
        });
        rEmailSent = r.success;
      }
      if (reschedPhone) {
        const r = await backendClient.sendSms({ to: reschedPhone, message: rSmsText });
        rSmsSent = r.success;
      }

      const rReply = [
        `Done! **${earlyPending.existingTitle}** rescheduled to **${dateRangeStr}**.`,
        rEmailSent ? `Rescheduling email sent to ${reschedEmail}.` : (!reschedEmail && custName ? `No email on file for ${custName} — update their CRM record.` : ''),
        rSmsSent   ? `SMS update sent to ${reschedPhone}.` : '',
      ].filter(Boolean).join('\n');

      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: rReply,         timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: rReply });
      return { response: rReply, escalate: false, capturedData: {} };
    }

    if (DENY_RE.test(cleanMessage.trim())) {
      await clearPendingIntent(input.sessionId);
      const cancelMsg = `No problem — **${earlyPending.existingTitle}** has not been changed. It stays on _${earlyPending.existingDateStr}_.`;
      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: cancelMsg,       timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: cancelMsg });
      return { response: cancelMsg, escalate: false, capturedData: {} };
    }
    // User typed something else while a reschedule confirmation was pending —
    // clear the stale state and continue with normal flow
    await clearPendingIntent(input.sessionId);
  }

  /* ── Handle customer reply after meeting creation ── */
  if (earlyPending?.pendingAction === 'await_customer' && earlyPending.activityId) {
    // If user typed a bare affirmation (yes/ok/sure) with no name content, re-ask
    const { name: parsedName, email: inlineEmail } = parseCustomerReply(cleanMessage.trim());
    if (!inlineEmail && AFFIRM_RE.test(parsedName.trim())) {
      const reask = `I still need to know who this meeting is with. Please share their **name** or **email address** so I can link them and send a confirmation.`;
      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: reask,          timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: reask });
      return { response: reask, escalate: false, capturedData: {} };
    }
    await clearPendingIntent(input.sessionId);

    // Search CRM by clean name only (not the full "name + email" string)
    let awaitCustCustomer: CRMSearchResult | null = parsedName ? await fuzzyMatchContact(parsedName, input.tenantId) : null;
    if (!awaitCustCustomer && parsedName) {
      const fallback = await backendClient.searchCRMRecords(input.tenantId, parsedName, 3);
      awaitCustCustomer = fallback[0] ?? null;
    }
    const { email: crmEmail, phone: crmPhone } = awaitCustCustomer ? extractContactInfo(awaitCustCustomer.data) : {};
    // Prefer inline email provided in the reply, fall back to CRM record
    const awaitEmail = inlineEmail ?? crmEmail;
    const rawPhone   = crmPhone;
    const awaitPhone = rawPhone ? normalizePhone(rawPhone) : undefined;
    const custInput  = parsedName || inlineEmail || cleanMessage.trim();

    if (awaitCustCustomer) {
      await backendClient.updateActivity(input.tenantId, earlyPending.activityId, {
        title: `Meeting with ${awaitCustCustomer.displayName}`,
        linkedPerson: {
          displayName: awaitCustCustomer.displayName,
          email:       awaitEmail,
          phone:       awaitPhone,
          module:      awaitCustCustomer.module,
          channel:     awaitCustCustomer.channel,
        },
      });
    }

    let acEmailSent = false, acSmsSent = false;
    const acEmailSubject = `Meeting Confirmed: ${earlyPending.dateRangeStr ?? 'your upcoming meeting'}`;
    const acEmailBody    = `<p>Hi <strong>${awaitCustCustomer?.displayName ?? 'there'}</strong>,</p><p>Your meeting with ${tenantConfig.companyName} is confirmed for <strong>${earlyPending.dateRangeStr ?? 'the scheduled time'}</strong>.</p><p>Best regards,<br/><strong>${tenantConfig.companyName}</strong></p>`;
    const acSmsText      = `Hi ${awaitCustCustomer?.displayName ?? custInput}, your meeting with ${tenantConfig.companyName} is confirmed for ${earlyPending.dateRangeStr ?? 'the scheduled time'}.`;

    if (awaitEmail) {
      const r = await backendClient.sendEmail({
        tenantId: input.tenantId, toEmail: awaitEmail,
        toName:   awaitCustCustomer?.displayName ?? custInput,
        subject:  acEmailSubject,
        body:     acEmailBody,
      });
      acEmailSent = r.success;
    }
    if (awaitPhone) {
      const r = await backendClient.sendSms({ to: awaitPhone, message: acSmsText });
      acSmsSent = r.success;
    }

    if (earlyPending.runId) {
      if (awaitCustCustomer) {
        await backendClient.updateAutomationStep(earlyPending.runId, 0, { status: 'success', result: `Found: ${awaitCustCustomer.displayName}`, customerEmail: awaitEmail, customerPhone: awaitPhone });
      }
      await backendClient.updateAutomationStep(earlyPending.runId, 2, {
        status: awaitEmail ? (acEmailSent ? 'success' : 'failed') : 'skipped',
        result: awaitEmail ? (acEmailSent ? `Sent to ${awaitEmail}` : 'Email failed') : 'No email on file',
        messageContent: awaitEmail ? { subject: acEmailSubject, body: acEmailBody, to: awaitEmail } : undefined,
      });
      await backendClient.updateAutomationStep(earlyPending.runId, 3, {
        status: awaitPhone ? (acSmsSent ? 'success' : 'failed') : 'skipped',
        result: awaitPhone ? (acSmsSent ? `SMS sent to ${awaitPhone}` : 'SMS failed') : 'No phone on file',
        runStatus: 'completed',
        messageContent: awaitPhone ? { text: acSmsText, to: awaitPhone } : undefined,
      });
    }

    const acReply = awaitCustCustomer
      ? [
          `Linked **${awaitCustCustomer.displayName}** to the meeting.`,
          acEmailSent ? `Confirmation email sent to ${awaitEmail}.` : (awaitEmail ? 'Email could not be sent — check Brevo config.' : `No email on file for ${awaitCustCustomer.displayName}. Add it in their CRM record.`),
          acSmsSent   ? `SMS confirmation sent to ${awaitPhone}.` : (awaitPhone ? 'SMS could not be sent — check Twilio config.' : ''),
        ].filter(Boolean).join('\n')
      : `I couldn't find **"${custInput}"** in your CRM. You can manually assign them from the Calendar. Or share their email address and I'll send the confirmation directly.`;

    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: acReply,         timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: acReply });
    return { response: acReply, escalate: false, capturedData: {} };
  }

  /* ── NLU: extract structured intent for automation tasks ────────────
     Only call the LLM if the message might be automation-related.
     This avoids an extra round-trip for pure CRM queries / general chat.
     Never for public widget visitors — this whole automation-intent path
     (schedule/reschedule a meeting, send an email) acts on real CRM contacts
     and must stay exclusive to the internal, staff-authenticated assistant
     (see the isPublicVisitor choke point below, which is the actual guarantee;
     skipping the call here is just avoiding wasted LLM cost, not the fix itself). ── */
  let chatIntent: ChatIntent | null = null;
  if (!isPublicVisitor && !isObviouslyGeneric && MIGHT_BE_AUTOMATION_RE.test(cleanMessage)) {
    chatIntent = await extractChatIntent(cleanMessage, history, tenantConfig.companyName);
  }

  // Check if user is replying to a pending clarification question from previous turn
  const pendingIntent = await getPendingIntent(input.sessionId);
  let resolvedPending: PendingIntent | null = null;

  if (pendingIntent) {
    const merged = mergePendingIntent(pendingIntent, chatIntent);
    if (merged.missingRequired.length === 0) {
      resolvedPending = merged;
      await clearPendingIntent(input.sessionId);
    } else {
      const question = merged.clarificationQuestion ?? `Could you provide the ${merged.missingRequired.join(' and ')}?`;
      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: question,       timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: question });
      await setPendingIntent(input.sessionId, merged);
      return { response: question, escalate: false, capturedData: {} };
    }
  } else if (chatIntent && chatIntent.confidence >= 0.75 && chatIntent.missingRequired.length > 0 &&
             (chatIntent.intent === 'schedule_meeting' || chatIntent.intent === 'reschedule_meeting')) {
    const question = chatIntent.clarificationQuestion ??
      `Could you provide the ${chatIntent.missingRequired.join(' and ')} for the ${chatIntent.intent.replace('_', ' ')}?`;
    await setPendingIntent(input.sessionId, {
      intent: chatIntent.intent,
      entities: chatIntent.entities,
      missingRequired: chatIntent.missingRequired,
      clarificationQuestion: chatIntent.clarificationQuestion,
    });
    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: question,       timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: question });
    return { response: question, escalate: false, capturedData: {} };
  }

  let activeIntent   = resolvedPending?.intent   ?? chatIntent?.intent;
  let activeEntities = resolvedPending?.entities ?? chatIntent?.entities ?? null;
  let activeConf     = resolvedPending ? 1.0 : (chatIntent?.confidence ?? 0);

  /* ── Safety-net: catch scheduling messages the LLM rated below threshold ──
     "ok, tomorrow 10 am to 11 am meeting schedule" → LLM may return low confidence
     because "ok" reads as an acknowledgement. We detect time + schedule words directly
     and force the schedule path so the user never gets a dead-end CRM response. ── */
  const SCHED_TIME_RE = /\b(tomorrow|today|next\s+\w+|\d{1,2}\s*(?:am|pm)|at\s+\d|\d:\d{2})\b/i;
  const SCHED_WORD_RE = /\b(meeting|appointment|call|booking|schedule|book|arrange)\b/i;
  if (!isPublicVisitor &&
      (!activeIntent || activeIntent === 'crm_query' || activeConf < 0.6) &&
      SCHED_TIME_RE.test(cleanMessage) && SCHED_WORD_RE.test(cleanMessage)) {
    activeIntent = 'schedule_meeting';
    activeConf   = 0.9;
    if (!activeEntities) {
      const rawDt = cleanMessage.replace(/^ok[!,.\s]*/i, '').trim() || cleanMessage;
      activeEntities = { person: chatIntent?.entities?.person ?? null, rawDateTime: rawDt, meetingType: null, notes: null };
    }
  }

  // Final choke point, placed after every possible source of activeIntent above
  // (resolvedPending, chatIntent, the regex safety-net) so nothing below this
  // line can ever run for a public widget visitor: reschedule_meeting/
  // schedule_meeting/send_email/etc. all read real CRM contacts and can create
  // real meetings or send real emails/SMS — must stay exclusive to the
  // internal, staff-authenticated assistant, same as the CRM-search block above.
  if (isPublicVisitor) { activeIntent = undefined; activeEntities = null; activeConf = 0; }

  /* ── Handle reschedule_meeting ── */
  if (activeIntent === 'reschedule_meeting' && activeConf >= 0.6) {
    const custName    = activeEntities?.person ?? '';
    const rawDt       = activeEntities?.rawDateTime ?? cleanMessage;
    const { startDate, endDate } = parseDateTimeWithChrono(rawDt);
    const dateRangeStr = formatDateTimeRange(startDate, endDate);

    // Find customer in CRM via fuzzy match
    let customer: CRMSearchResult | null = null;
    if (custName) customer = await fuzzyMatchContact(custName, input.tenantId);

    // Find existing meeting and ask for confirmation before updating
    let existingMeeting: { _id: string; title: string; startDate?: string; endDate?: string } | null = null;
    try {
      const actRes = await backendClient.listActivities(input.tenantId, { search: custName || '', limit: 5 });
      existingMeeting = actRes?.[0] ?? null;
    } catch { /* no existing activity found */ }

    if (existingMeeting?._id) {
      // Format existing time for display
      const existStart = existingMeeting.startDate ? new Date(existingMeeting.startDate) : null;
      const existEnd   = existingMeeting.endDate   ? new Date(existingMeeting.endDate)   : null;
      const existingDateStr = existStart
        ? formatDateTimeRange(existStart, existEnd ?? new Date(existStart.getTime() + 3600000))
        : 'unknown time';

      // Ask for confirmation — don't update yet
      await setPendingIntent(input.sessionId, {
        intent:              'reschedule_meeting',
        entities:            activeEntities ?? { person: null, rawDateTime: null, meetingType: null, notes: null },
        missingRequired:     [],
        clarificationQuestion: null,
        pendingAction:       'await_reschedule_confirm',
        existingActivityId:  existingMeeting._id,
        existingTitle:       existingMeeting.title,
        existingDateStr,
        newStartIso:         startDate.toISOString(),
        newEndIso:           endDate.toISOString(),
        custName:            customer?.displayName ?? custName,
      });

      const confirmAsk = `I found **"${existingMeeting.title}"** currently scheduled for _${existingDateStr}_.\n\nShould I reschedule it to **${dateRangeStr}**? Reply **yes** to confirm or **no** to cancel.`;
      await Promise.all([
        appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
        appendMessage(input.sessionId, { role: 'assistant', content: confirmAsk,     timestamp: Date.now() }),
      ]);
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
      backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: confirmAsk });
      return { response: confirmAsk, escalate: false, capturedData: {} };
    }

    // No existing meeting found — inform user
    const noMeetingMsg = `I couldn't find an existing meeting${custName ? ` with "${customer?.displayName ?? custName}"` : ''} to reschedule. Would you like me to **create a new meeting** for **${dateRangeStr}** instead? Reply yes or no.`;
    await setPendingIntent(input.sessionId, {
      intent: 'reschedule_meeting', entities: activeEntities ?? { person: null, rawDateTime: null, meetingType: null, notes: null },
      missingRequired: [], clarificationQuestion: null,
    });
    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: noMeetingMsg,  timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: noMeetingMsg });
    return { response: noMeetingMsg, escalate: false, capturedData: {} };
  }

  /* ── Handle schedule_meeting ── */
  if (activeIntent === 'schedule_meeting' && activeConf >= 0.6) {
    const custName = activeEntities?.person ?? '';
    const { startDate, endDate } = parseDateTimeWithChrono(activeEntities?.rawDateTime ?? cleanMessage);
    const dateRangeStr = formatDateTimeRange(startDate, endDate);

    // Create tracking record
    const runResult = await backendClient.createAutomationRun({
      tenantId:     input.tenantId,
      sessionId:    input.sessionId,
      trigger:      input.message,
      customerName: custName || 'Unknown',
      steps: [
        { name: 'Find Customer',  status: 'pending' },
        { name: 'Create Meeting', status: 'pending' },
        { name: 'Send Email',     status: 'pending' },
        { name: 'Send WhatsApp',  status: 'pending' },
      ],
    });
    const runId = runResult.runId ?? '';

    // Step 0: Find customer in CRM via fuzzy match
    let customer: CRMSearchResult | null = null;
    if (custName) customer = await fuzzyMatchContact(custName, input.tenantId);
    const { email: customerEmail, phone: customerPhoneRaw } = customer ? extractContactInfo(customer.data) : {};
    const customerPhone = customerPhoneRaw ? normalizePhone(customerPhoneRaw) : undefined;
    if (runId) {
      await backendClient.updateAutomationStep(runId, 0, {
        status:        custName ? (customer ? 'success' : 'failed') : 'skipped',
        result:        custName ? (customer ? `Found: ${customer.displayName}` : `"${custName}" not found in CRM`) : 'No customer name specified',
        customerEmail,
        customerPhone,
      });
    }

    // Step 1: Create calendar activity
    const meetingTitle = customer?.displayName
      ? `Meeting with ${customer.displayName}`
      : custName ? `Meeting with ${custName}` : 'Meeting';
    const activity = await backendClient.createActivity({
      tenantId:    input.tenantId,
      type:        'appointment',
      title:       meetingTitle,
      startDate:   startDate.toISOString(),
      endDate:     endDate.toISOString(),
      notes:       `Scheduled via AI chat: "${input.message}"`,
      linkedPerson: customer
        ? { displayName: customer.displayName, email: customerEmail, phone: customerPhone, module: customer.module, channel: customer.channel }
        : undefined,
    });
    if (runId) {
      await backendClient.updateAutomationStep(runId, 1, {
        status:     activity.success ? 'success' : 'failed',
        result:     activity.success ? `Meeting created: ${dateRangeStr}` : 'Failed to create activity',
        activityId: activity.activityId,
      });
    }

    // ── Find matching template by type (email vs sms/whatsapp) ──
    const findTemplateByChannel = (channel: 'email' | 'whatsapp' | 'sms', ref?: string) => {
      const search = (ref ?? 'meeting').toLowerCase();
      return tenantConfig.templates.find((t) =>
        t.type === channel &&
        (t.name.toLowerCase().includes(search) || t.category.toLowerCase().includes(search))
      );
    };
    const emailTemplate = findTemplateByChannel('email', activeEntities?.notes ?? 'meeting');

    const applyVars = (text: string) =>
      text
        .replace(/\{\{name\}\}/gi,    customer?.displayName ?? custName ?? 'Customer')
        .replace(/\{\{date\}\}/gi,    startDate.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }))
        .replace(/\{\{time\}\}/gi,    dateRangeStr)
        .replace(/\{\{company\}\}/gi, tenantConfig.companyName ?? '')
        .replace(/\{\{meeting\}\}/gi, meetingTitle);

    // Step 2: Send email confirmation
    let emailSent = false;
    let emailSubjectFinal = '';
    let emailBodyFinal    = '';
    if (customerEmail) {
      emailSubjectFinal = emailTemplate?.subject
        ? applyVars(emailTemplate.subject)
        : `Meeting Scheduled: ${dateRangeStr}`;
      emailBodyFinal = emailTemplate?.body
        ? applyVars(emailTemplate.body)
        : [
            `<p style="font-family:Arial,sans-serif">Hi <strong>${customer?.displayName ?? custName ?? 'there'}</strong>,</p>`,
            `<p>Your meeting has been scheduled for <strong>${dateRangeStr}</strong>.</p>`,
            `<p>If you have any questions, please don't hesitate to reach out.</p>`,
            `<p>Best regards,<br/><strong>${tenantConfig.companyName}</strong></p>`,
          ].join('');
      const emailResult = await backendClient.sendEmail({
        tenantId: input.tenantId,
        toEmail:  customerEmail,
        toName:   customer?.displayName ?? custName,
        subject:  emailSubjectFinal,
        body:     emailBodyFinal,
      });
      emailSent = emailResult.success;
    }
    if (runId) {
      await backendClient.updateAutomationStep(runId, 2, {
        status: customerEmail ? (emailSent ? 'success' : 'failed') : 'skipped',
        result: customerEmail
          ? (emailSent ? `Sent to ${customerEmail}` : 'Brevo delivery failed')
          : 'No email address on file',
        messageContent: customerEmail ? { subject: emailSubjectFinal, body: emailBodyFinal, to: customerEmail } : undefined,
      });
    }

    // Step 3: Send SMS/WhatsApp
    let smsSent = false;
    let smsFinalText = '';
    if (customerPhone) {
      const smsTemplate =
        findTemplateByChannel('whatsapp', activeEntities?.notes ?? 'meeting') ??
        findTemplateByChannel('sms',      activeEntities?.notes ?? 'meeting');
      smsFinalText = smsTemplate?.body
        ? applyVars(smsTemplate.body)
        : `Hi ${customer?.displayName ?? custName ?? 'there'}, your meeting with ${tenantConfig.companyName} is scheduled for ${dateRangeStr}. See you then!`;
      const smsResult = await backendClient.sendSms({ to: customerPhone, message: smsFinalText });
      smsSent = smsResult.success;
    }
    if (runId) {
      await backendClient.updateAutomationStep(runId, 3, {
        status:    customerPhone ? (smsSent ? 'success' : 'failed') : 'skipped',
        result:    customerPhone
          ? (smsSent ? `SMS sent to ${customerPhone}` : 'Twilio delivery failed')
          : 'No phone number on file',
        runStatus: customer ? 'completed' : (custName ? 'partial' : 'completed'),
        messageContent: customerPhone ? { text: smsFinalText, to: customerPhone } : undefined,
      });
    }

    // Build reply — if no customer, ask who it's with so we can send email/SMS
    const replyParts: string[] = [
      `Meeting **${meetingTitle}** scheduled for **${dateRangeStr}**.`,
    ];
    if (emailTemplate) replyParts.push(`Used template: _${emailTemplate.name}_.`);
    if (customerEmail) replyParts.push(emailSent ? `Confirmation email sent to ${customerEmail}.` : 'Email confirmation could not be sent (check Brevo config).');
    if (customerPhone) replyParts.push(smsSent ? `WhatsApp/SMS sent to ${customerPhone}.` : 'SMS notification could not be sent (check Twilio config).');
    if (!customer && custName) replyParts.push(`Note: "${custName}" was not found in CRM — the meeting was still created.`);

    let needsCustomerFollowUp = false;
    if (!customer) {
      // Store pending state so next message is treated as customer name/email
      await setPendingIntent(input.sessionId, {
        intent: 'schedule_meeting',
        entities: { person: null, rawDateTime: null, meetingType: null, notes: null },
        missingRequired: [], clarificationQuestion: null,
        pendingAction: 'await_customer',
        activityId:   activity.activityId,
        dateRangeStr,
        runId,
      });
      replyParts.push('\n**Who is this meeting with?** Reply with their name or email and I\'ll link them in CRM and send a confirmation email + SMS right away.');
      needsCustomerFollowUp = true;
    }
    if (!needsCustomerFollowUp) replyParts.push('Track this in the **Automation** tab · View appointment in your **Calendar**.');
    const reply = replyParts.join('\n');

    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: reply,          timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: reply });
    void backendClient.logAIAction({
      tenantId:    input.tenantId,
      sessionId:   input.sessionId,
      actionType:  'schedule_meeting',
      summary:     `Meeting scheduled with ${customer?.displayName ?? custName} for ${dateRangeStr}`,
      userMessage: input.message.slice(0, 300),
      metadata:    { customerName: customer?.displayName, startDate: startDate.toISOString(), endDate: endDate.toISOString(), runId },
    });
    return { response: reply, escalate: false, capturedData: {} };
  }

  /* ── Handle send_email ── */
  if (activeIntent === 'send_email' && activeConf >= 0.6) {
    const custName = activeEntities?.person ?? '';
    const template = tenantConfig.templates.find((t) => t.category === 'followup');
    const templateInfo = template
      ? `I have your "${template.name}" template ready to use.`
      : "I'll compose a professional follow-up email.";
    const nameInfo = custName ? ` for ${custName}` : '';
    const confirmMsg = `I'll send a follow-up email${nameInfo}. ${templateInfo} Could you confirm their email address so I can send it right away?`;
    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: confirmMsg,     timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'user',      content: input.message });
    backendClient.saveChatMessage({ tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: confirmMsg });
    return { response: confirmMsg, escalate: false, capturedData: {} };
  }

  /* ── CRM intent detection ── */
  const crmIntent = detectCRMIntent(
    cleanMessage,
    tenantConfig.crmModules,
    tenantConfig.hasConnectors,
    '', // inline records no longer pre-injected — we search dynamically
    prevCRMState?.channel // prefer same connector as previous turn for follow-ups
  );

  // Detect numeric filter BEFORE deciding whether to reuse cache.
  // "price less than 1000" → filter in code, not in LLM (LLMs are bad at math).
  const numericFilter   = extractNumericFilter(input.message);
  const stringFilter    = !numericFilter ? extractStringFilter(input.message) : null;
  const aggregateIntent = !numericFilter && !stringFilter ? detectAggregateIntent(input.message) : null;

  const isFollowUpCRM =
    !numericFilter &&          // always re-fetch when user adds a numeric filter condition
    !stringFilter &&           // always re-fetch for status/stage filter queries
    !crmIntent.isCRMQuery &&
    !crmIntent.explicitChannel && // don't reuse cache when user redirects to a different channel
    prevCRMState?.mode === 'crm' &&
    cleanMessage.split(' ').length <= 12;

  /* ── Universal CRM search ───────────────────────────────────────────
     When the tenant has connectors, ALWAYS search MongoDB for the entity
     the user mentioned. We don't rely on CRM intent detection — if data
     is found it gets injected regardless of what mode the bot is in.
     This ensures any question about any record (product, contact, deal,
     invoice) returns real data from DB without needing perfect phrasing.

     NEVER for public-widget visitors (isPublicVisitor) — this block reads
     real customer/deal/invoice records, which must stay exclusive to the
     internal, staff-authenticated assistant. */
  let crmDataBlock = '';

  if (tenantConfig.hasConnectors && !isPublicVisitor) {
    if (isFollowUpCRM && prevCRMState?.recordsBlock) {
      // Reuse last turn's search results for short follow-ups ("what is the price?")
      crmDataBlock = prevCRMState.recordsBlock;
    } else {
      // Use the ORIGINAL message (not PII-masked cleanMessage) so emails/phones
      // are not replaced with [EMAIL]/[PHONE] before the DB search runs.
      // PII masking is only for the LLM prompt, not for search term extraction.
      const searchTerm = extractSearchTerm(input.message);

      // Skip Meilisearch when the user is filtering a known module (numeric or status/stage filter).
      // Meilisearch searches ALL modules — for "deals above 50000" it returns DealHistory records
      // containing those amounts, contaminating the response. Filter queries must go directly to
      // fetchAndFilter() which uses the correct channel+module from intent detection.
      const hasFilterWithModule =
        (numericFilter !== null || stringFilter !== null) &&
        crmIntent.isCRMQuery &&
        crmIntent.channel != null &&
        crmIntent.module != null;

      if (searchTerm.length >= 3 && !hasFilterWithModule) {
        let searchResults = await backendClient.searchCRMRecords(input.tenantId, searchTerm);
        if (searchResults.length > 0) {
          // If all results are from "Recently Viewed" (no email/phone fields), also fetch
          // from Contacts/Leads modules directly so the LLM gets real contact data.
          const allRecentlyViewed = searchResults.every(r =>
            r.module === 'RecentlyViewed' || r.module === 'Recently Viewed'
          );
          if (allRecentlyViewed) {
            const contactSearch = await backendClient.searchCRMRecords(input.tenantId, searchTerm + ' contacts', 6);
            if (contactSearch.length > 0) searchResults = [...contactSearch, ...searchResults];
          }
          crmDataBlock = formatSearchResults(searchResults, searchTerm);
        }
      }

      // Helper: fetch records, apply numeric filter in code, format for LLM
      const fetchAndFilter = async (
        channel: string,
        module: string
      ): Promise<string> => {
        const raw = await backendClient.getCRMRecords(
          input.tenantId, channel, module, { limit: 50 }
        ) as CRMRecord[];
        if (!raw.length) return '';

        const cachedCount = tenantConfig.crmModules[channel]?.find((m) => m.module === module)?.count;
        // Use whichever is higher — cached count may be stale if records were added after last sync
        const totalInDB = cachedCount !== undefined ? Math.max(raw.length, cachedCount) : raw.length;
        const modDisplay = moduleDisplayName(module);

        if (numericFilter) {
          const filtered = applyNumericFilter(raw, numericFilter);
          const label = filterLabel(numericFilter);
          if (!filtered.length) {
            return `(Filter: ${label} — 0 of ${raw.length} ${modDisplay} match this condition)\nNo records found matching this filter.`;
          }
          const header = `(Filter: ${label} — ${filtered.length} of ${raw.length} ${modDisplay} match)\n`;
          return header + formatRecordsForLLM(filtered, module, channel, filtered.length, true);
        }

        if (stringFilter) {
          const filtered = applyStringFilter(raw, stringFilter);
          if (filtered.length > 0) {
            const header = `(Filter: "${stringFilter.value}" — ${filtered.length} of ${raw.length} ${modDisplay} match)\n`;
            return header + formatRecordsForLLM(filtered, module, channel, filtered.length, true);
          }
          // 0 code-level matches (field may have different name) — send all to LLM (Rule 6)
        }

        // Pre-compute aggregate server-side so LLM never has to do arithmetic
        if (aggregateIntent) {
          const values = raw
            .map((r) => getNumericFieldValue(r.data))
            .filter((v): v is number => v !== null && v > 0);
          if (values.length > 0) {
            const total = values.reduce((s, v) => s + v, 0);
            const aggPrefix = aggregateIntent === 'sum'
              ? `[PRE-COMPUTED] Total ${modDisplay} Amount: ${total} (across ${values.length} records)\n\n`
              : `[PRE-COMPUTED] Average ${modDisplay} Amount: ${Math.round(total / values.length)} (across ${values.length} records)\n\n`;
            return aggPrefix + formatRecordsForLLM(raw, module, channel, totalInDB);
          }
        }

        return formatRecordsForLLM(raw, module, channel, totalInDB);
      };

      // Fallback 1: intent detected specific module → fetch it directly
      if (!crmDataBlock && crmIntent.isCRMQuery && crmIntent.channel && crmIntent.module) {
        crmDataBlock = await fetchAndFilter(crmIntent.channel, crmIntent.module);
      }

      // Fallback 2: searchTerm was empty (all words stripped as noise) AND no module from intent
      // → try matching message words against real module names in the DB
      // e.g. "show me all deals" → strips to "" → but "deals" matches "Deals" module
      if (!crmDataBlock && crmIntent.isCRMQuery) {
        const preferCh = crmIntent.explicitChannel || prevCRMState?.channel;
        const matched = findBestModule(input.message, tenantConfig.crmModules, preferCh);
        if (matched) {
          crmDataBlock = await fetchAndFilter(matched.channel, matched.module);
        }
      }

      // Channel redirect: "in zoho" after Salesforce campaigns were shown → re-fetch from new channel
      // detectCRMIntent returns explicitChannel even when isCRMQuery=false, enabling this redirect
      if (!crmDataBlock && crmIntent.explicitChannel && prevCRMState?.module &&
          crmIntent.explicitChannel !== prevCRMState.channel) {
        crmDataBlock = await fetchAndFilter(crmIntent.explicitChannel, prevCRMState.module);
      }

      // If search found nothing but the previous turn had records, reuse them.
      // This fixes follow-up questions like "what is this due date?" where the
      // search term is empty but the user is clearly asking about the last result.
      if (!crmDataBlock && prevCRMState?.mode === 'crm' && prevCRMState?.recordsBlock) {
        crmDataBlock = prevCRMState.recordsBlock;
      }

      // Cache for follow-up turns
      if (crmDataBlock || crmIntent.isCRMQuery) {
        await setCRMState(input.sessionId, {
          mode:         'crm',
          channel:      crmIntent.channel ?? crmIntent.explicitChannel,
          module:       crmIntent.module  ?? prevCRMState?.module,
          recordsBlock: crmDataBlock,
        });
      } else if (prevCRMState?.mode === 'crm') {
        void setCRMState(input.sessionId, { mode: 'lead' });
      }
    }
  }

  /* ── Build Q&A block (trained answers — LLM uses these first) ── */
  const qnaBlock = formatQnAContext(tenantConfig.qnaPairs);

  /* ── Build system prompt ── */
  let systemContent: string;

  // Use CRM analyst prompt only when we have actual DB search results.
  // When crmDataBlock is empty (no match found), fall through to lead/general mode
  // so the bot doesn't hallucinate from the CRM overview or customer list.
  // Every branch below is additionally gated on !isPublicVisitor — public widget
  // visitors must never see crmDataBlock/crmContext/customerContext, regardless
  // of what crmIntent/isFollowUpCRM computed (belt-and-suspenders alongside the
  // search block above already being skipped for them).
  if (!isPublicVisitor && (crmDataBlock || isFollowUpCRM)) {
    systemContent = buildCRMQueryPrompt({
      companyName: tenantConfig.companyName,
      agentName:   tenantConfig.agentName,
      language:    tenantConfig.language,
      // Pass only the search results as crmContext — the prompt will label it clearly
      crmContext:  tenantConfig.crmContext + '\n\n' + crmDataBlock,
      // customerContext deliberately NOT passed — it causes LLM to misuse contact names
    });
  } else if (!isPublicVisitor && crmIntent.isCRMQuery) {
    // Explicit CRM query but search found nothing — tell LLM to say so clearly
    systemContent = buildCRMQueryPrompt({
      companyName: tenantConfig.companyName,
      agentName:   tenantConfig.agentName,
      language:    tenantConfig.language,
      crmContext:  tenantConfig.crmContext, // overview only — no records
    });
  } else {
    // Lead capture / visitor mode
    systemContent = input.systemPrompt;
    if (tenantConfig.systemPrompt) systemContent = tenantConfig.systemPrompt + '\n\n' + systemContent;
    if (!isPublicVisitor) {
      if (tenantConfig.crmContext)      systemContent += '\n\n' + tenantConfig.crmContext;
      if (tenantConfig.customerContext) systemContent += '\n\n' + tenantConfig.customerContext;
    } else {
      systemContent += '\n\n' + [
        'You have tools available: search_website_knowledge (search this business\'s own site for an answer),',
        'check_meeting_availability (get REAL open times — never guess or invent times yourself),',
        'and book_meeting (confirm a booking on one of the times check_meeting_availability just returned).',
        'When a tool returns real data, base your reply directly on that data — do not ignore it or ask a generic clarifying question instead.',
        'After answering a factual question, naturally suggest a next step (booking a time, or asking if there\'s anything else) — without being pushy or repeating it if you already have.',
        ...(tenantConfig.hasWidgetDepartments
          ? [
              'This business has multiple departments/doctors. Before checking availability for a booking, call list_departments first.',
              'If it returns real departments, ask the visitor which one, then call list_doctors for that department, ask which doctor they\'d like,',
              'and pass that doctor\'s staffId into check_meeting_availability and book_meeting. If list_departments returns none, proceed exactly as normal.',
            ]
          : []),
      ].join(' ');
      if (tenantConfig.websiteProfile) {
        const profileBlock = formatWebsiteProfileContext(tenantConfig.websiteProfile);
        if (profileBlock) systemContent += '\n\n' + profileBlock;
      }
      const progressBlock = formatLeadCaptureProgressContext(leadCaptureState ?? {}, bookingState);
      if (progressBlock) systemContent += '\n\n' + progressBlock;
    }
  }

  // Inject Q&A trained answers (before RAG so LLM sees them first)
  if (qnaBlock) systemContent = qnaBlock + '\n\n' + systemContent;

  // Append RAG knowledge base context
  if (ragContext) systemContent += '\n\n' + ragContext;

  /* ── Build message array ── */
  const messages: LLMMessage[] = [
    { role: 'system', content: systemContent },
    ...history.map((h) => ({ role: h.role as 'user' | 'assistant', content: h.content })),
    { role: 'user', content: input.message },
  ];

  /* ── LLM call with timeout — tool-bound for surfaces that have tools ──
     internal_staff has zero tools in this phase, so this is always the
     plain llm.generate() path for the existing internal assistant — no
     behavior change there. ── */
  const bookingOnly = isPublicVisitor && isBookingOnlyMessage(cleanMessage);
  const surfaceTools = getToolsForSurface(isPublicVisitor ? 'public_widget' : 'internal_staff', { bookingOnly });
  let toolCallsLog: ToolCallLog[] = [];
  let responseUsage: UsageMetadata | undefined;
  const llmStart = Date.now();
  try {
    let responseContent: string;
    let responseProvider: string;
    let responseModel: string;

    if (surfaceTools.length) {
      const toolResult = await withTimeout(
        runToolLoop({
          messages,
          tools: surfaceTools,
          surface: isPublicVisitor ? 'public_widget' : 'internal_staff',
          ctx: {
            tenantId: input.tenantId,
            sessionId: input.sessionId,
            visitorId: input.visitorId,
            pageUrl: input.pageUrl,
            companyName: tenantConfig.companyName,
            timezone: 'UTC',
          },
          toolModelOverride: tenantConfig.toolModelPreset ? TOOL_MODEL_PRESETS[tenantConfig.toolModelPreset] : undefined,
        }),
        TOOL_LOOP_TIMEOUT_MS,
        'LLM tool loop'
      );
      responseContent  = toolResult.content;
      responseProvider = toolResult.provider;
      responseModel    = toolResult.model;
      toolCallsLog      = toolResult.toolCalls;
      responseUsage     = toolResult.usage;
    } else {
      const plainResult = await withTimeout(llm.generate(messages), LLM_TIMEOUT_MS, 'LLM generate');
      responseContent  = plainResult.content;
      responseProvider = plainResult.provider;
      responseModel    = plainResult.model;
      responseUsage    = plainResult.usage;
    }

    const llmMs = Date.now() - llmStart;
    const result = { content: responseContent, provider: responseProvider, model: responseModel };
    let response = stripLeakedProgressNotes(result.content);
    const estimatedCost = responseUsage
      ? estimateCostUsd(result.provider, result.model, responseUsage.input_tokens, responseUsage.output_tokens)
      : undefined;

    if (responseUsage) {
      backendClient.trackAiTokenUsage({
        tenantId: input.tenantId,
        promptTokens: responseUsage.input_tokens,
        completionTokens: responseUsage.output_tokens,
        totalTokens: responseUsage.total_tokens,
        estimatedCostUsd: estimatedCost ?? 0,
        usedModerationFallback: modResult.usedFallback,
      });
    }

    /* ── Confidence gate — public widget only. Rather than trust the LLM to
       self-report its own uncertainty (confirmed unreliable via live
       testing), compute confidence deterministically from real signals: did
       a product-catalog tool actually match something, did RAG's own
       similarity score clear a real bar. If a reply was attempted without
       either, override it with a human-handoff message — which also
       contains "connect you with", so the existing shouldEscalate() below
       picks it up automatically, with no separate escalation code needed. ── */
    // Did this look like a profile-shaped question ("tell me about this
    // website", "where are you located"...) AND does the tenant actually
    // have real, non-empty profile data — the case where the LLM answered
    // via formatWebsiteProfileContext's always-on injection above rather
    // than a tool call, so toolCallsLog alone wouldn't reveal it.
    const hasNonEmptyProfile = !!tenantConfig.websiteProfile && Object.values(tenantConfig.websiteProfile).some(
      (v) => (Array.isArray(v) ? v.length > 0 : v && typeof v === 'object' ? Object.keys(v).length > 0 : !!v)
    );
    const isProfileShapedQuestion = looksLikeProfileQuestion(cleanMessage);
    const hasProfileMatch = hasNonEmptyProfile && isProfileShapedQuestion;
    // A profile-shaped question counts as "attempted grounding" even when it
    // scored zero RAG similarity and called no tool — otherwise a never-
    // crawled tenant asked "tell me about your business" skips this gate
    // entirely (no score, no tool call) and the LLM free-associates using
    // only companyName instead of admitting it doesn't know yet.
    const attemptedGrounding = ragTopScore > 0 || toolCallsLog.length > 0 || isProfileShapedQuestion;
    const { source: responseSource, confidence: responseConfidence } = classifyResponseConfidence(ragTopScore, toolCallsLog, hasProfileMatch);
    if (isPublicVisitor && attemptedGrounding && responseConfidence < CONFIDENCE_THRESHOLD) {
      response = "That's a good question, but I don't have a confident answer from our records — let me connect you with our team so they can help directly. Could I get your name and best way to reach you?";
    }

    /* ── Persist to Redis conversation memory + MongoDB chat session ── */
    const escalate    = shouldEscalate(response, input.message);
    const capturedNow = extractCapturedData(input.message);
    const capturedData = capturedNow;

    await Promise.all([
      appendMessage(input.sessionId, { role: 'user',      content: input.message, timestamp: Date.now() }),
      appendMessage(input.sessionId, { role: 'assistant', content: response,       timestamp: Date.now() }),
    ]);
    backendClient.saveChatMessage({
      tenantId: input.tenantId, sessionId: input.sessionId, role: 'user', content: input.message,
      visitorName: capturedNow.name, visitorEmail: capturedNow.email, visitorPhone: capturedNow.phone,
    });
    backendClient.saveChatMessage({
      tenantId: input.tenantId, sessionId: input.sessionId, role: 'assistant', content: response,
      metadata: { provider: result.provider, model: result.model },
      escalated: escalate || undefined,
    });

    logger.info('Agent response generated', {
      tenantId:  input.tenantId,
      sessionId: input.sessionId,
      provider:  result.provider,
      model:     result.model,
      escalate,
      hasConnectors: tenantConfig.hasConnectors,
      captured: Object.keys(capturedData),
    });

    // Persist log to MongoDB (non-blocking)
    backendClient.writeLog({
      tenantId:  input.tenantId,
      sessionId: input.sessionId,
      level:     escalate ? 'warn' : 'info',
      event:     escalate ? 'agent.escalation' : 'agent.response',
      message:   escalate
        ? `AI escalated session to human agent`
        : `AI responded successfully`,
      metadata: {
        sessionId:     input.sessionId,
        provider:      result.provider,
        model:         result.model,
        escalate,
        capturedFields: Object.keys(capturedData),
        hasConnectors:  tenantConfig.hasConnectors,
        promptTokens:     responseUsage?.input_tokens,
        completionTokens: responseUsage?.output_tokens,
        totalTokens:      responseUsage?.total_tokens,
        estimatedCostUsd: estimatedCost,
        stageTimingsMs:   { guardrails: guardrailsMs, moderation: moderationMs, rag: ragMs, llm: llmMs },
        responseSource:     attemptedGrounding ? responseSource : undefined,
        responseConfidence: attemptedGrounding ? responseConfidence : undefined,
      },
    });

    // ── AI Action log (fire-and-forget, never blocks response) ──────────
    try {
      let actionType = 'general';
      let actionSummary = 'Chat response';
      const actionMeta: Record<string, unknown> = {
        promptTokens:     responseUsage?.input_tokens,
        completionTokens: responseUsage?.output_tokens,
        totalTokens:      responseUsage?.total_tokens,
        estimatedCostUsd: estimatedCost,
        responseSource:     attemptedGrounding ? responseSource : undefined,
        responseConfidence: attemptedGrounding ? responseConfidence : undefined,
      };

      if (capturedNow.email || capturedNow.phone || capturedNow.name) {
        actionType    = 'lead_capture';
        actionSummary = `Lead captured${capturedNow.name ? ': ' + capturedNow.name : ''}`;
        actionMeta.leadName  = capturedNow.name;
        actionMeta.leadEmail = capturedNow.email;
        actionMeta.leadPhone = capturedNow.phone;
      } else if (escalate) {
        actionType    = 'escalation';
        actionSummary = 'Session escalated to human agent';
      } else if (crmDataBlock) {
        // Parse the CRM data block header to extract channel / module / counts
        const isFilter   = crmDataBlock.includes('Filter:');
        const dataHeader = crmDataBlock.match(/CRM DATA:\s*(.+?) from ([A-Z]+)\s*\((\d+)/);
        const filterInfo = crmDataBlock.match(/Filter:\s*(.+?)\s*—\s*(\d+) of (\d+)/);
        actionType       = isFilter ? 'crm_filter' : 'crm_query';
        if (dataHeader) {
          actionMeta.module  = dataHeader[1].trim();
          actionMeta.channel = dataHeader[2].toLowerCase();
          actionMeta.recordCount = parseInt(dataHeader[3], 10);
        }
        if (filterInfo) {
          actionMeta.filterExpression = filterInfo[1].trim();
          actionMeta.filteredCount    = parseInt(filterInfo[2], 10);
        }
        actionSummary = actionMeta.module
          ? `${isFilter ? 'Filtered' : 'Queried'} ${actionMeta.filteredCount ?? actionMeta.recordCount ?? '?'} ${actionMeta.module} (${String(actionMeta.channel ?? '').toUpperCase()})`
          : 'CRM data query';
      } else if (ragContext) {
        actionType    = 'knowledge_query';
        actionSummary = 'Knowledge base query';
      } else if (toolCallsLog.length > 0) {
        actionType    = 'tool_call';
        actionSummary = `Called ${toolCallsLog.map((t) => t.name).join(', ')}`;
      }
      if (toolCallsLog.length > 0) actionMeta.toolCalls = toolCallsLog;

      void backendClient.logAIAction({
        tenantId:    input.tenantId,
        sessionId:   input.sessionId,
        actionType,
        summary:     actionSummary,
        userMessage: input.message.slice(0, 300),
        metadata:    actionMeta,
      });
    } catch { /* logging must never crash the agent */ }
    // ─────────────────────────────────────────────────────────────────────

    // Fire-and-forget — the visible reply text above is already fully
    // computed and persisted by this point, so this background bookkeeping
    // step cannot change what the visitor sees this turn. Matches this
    // file's own established fire-and-forget idiom (void fn(), see
    // logAIAction above), rather than blocking the HTTP response on an
    // extra LLM call the visitor gets no benefit from waiting on.
    void maybeCaptureWidgetLead(input, capturedNow);

    return { response, escalate, capturedData };
  } catch (err) {
    logger.error('Agent generation error', { error: (err as Error).message, tenantId: input.tenantId });
    return {
      response: "I'm having trouble connecting right now. Please try again in a moment.",
      escalate: false,
      capturedData: {},
    };
  }
  } catch (err) {
    logger.error('Agent run error', { error: (err as Error).message, tenantId: input.tenantId });
    return {
      response: "Something went wrong on my end. Please try again.",
      escalate: false,
      capturedData: {},
    };
  }
}
