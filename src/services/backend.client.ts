import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import { logger } from '../utils/logger';
import {
  getCachedTenantContext, setCachedTenantContext,
  getCachedCatalogSearch, setCachedCatalogSearch,
  getCachedCatalogItem, setCachedCatalogItem,
  getCachedWidgetTeams, setCachedWidgetTeams,
  getCachedWidgetStaff, setCachedWidgetStaff,
} from '../memory/conversation.memory';
import type { QueryPlan, DatasetSchemaContext } from '../query-router/types';

/* ── Types matching the backend's internal endpoint response ── */
export interface TenantAIConfig {
  systemPrompt?: string;
  language?: string;
  fallbackToHuman?: boolean;
  agentName?: string;
  monthlyTokenLimit?: number;
  /** Monthly minute budget for CONTINUOUS voice conversations specifically
   * (LiveKit room-minutes) — a separate meter from monthlyTokenLimit above.
   * See tenant.model.ts's own field comment for the full rationale. */
  monthlyVoiceMinutesLimit?: number;
  /** Which already-integrated LLM provider/model powers RAG/catalog/booking
   * tool-calling for this tenant specifically — undefined means "use the
   * global primary/fallback pair", today's unchanged default. See
   * config/index.ts's TOOL_MODEL_PRESETS for what each preset resolves to. */
  toolModelPreset?: 'groq' | 'anthropic' | 'openai' | 'google';
}

export interface TenantBranding {
  logoUrl?: string;
  primaryColor?: string;
  companyName?: string;
}

export interface TenantSettings {
  allowedChannels?: string[];
  timezone?: string;
  language?: string;
  crmOption?: string;
}

export interface CatalogSearchResult {
  _id: string; title: string; sku?: string; category?: string; subCategory?: string; shortDescription?: string;
  /** Key-normalized (e.g. "estimated_price_amount_inr"), whatever the
   * tenant's own source columns happened to be — search_products scans
   * this for anything price-shaped so a broad browse/price-range question
   * can be answered without a separate get_product_details call per item. */
  specifications?: Record<string, string>;
}
export interface CatalogItemDetail extends CatalogSearchResult {
  longDescription?: string; specifications?: Record<string, string>; attributes?: Record<string, string>;
  pdfs?: string[]; images?: string[]; videos?: string[]; tags?: string[];
}

export interface ConnectorSummary {
  type: string;
  name: string;
  isActive: boolean;
  lastSyncAt?: string;
  syncStatus?: string;
}

export interface CustomerSummary {
  name: string;
  email?: string;
  phone?: string;
  status: string;
  channel: string;
  recordType?: string;
  tags?: string[];
  intent?: string;
  lastContactedAt?: string;
  daysAgo?: number | null;
}

export interface CRMModuleEntry {
  module: string;
  count: number;
}

export interface TemplateSummary {
  name: string;
  type: string;
  category: string;
  subject?: string;
  body: string;
  variables: string[];
}

export interface QnAPairSummary {
  question: string;
  answer: string;
  category: string;
}

export interface InlineRecord {
  channel: string;
  module: string;
  displayName: string;
  data: Record<string, unknown>;
}

/** Built once per website crawl (ai/src/rag/website-profile-extractor.ts),
 * not per-question — see website-profile-fast-path.ts / base.agent.ts's
 * formatWebsiteProfileContext() for how this gets used in a live turn. */
export interface WebsiteProfileSummary {
  summary?: string;
  services?: string[];
  contact?: { phone?: string; email?: string; address?: string };
  hours?: string;
  staff?: Array<{ name: string; title?: string }>;
  faqs?: Array<{ question: string; answer: string }>;
}

export interface TenantContext {
  tenant: {
    id: string;
    name: string;
    slug: string;
    plan: string;
    settings: TenantSettings;
    branding: TenantBranding;
    aiConfig: TenantAIConfig;
  };
  /** Only the widget fields the continuous-voice worker needs — the
   * deterministic session.say() greeting text, the per-call duration cap,
   * and (now) the tenant's actual saved voice persona/language config,
   * previously saved but never consumed by worker.ts's own Deepgram/
   * Cartesia construction. Not the whole widget sub-document. */
  widget?: {
    greeting?: string;
    voice?: {
      maxSessionMinutes?: number;
      allowTextDuringVoice?: boolean;
      voiceName?: string;
      sttLanguage?: string;
      voicePreset?: {
        provider: 'cartesia';
        voiceId: string;
        displayName: string;
        gender: 'male' | 'female';
        language: string;
      };
    };
  };
  connectors: ConnectorSummary[];
  recentCustomers: CustomerSummary[];
  crmModules: Record<string, CRMModuleEntry[]>;
  inlineRecords: Record<string, InlineRecord[]>;
  templates: TemplateSummary[];
  qnaPairs: QnAPairSummary[];
  websiteProfile: WebsiteProfileSummary | null;
  hasWidgetDepartments: boolean;
  /** Left undefined when the tenant has never explicitly configured it —
   * see ResolvedTenantConfig's own comment for the `?? hasWidgetDepartments`
   * fallback this enables. */
  bookingRequireTeam?: boolean;
  bookingRequireService?: boolean;
  bookingRequireName: boolean;
  bookingContactRequirement: 'email_only' | 'phone_only' | 'email_or_phone' | 'email_and_phone';
  bookingStaffLabel: string;
  /** IANA timezone the tenant's booking hours are configured in — used to
   * ground the model's own "today"/"tomorrow" date math to the right
   * calendar day (see base.agent.ts's booking prompt block). */
  bookingTimezone: string;
}

export interface WidgetTeamSummary { teamId: string; name: string; }
export interface WidgetStaffSummary { staffId: string; name: string; }

export interface CRMRecord {
  externalId: string;
  displayName: string;
  data: Record<string, unknown>;
  syncedAt: string;
}

export interface CRMSearchResult {
  channel: string;
  module: string;
  displayName: string;
  data: Record<string, unknown>;
}

/* ── Client ── */
class BackendClient {
  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({
      baseURL: config.backend.url,
      timeout: 10000,
      headers: {
        'x-internal-key': config.backend.internalServiceKey,
        'Content-Type': 'application/json',
      },
    });
  }

  async getTenantContext(tenantId: string): Promise<TenantContext | null> {
    const cached = await getCachedTenantContext<TenantContext>(tenantId);
    if (cached) return cached;
    try {
      const res = await this.http.get<{ data: TenantContext }>(
        `/api/internal/tenant-context/${tenantId}`
      );
      const ctx = res.data.data;
      void setCachedTenantContext(tenantId, ctx);
      return ctx;
    } catch (err) {
      logger.warn('BackendClient: could not fetch tenant context', {
        tenantId,
        error: (err as Error).message,
      });
      return null;
    }
  }

  async getCRMRecords(
    tenantId: string,
    channel: string,
    module: string,
    options?: { limit?: number; search?: string }
  ): Promise<CRMRecord[]> {
    try {
      const res = await this.http.get<{ data: CRMRecord[] }>(
        `/api/internal/crm-records/${tenantId}/${channel}/${module}`,
        { params: options }
      );
      return res.data.data;
    } catch (err) {
      logger.warn('BackendClient: could not fetch CRM records', {
        tenantId, channel, module,
        error: (err as Error).message,
      });
      return [];
    }
  }

  /** Search CRM records by display name across all channels and modules.
   * `internalAuth` (opaque, forwarded from AgentInput) is the signed
   * identity assertion /api/internal/crm-search verifies to authorize the
   * native-CRM portion of this search — see that route's own comments. */
  async searchCRMRecords(tenantId: string, query: string, limit = 6, internalAuth?: string): Promise<CRMSearchResult[]> {
    try {
      const res = await this.http.get<{ data: CRMSearchResult[] }>(
        `/api/internal/crm-search/${tenantId}`,
        { params: { q: query, limit, internalAuth } }
      );
      return res.data.data || [];
    } catch {
      return [];
    }
  }

  /** Write an activity log entry to the backend DB. Fire-and-forget — never throws. */
  async writeLog(params: {
    tenantId: string;
    service?: 'ai' | 'backend';
    level?: 'info' | 'warn' | 'error' | 'debug';
    event: string;
    message: string;
    metadata?: Record<string, unknown>;
    sessionId?: string;
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/logs', { ...params, service: params.service ?? 'ai' });
    } catch { /* Logging must never crash the agent */ }
  }

  /** Log a security event to the backend SecurityEvent collection. Fire-and-forget. */
  async logSecurityEvent(params: {
    event: string; tenantId?: string; ip?: string; userAgent?: string;
    detail?: Record<string, unknown>;
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/security-event', params);
    } catch { /* Security logging must never crash the agent */ }
  }

  /** Persist a chat message to MongoDB. Fire-and-forget.
   * `channel` is only meaningful on a NEW session (backend's own
   * $setOnInsert) — an existing session's channel never flips mid-conversation. */
  async saveChatMessage(params: {
    tenantId: string; sessionId: string; role: 'user' | 'assistant';
    content: string; metadata?: Record<string, unknown>;
    visitorId?: string; visitorName?: string; visitorEmail?: string; visitorPhone?: string; escalated?: boolean;
    channel?: 'text' | 'push_to_talk' | 'continuous_voice';
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/chat-session', params);
    } catch { /* non-critical */ }
  }

  /** Log an AI action to the activity feed. Fire-and-forget — never throws. */
  async logAIAction(params: {
    tenantId: string;
    sessionId: string;
    actionType: string;
    summary: string;
    userMessage?: string;
    metadata?: Record<string, unknown>;
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/ai-action', params);
    } catch (err) {
      logger.warn('BackendClient: logAIAction failed', {
        error: (err as Error).message,
        tenantId: params.tenantId,
        actionType: params.actionType,
      });
    }
  }

  /** Records one turn's worth of LLM token usage/cost against the tenant's
   * daily-bucketed counter — fire-and-forget, mirrors writeLog/logAIAction's
   * own never-throws convention (usage tracking must never break a real
   * chat response). */
  async trackAiTokenUsage(params: {
    tenantId: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    estimatedCostUsd: number;
    usedModerationFallback?: boolean;
    sttSeconds?: number;
    ttsCharacters?: number;
    voiceCostUsd?: number;
    isVoiceRequest?: boolean;
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/ai-token-usage', params);
    } catch (err) {
      logger.warn('BackendClient: trackAiTokenUsage failed', { error: (err as Error).message, tenantId: params.tenantId });
    }
  }

  /** Month-to-date total tokens used by a tenant — the source of truth
   * checkTenantTokenQuota() (rate-limiter.ts) briefly caches in Redis rather
   * than calling this on every message. */
  async getTenantTokenUsageThisMonth(tenantId: string): Promise<number> {
    try {
      const res = await this.http.get<{ data: { totalTokens: number } }>(`/api/internal/ai-token-usage/${tenantId}`);
      return res.data.data.totalTokens;
    } catch (err) {
      logger.warn('BackendClient: getTenantTokenUsageThisMonth failed', { error: (err as Error).message, tenantId });
      return 0;
    }
  }

  /** Reports one COMPLETED continuous-voice session's real usage — called
   * once by the voice-agent worker at session close (a separate, long-running
   * process, not ai/'s Express app), same never-throws convention as
   * trackAiTokenUsage above. */
  async reportContinuousVoiceUsage(params: {
    tenantId: string; minutes: number; deepgramSttSeconds: number;
    cartesiaTtsCharacters: number; estimatedCostUsd: number;
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/continuous-voice-usage', params);
    } catch (err) {
      logger.warn('BackendClient: reportContinuousVoiceUsage failed', { error: (err as Error).message, tenantId: params.tenantId });
    }
  }

  /** Month-to-date continuous-voice minutes used by a tenant — the source of
   * truth checkTenantVoiceMinutesQuota() (rate-limiter.ts) briefly caches in
   * Redis, same pattern as getTenantTokenUsageThisMonth() above. */
  async getTenantVoiceMinutesUsageThisMonth(tenantId: string): Promise<number> {
    try {
      const res = await this.http.get<{ data: { minutes: number } }>(`/api/internal/continuous-voice-usage/${tenantId}`);
      return res.data.data.minutes;
    } catch (err) {
      logger.warn('BackendClient: getTenantVoiceMinutesUsageThisMonth failed', { error: (err as Error).message, tenantId });
      return 0;
    }
  }

  /* ── Product Catalog ──────────────────────────────────────────────── */

  /** Product search tool's backing call — a real Mongo text search, not RAG.
   * Short-TTL cached (same 10s convention as getTenantContext) since a
   * booking-then-catalog-question conversation can otherwise repeat the
   * identical search within seconds. */
  async searchCatalogItems(tenantId: string, opts: { query?: string; category?: string }): Promise<CatalogSearchResult[]> {
    const cached = await getCachedCatalogSearch<CatalogSearchResult[]>(tenantId, opts.query, opts.category);
    if (cached) return cached;
    try {
      const res = await this.http.post<{ data: { items: CatalogSearchResult[] } }>('/api/internal/catalog/search', { tenantId, ...opts });
      const items = res.data.data.items;
      void setCachedCatalogSearch(tenantId, opts.query, opts.category, items);
      return items;
    } catch (err) {
      logger.warn('BackendClient: searchCatalogItems failed', { error: (err as Error).message, tenantId });
      return [];
    }
  }

  /** get_product_details tool's backing call. Short-TTL cached, same convention. */
  async getCatalogItemBySku(tenantId: string, sku: string): Promise<CatalogItemDetail | null> {
    const cached = await getCachedCatalogItem<CatalogItemDetail>(tenantId, sku);
    if (cached) return cached;
    try {
      const res = await this.http.get<{ data: { item: CatalogItemDetail | null } }>(`/api/internal/catalog/${tenantId}/sku/${encodeURIComponent(sku)}`);
      const item = res.data.data.item;
      if (item) void setCachedCatalogItem(tenantId, sku, item);
      return item;
    } catch (err) {
      logger.warn('BackendClient: getCatalogItemBySku failed', { error: (err as Error).message, tenantId });
      return null;
    }
  }

  /** Finds-or-creates the persistent KnowledgeSource identity for a website
   * crawl, marking it 'running' — called once per ingestWebsite() run. */
  async startCatalogKnowledgeSource(tenantId: string, type: 'website' | 'excel' | 'csv' | 'json', label: string): Promise<string | null> {
    try {
      const res = await this.http.post<{ data: { knowledgeSourceId: string } }>('/api/internal/catalog/knowledge-source/start', { tenantId, type, label });
      return res.data.data.knowledgeSourceId;
    } catch (err) {
      logger.warn('BackendClient: startCatalogKnowledgeSource failed', { error: (err as Error).message, tenantId });
      return null;
    }
  }

  async finishCatalogKnowledgeSource(knowledgeSourceId: string, outcome: 'completed' | 'failed', durationMs: number, error?: string): Promise<void> {
    try {
      await this.http.post('/api/internal/catalog/knowledge-source/finish', { knowledgeSourceId, outcome, durationMs, error });
    } catch (err) {
      logger.warn('BackendClient: finishCatalogKnowledgeSource failed', { error: (err as Error).message, knowledgeSourceId });
    }
  }

  /** Upserts one catalog item extracted from a crawled page's Product
   * JSON-LD — content-hash-deduped and KnowledgeSource-counted on the
   * backend side (see upsertCatalogItemFromSource). */
  async upsertCatalogItemFromCrawl(
    tenantId: string, knowledgeSourceId: string, sourceUrl: string, fields: Record<string, unknown>
  ): Promise<void> {
    try {
      await this.http.post('/api/internal/catalog/upsert-from-crawl', { tenantId, knowledgeSourceId, sourceUrl, fields });
    } catch (err) {
      logger.warn('BackendClient: upsertCatalogItemFromCrawl failed', { error: (err as Error).message, tenantId, sourceUrl });
    }
  }

  async upsertWebsiteProfileFromCrawl(
    tenantId: string, knowledgeSourceId: string, fields: Record<string, unknown>
  ): Promise<void> {
    try {
      await this.http.post('/api/internal/website-profile/upsert-from-crawl', { tenantId, knowledgeSourceId, fields });
    } catch (err) {
      logger.warn('BackendClient: upsertWebsiteProfileFromCrawl failed', { error: (err as Error).message, tenantId });
    }
  }

  /** Send an email via the backend message service. */
  async sendEmail(params: {
    tenantId: string; toEmail: string; toName?: string;
    subject: string; body: string; templateName?: string;
  }): Promise<{ success: boolean; messageId?: string }> {
    try {
      const res = await this.http.post<{ data: { messageId?: string } }>(
        '/api/internal/send-email', params
      );
      return { success: true, messageId: res.data.data?.messageId };
    } catch (err) {
      logger.warn('BackendClient: send-email failed', { error: (err as Error).message });
      return { success: false };
    }
  }

  /** Create a calendar activity (appointment/meeting) in the backend. */
  async createActivity(params: {
    tenantId: string;
    type: string;
    title: string;
    startDate: string;
    endDate: string;
    notes?: string;
    linkedPerson?: {
      displayName: string;
      email?: string;
      phone?: string;
      module: string;
      channel: string;
    };
  }): Promise<{ success: boolean; activityId?: string }> {
    try {
      const res = await this.http.post<{ data: { activityId: string } }>(
        '/api/internal/create-activity', params
      );
      return { success: true, activityId: res.data.data?.activityId };
    } catch (err) {
      logger.warn('BackendClient: create-activity failed', { error: (err as Error).message });
      return { success: false };
    }
  }

  /** Send an SMS via Twilio. */
  async sendSms(params: { to: string; message: string }): Promise<{ success: boolean }> {
    try {
      await this.http.post('/api/internal/send-sms', params);
      return { success: true };
    } catch (err) {
      logger.warn('BackendClient: send-sms failed', { error: (err as Error).message });
      return { success: false };
    }
  }

  /** Create a new AutomationRun tracking record. */
  async createAutomationRun(params: {
    tenantId: string;
    sessionId: string;
    trigger: string;
    customerName: string;
    steps: Array<{ name: string; status: string }>;
  }): Promise<{ success: boolean; runId?: string }> {
    try {
      const res = await this.http.post<{ data: { runId: string } }>(
        '/api/internal/create-automation-run', params
      );
      return { success: true, runId: res.data.data?.runId };
    } catch (err) {
      logger.warn('BackendClient: create-automation-run failed', { error: (err as Error).message });
      return { success: false };
    }
  }

  /** Update a single step within an AutomationRun. Fire-and-forget — never throws. */
  async updateAutomationStep(
    runId: string,
    stepIndex: number,
    update: {
      status: string;
      result?: string;
      error?: string;
      runStatus?: string;
      customerEmail?: string;
      customerPhone?: string;
      activityId?: string;
      messageContent?: { subject?: string; body?: string; text?: string; to?: string };
    }
  ): Promise<void> {
    try {
      await this.http.put(`/api/internal/automation-run/${runId}/step`, { stepIndex, ...update });
    } catch (err) {
      logger.warn('BackendClient: updateAutomationStep failed', { error: (err as Error).message });
    }
  }

  /** List activities for a tenant (used by AI to find existing meetings to reschedule). */
  async listActivities(
    tenantId: string,
    params: { search?: string; type?: string; limit?: number }
  ): Promise<Array<{ _id: string; type: string; title: string; startDate?: string; endDate?: string; status: string }>> {
    try {
      const qs = new URLSearchParams({ tenantId, limit: String(params.limit ?? 5) });
      if (params.type) qs.set('type', params.type);
      const res = await this.http.get<{ data: Array<{ _id: string; type: string; title: string; startDate?: string; endDate?: string; status: string }> }>(
        `/api/internal/activities?${qs.toString()}`
      );
      const items = res.data.data ?? [];
      if (params.search) {
        const q = params.search.toLowerCase();
        return items.filter(a => a.title.toLowerCase().includes(q));
      }
      return items;
    } catch {
      return [];
    }
  }

  /** Creates a real CRM Lead from the website widget's own conversation —
   * calls the backend's new internal endpoint, which reuses the
   * ALREADY-BUILT captureLeadFromExternalSource() (the same function the
   * browser extension's own lead-capture flow calls), tagged platform:
   * 'chatbot', with a round-robin-assigned staff owner. The AI service
   * never touches MongoDB directly — this is the one and only path a
   * widget conversation turns into a real CRM record. */
  async createLeadFromWidget(params: {
    tenantId: string; sessionId: string; visitorId?: string; sourceUrl?: string;
    firstName: string; lastName?: string; email?: string; phone?: string; company?: string; service?: string;
    leadScore?: number; buyingIntent?: 'low' | 'medium' | 'high';
    interestedItems?: Array<{ datasetId: string; recordId: string; title: string; datasetVersion: number }>;
    requirement?: string; conversationSummary?: string;
  }): Promise<{ success: boolean; leadId?: string; leadDisplayId?: string; alreadyCreated?: boolean }> {
    try {
      const res = await this.http.post<{ data: { leadId: string; leadDisplayId?: string; alreadyCreated?: boolean } }>(
        '/api/internal/widget-lead-capture', params,
      );
      return { success: true, leadId: res.data.data?.leadId, leadDisplayId: res.data.data?.leadDisplayId, alreadyCreated: res.data.data?.alreadyCreated };
    } catch (err) {
      logger.warn('BackendClient: createLeadFromWidget failed', { error: (err as Error).message, tenantId: params.tenantId });
      return { success: false };
    }
  }

  /** Enriches an ALREADY-CREATED Lead as buying intent increases later in
   * the same session — never creates a second Lead, only updates the one
   * `leadId` already anchors (see base.agent.ts's continuous-enrichment
   * comment on why score/rating aren't a frozen creation-time snapshot). */
  async updateLeadFromWidget(tenantId: string, leadId: string, params: {
    leadScore?: number; buyingIntent?: 'low' | 'medium' | 'high';
    interestedItems?: Array<{ datasetId: string; recordId: string; title: string; datasetVersion: number }>;
    requirement?: string; conversationSummary?: string;
  }): Promise<{ success: boolean }> {
    try {
      await this.http.patch(`/api/internal/widget-lead-update/${leadId}`, { tenantId, ...params });
      return { success: true };
    } catch (err) {
      logger.warn('BackendClient: updateLeadFromWidget failed', { error: (err as Error).message, tenantId, leadId });
      return { success: false };
    }
  }

  /** These two calls record a website crawl's lifecycle onto the tenant's
   * own widget config — deliberately NOT in updateTenant()'s client-editable
   * field allow-list (a caller shouldn't be able to just claim "I crawled N
   * pages" via the generic tenant-update endpoint); these internal,
   * service-key-gated routes are the only path that can set them, matching
   * the same "AI service is the source of truth for what it actually did"
   * pattern createLeadFromWidget() above already uses. */
  /** Called the instant a crawl begins — sets crawlStatus:'crawling' and
   * resets the current-run counters immediately, so a page reload mid-crawl
   * never shows a previous run's numbers as though they're the current
   * one. See internal.routes.ts's widget-crawl-start for the full doc. */
  async startWebsiteCrawl(tenantId: string): Promise<void> {
    try {
      await this.http.post('/api/internal/widget-crawl-start', { tenantId });
    } catch (err) {
      logger.warn('BackendClient: startWebsiteCrawl failed', { error: (err as Error).message, tenantId });
    }
  }

  async recordWebsiteCrawlResult(tenantId: string, result: {
    pagesCrawled: number; pagesIndexed: number; pagesFailed: number;
    chunksIndexed: number; status: 'ready' | 'ready_with_warnings' | 'failed';
  }): Promise<void> {
    try {
      await this.http.post('/api/internal/widget-crawl-complete', { tenantId, ...result });
    } catch (err) {
      logger.warn('BackendClient: recordWebsiteCrawlResult failed', { error: (err as Error).message, tenantId });
    }
  }

  /** Real, tenant-wide business-hours availability for the widget's booking
   * tool — see the backend's availability.service.ts for the single-capacity
   * model (no per-staff calendars exist anywhere in this codebase). An
   * optional staffId narrows this to one chosen doctor's own availability,
   * for tenants using the department/doctor booking wizard. */
  async getWidgetAvailability(
    tenantId: string,
    opts: {
      days?: number; timeOfDay?: 'morning' | 'afternoon' | 'any'; staffId?: string; teamId?: string;
      /** A visitor-requested calendar date, YYYY-MM-DD, resolved against the
       * tenant's own timezone server-side — see internal.routes.ts's
       * `/widget-availability` and availability.service.ts's own comments. */
      date?: string;
    } = {}
  ): Promise<Array<{ startIso: string; endIso: string; label: string }>> {
    try {
      const res = await this.http.get<{ data: { slots: Array<{ startIso: string; endIso: string; label: string }> } }>(
        '/api/internal/widget-availability', { params: { tenantId, ...opts } },
      );
      return res.data.data?.slots ?? [];
    } catch (err) {
      logger.warn('BackendClient: getWidgetAvailability failed', { error: (err as Error).message, tenantId });
      return [];
    }
  }

  /** Departments (Teams marked showInWidget:true) a visitor may choose
   * between when booking. Empty result reads as "no departments configured"
   * — proceed straight to availability, not an error. */
  async getWidgetTeams(tenantId: string, serviceHint?: string): Promise<Array<{ teamId: string; name: string }>> {
    // serviceHint-filtered results are cached separately from the plain
    // (unfiltered) list — different cache keys, no cross-contamination.
    const cacheKey = serviceHint ? `${tenantId}:${serviceHint.toLowerCase()}` : tenantId;
    const cached = await getCachedWidgetTeams<Array<{ teamId: string; name: string }>>(cacheKey);
    if (cached) return cached;
    try {
      const res = await this.http.get<{ data: { teams: Array<{ teamId: string; name: string }> } }>(
        '/api/internal/widget-teams', { params: { tenantId, serviceHint } },
      );
      const teams = res.data.data?.teams ?? [];
      void setCachedWidgetTeams(cacheKey, teams);
      return teams;
    } catch (err) {
      logger.warn('BackendClient: getWidgetTeams failed', { error: (err as Error).message, tenantId });
      return [];
    }
  }

  /** Active staff (doctors) within one chosen department. */
  async getWidgetStaff(tenantId: string, teamId: string): Promise<Array<{ staffId: string; name: string }>> {
    const cached = await getCachedWidgetStaff<Array<{ staffId: string; name: string }>>(tenantId, teamId);
    if (cached) return cached;
    try {
      const res = await this.http.get<{ data: { staff: Array<{ staffId: string; name: string }> } }>(
        '/api/internal/widget-staff', { params: { tenantId, teamId } },
      );
      const staff = res.data.data?.staff ?? [];
      void setCachedWidgetStaff(tenantId, teamId, staff);
      return staff;
    } catch (err) {
      logger.warn('BackendClient: getWidgetStaff failed', { error: (err as Error).message, tenantId });
      return [];
    }
  }

  /** Converts an offered slot into a real Lead + Meeting — see the backend's
   * own /api/internal/widget-book-meeting for the full orchestration
   * (re-check availability, round-robin or a chosen doctor,
   * captureLeadFromExternalSource, createMeeting). Never throws — a failure
   * just means the visitor gets told the booking couldn't be completed, same
   * posture as createLeadFromWidget() above. */
  async bookWidgetMeeting(params: {
    tenantId: string; sessionId: string; visitorId?: string; sourceUrl?: string;
    startIso: string; endIso: string; firstName: string; lastName?: string; email?: string; phone?: string; topic?: string;
    staffId?: string;
    // Booking-parity fields (Product Cards / lead-gen phase) — a visitor who
    // reaches high buying intent and books directly (never going through a
    // separate "Request Quote" turn) must get the same rich Lead as the
    // plain capture path, not an impoverished booking-only record.
    leadScore?: number; buyingIntent?: 'low' | 'medium' | 'high';
    interestedItems?: Array<{ datasetId: string; recordId: string; title: string; datasetVersion: number }>;
    requirement?: string; conversationSummary?: string;
  }): Promise<{ success: boolean; meetingId?: string; staffName?: string; leadId?: string; alreadyCreated?: boolean; error?: string; reason?: 'slot_taken' }> {
    try {
      const res = await this.http.post<{ data: { meetingId: string; staffName?: string; leadId: string; alreadyCreated?: boolean } }>(
        '/api/internal/widget-book-meeting', params,
      );
      return { success: true, ...res.data.data };
    } catch (err: any) {
      const message = err?.response?.data?.message || (err as Error).message;
      logger.warn('BackendClient: bookWidgetMeeting failed', { error: message, tenantId: params.tenantId });
      // A 409 always means the slot itself is unavailable (pre-check or the
      // race-guard unique index) — checked via status code, not the message
      // string, so a later wording change to either backend message can't
      // silently break this.
      const reason = err?.response?.status === 409 ? ('slot_taken' as const) : undefined;
      return { success: false, error: message, reason };
    }
  }

  /* ── Generic Dataset system (search_dataset / get_dataset_record) ──── */

  /** Every dataset this tenant has enabled for the widget — search_dataset
   * uses this to resolve a datasetId (or offer a choice) since the model is
   * never told one in advance. */
  async listDatasetsForChatbot(tenantId: string): Promise<Array<{ datasetId: string; name: string }>> {
    try {
      const res = await this.http.get<{ data: { datasets: Array<{ datasetId: string; name: string }> } }>(
        '/api/internal/datasets', { params: { tenantId } },
      );
      return res.data.data?.datasets ?? [];
    } catch (err) {
      logger.warn('BackendClient: listDatasetsForChatbot failed', { error: (err as Error).message, tenantId });
      return [];
    }
  }

  /** The column/role schema for one dataset — feeds the query router's
   * fast-path/classifier so a plan only ever references real fields. */
  async getDatasetSchema(tenantId: string, datasetId: string): Promise<DatasetSchemaContext['columns']> {
    try {
      const res = await this.http.get<{ data: { columns: DatasetSchemaContext['columns'] } }>(
        `/api/internal/datasets/${datasetId}/schema`, { params: { tenantId } },
      );
      return res.data.data?.columns ?? [];
    } catch (err) {
      logger.warn('BackendClient: getDatasetSchema failed', { error: (err as Error).message, tenantId, datasetId });
      return [];
    }
  }

  /** Same call target as getDatasetSchema — helper name kept for tool
   * readability where the caller specifically needs the normalizedName ->
   * originalName relabel map (hardening Gap 5), not the query-router shape. */
  buildDatasetLabelMap(columns: DatasetSchemaContext['columns']): Map<string, string> {
    return new Map(columns.map((c) => [c.normalizedName, c.originalName]));
  }

  /** Executes a QueryPlan the query router built — tenantId/datasetId are
   * passed as explicit top-level params here (never read out of `plan`
   * itself, which structurally has no such fields — see query-router/types.ts),
   * mirroring executeDatasetQuery()'s own signature on the backend side. */
  async executeDatasetQuery(
    tenantId: string, datasetId: string, plan: QueryPlan,
  ): Promise<{ results: Array<{ recordId: string; data: Record<string, unknown>; datasetId: string; datasetName: string; datasetVersion: number; sourceLabel: string; rowNumber: number }>; count?: number; datasetName: string | null; degraded?: boolean }> {
    try {
      const res = await this.http.post<{ data: { results: Array<{ recordId: string; data: Record<string, unknown>; datasetId: string; datasetName: string; datasetVersion: number; sourceLabel: string; rowNumber: number }>; count?: number; datasetName: string | null; degraded?: boolean } }>(
        '/api/internal/datasets/query', { tenantId, datasetId, plan },
      );
      return res.data.data ?? { results: [], datasetName: null };
    } catch (err) {
      logger.warn('BackendClient: executeDatasetQuery failed', { error: (err as Error).message, tenantId, datasetId });
      return { results: [], datasetName: null };
    }
  }

  /** get_dataset_record tool's backing call — exact-record lookup by id. */
  async getDatasetRecord(
    tenantId: string, datasetId: string, recordId: string,
  ): Promise<{ recordId: string; data: Record<string, unknown>; datasetId: string; datasetName: string; datasetVersion: number; sourceLabel: string; rowNumber: number } | null> {
    try {
      const res = await this.http.get<{ data: { record: { recordId: string; data: Record<string, unknown>; datasetId: string; datasetName: string; datasetVersion: number; sourceLabel: string; rowNumber: number } | null } }>(
        `/api/internal/datasets/${datasetId}/record/${encodeURIComponent(recordId)}`, { params: { tenantId } },
      );
      return res.data.data?.record ?? null;
    } catch (err) {
      logger.warn('BackendClient: getDatasetRecord failed', { error: (err as Error).message, tenantId, datasetId, recordId });
      return null;
    }
  }

  /** Update an existing activity (for reschedule, status change, etc.). */
  async updateActivity(
    tenantId: string,
    activityId: string,
    update: {
      startDate?: string; endDate?: string; title?: string; status?: string;
      linkedPerson?: { displayName: string; email?: string; phone?: string; module?: string; channel?: string };
    }
  ): Promise<{ success: boolean }> {
    try {
      await this.http.put(`/api/internal/activity/${activityId}`, { tenantId, ...update });
      return { success: true };
    } catch (err) {
      logger.warn('BackendClient: updateActivity failed', { error: (err as Error).message });
      return { success: false };
    }
  }
}

export const backendClient = new BackendClient();
