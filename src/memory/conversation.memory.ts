import Redis from 'ioredis';
import { config } from '../config';
import { logger } from '../utils/logger';

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

let redis: Redis | null = null;
// Set to true when a fatal, non-retryable error occurs (e.g. wrong password).
// Prevents an infinite reconnect loop that spams logs and wastes CPU.
let redisDead = false;

function getRedis(): Redis | null {
  if (redisDead) return null;
  if (!redis) {
    try {
      const opts = {
        enableOfflineQueue: true,   // queue commands during reconnect instead of throwing
        maxRetriesPerRequest: 3,
        connectTimeout: 10000,
        commandTimeout: 5000,
        retryStrategy: (times: number) => (times > 5 ? null : Math.min(times * 300, 3000)),
      };
      redis = config.redis.url
        ? new Redis(config.redis.url, opts)
        : new Redis({ host: config.redis.host, port: config.redis.port, password: config.redis.password || undefined, tls: config.redis.tls ? {} : undefined, ...opts });

      redis.on('error', (err) => {
        const msg = (err as Error).message || '';
        // WRONGPASS / ERR AUTH — wrong credentials, retrying is pointless
        if (msg.includes('WRONGPASS') || msg.includes('ERR AUTH') || msg.includes('invalid username-password')) {
          logger.error('Redis auth failed — disabling Redis for this session. Fix REDIS_PASSWORD in ai/.env.', { error: msg });
          redisDead = true;
          try { redis?.disconnect(); } catch { /* ignore */ }
          redis = null;
          return;
        }
        logger.warn('Redis connection error — resetting client', { error: msg });
        try { redis?.disconnect(); } catch { /* ignore */ }
        redis = null;
      });
      redis.on('close', () => { if (!redisDead) redis = null; });
    } catch {
      return null;
    }
  }
  return redis;
}

const TTL = 86400;
const MAX_MESSAGES = 20;

/** Every conversation-state key in this file is namespaced by BOTH tenantId
 * and sessionId — sessionId alone used to be the sole key component, a real
 * cross-tenant data-isolation gap (confirmed live: two different tenants'
 * conversations sharing one sessionId shared the same Redis-backed state —
 * history, CRM mode, lead-capture progress, booking selections — since
 * nothing server-side enforces sessionId uniqueness/format beyond "non-empty
 * string"; the widget's own crypto.randomUUID() generation is a client-side
 * convention, not a server-enforced guarantee). Old, unscoped keys simply
 * age out via their existing TTLs — no migration needed. */
function scopedKey(prefix: string, tenantId: string, sessionId: string): string {
  return `${prefix}:${tenantId}:${sessionId}`;
}

export async function getHistory(tenantId: string, sessionId: string): Promise<ConversationMessage[]> {
  const client = getRedis();
  if (!client) return [];
  try {
    const raw = await client.get(scopedKey('chat:hist', tenantId, sessionId));
    return raw ? (JSON.parse(raw) as ConversationMessage[]) : [];
  } catch (err) {
    logger.warn('Could not fetch conversation history', { tenantId, sessionId, error: (err as Error).message });
    return [];
  }
}

export async function appendMessage(
  tenantId: string,
  sessionId: string,
  message: ConversationMessage
): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    const history = await getHistory(tenantId, sessionId);
    history.push(message);
    const trimmed = history.slice(-MAX_MESSAGES);
    await client.setex(scopedKey('chat:hist', tenantId, sessionId), TTL, JSON.stringify(trimmed));
  } catch (err) {
    logger.warn('Could not save conversation message', { tenantId, sessionId, error: (err as Error).message });
  }
}

export async function clearHistory(tenantId: string, sessionId: string): Promise<void> {
  const client = getRedis();
  if (!client) return;
  await client.del(scopedKey('chat:hist', tenantId, sessionId));
}

/* ── CRM session state — remembers which module was last discussed ── */
export interface CRMSessionState {
  mode: 'crm' | 'lead';
  channel?: string;
  module?: string;
  recordsBlock?: string; // Formatted records text injected into last response
}

export async function getCRMState(tenantId: string, sessionId: string): Promise<CRMSessionState | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(scopedKey('chat:crm', tenantId, sessionId));
    return raw ? (JSON.parse(raw) as CRMSessionState) : null;
  } catch { return null; }
}

export async function setCRMState(tenantId: string, sessionId: string, state: CRMSessionState): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(scopedKey('chat:crm', tenantId, sessionId), TTL, JSON.stringify(state));
  } catch { /* non-critical */ }
}

/* ── Pending intent state — stores partial intent waiting for user clarification ── */
export interface PendingIntent {
  intent: string;
  entities: {
    person: string | null;
    rawDateTime: string | null;
    meetingType: string | null;
    notes: string | null;
  };
  missingRequired: string[];
  clarificationQuestion: string | null;
  // Extended: discriminates what we're waiting for next
  pendingAction?: 'await_customer' | 'await_reschedule_confirm';
  // await_customer: meeting created, waiting for user to name the customer
  activityId?: string;
  dateRangeStr?: string;
  runId?: string;
  // await_reschedule_confirm: found existing meeting, waiting for yes/no
  existingActivityId?: string;
  existingTitle?: string;
  existingDateStr?: string;
  newStartIso?: string;
  newEndIso?: string;
  custName?: string;
}

const PENDING_TTL = 300; // 5 minutes — discard if user doesn't reply

export async function getPendingIntent(tenantId: string, sessionId: string): Promise<PendingIntent | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(scopedKey('chat:pending', tenantId, sessionId));
    return raw ? (JSON.parse(raw) as PendingIntent) : null;
  } catch { return null; }
}

export async function setPendingIntent(tenantId: string, sessionId: string, intent: PendingIntent): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(scopedKey('chat:pending', tenantId, sessionId), PENDING_TTL, JSON.stringify(intent));
  } catch { /* non-critical */ }
}

export async function clearPendingIntent(tenantId: string, sessionId: string): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try { await client.del(scopedKey('chat:pending', tenantId, sessionId)); } catch { /* non-critical */ }
}

/* ── Website widget lead-capture state — fields collected so far, across
   turns, for the widget's own lead-capture flow. Same TTL/keying convention
   as everything else in this file. Fails open (returns null / no-ops) on a
   Redis outage, same as CRMSessionState/PendingIntent above — the worst
   case is a degraded conversation (re-asking for info), never a phantom
   Lead, since nothing is durable until captureLeadFromWidget() actually
   succeeds against the backend. ── */
export interface LeadCaptureState {
  firstName?: string;
  lastName?:  string;
  email?:     string;
  phone?:     string;
  company?:   string;
  /** The visitor's stated reason for contacting/what they're interested in
   * — maps onto Lead.interestedServices on the backend. Merged additively,
   * same non-destructive rule as every other field here. */
  service?:   string;
  leadCreated?: boolean;
  leadId?:      string;
  /** Has the "want to book a time, or anything else?" trailing nudge
   * already been shown this session — shown once, not appended to every
   * informational answer, to avoid feeling repetitive. */
  nudgeShown?: boolean;
  /** Buying-intent / lead-score continuous-enrichment fields — see
   * ai/src/agents/buying-intent.ts. `buyingIntent`/`leadScore` are merged
   * "keep the highest-seen" across turns, same non-downgrade rule as
   * email/phone above. `lastSent*` records what was last actually written
   * to the backend Lead (at creation or a later enrichment call), so a
   * later turn can detect a genuinely NEW upgrade worth another network
   * call rather than re-sending on every turn. */
  buyingIntent?: 'low' | 'medium' | 'high';
  leadScore?:    number;
  lastSentBuyingIntent?: 'low' | 'medium' | 'high';
  /** What was last actually sent to the backend for requirement/interested-
   * items, so the enrichment path only calls the backend again on a
   * genuinely NEW value/item, not on every turn. */
  lastSentRequirement?: string;
  lastSentItemCount?: number;
  /** Deterministic, verbatim capture of the visitor's own message — set
   * whenever a requirement-shaped signal (quantity/application/need
   * language) is detected, continuing to update AFTER the Lead has already
   * been created (not gated on contact info still being missing). */
  requirement?: string;
  /** Deduplicated by datasetId+recordId, capped at 10 — every distinct item
   * this session's search_dataset calls have surfaced, not just the last one.
   * datasetVersion records WHICH version the visitor actually saw, since a
   * re-upload can change price/specs under the same datasetId+recordId. */
  interestedItems?: Array<{ datasetId: string; recordId: string; title: string; datasetVersion: number }>;
  /** The single visitor message that triggered the session's highest
   * buying-intent classification so far — the deterministic, cheap
   * conversationSummary source (no LLM summarization call). */
  conversationSummary?: string;
  /** Set by request-quote-shortcut.ts the moment it asks "could I get your
   * name and email?" after a Request Quote/Demo click — the item/topic name
   * being quoted, so a FOLLOW-UP reply (e.g. a bare "John, john@x.com" with
   * no "quote" wording at all) is still recognised as continuing this exact
   * deterministic flow rather than falling through to the general LLM path.
   * Cleared once contact info is captured (mirrors BookingState.
   * confirmingSlot's own lifecycle). */
  awaitingQuoteContactFor?: string;
}

export async function getLeadCaptureState(tenantId: string, sessionId: string): Promise<LeadCaptureState | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(scopedKey('chat:lead', tenantId, sessionId));
    return raw ? (JSON.parse(raw) as LeadCaptureState) : null;
  } catch { return null; }
}

export async function setLeadCaptureState(tenantId: string, sessionId: string, state: LeadCaptureState): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(scopedKey('chat:lead', tenantId, sessionId), TTL, JSON.stringify(state));
  } catch { /* non-critical */ }
}

/** Per-session, per-dataset record of which recordIds have already been
 * surfaced as item cards via search_dataset — fixes a real gap where "Show
 * more" had no way to avoid re-showing the same top-N results. Flat/global
 * per dataset rather than keyed by exact query text, since a visitor who's
 * already seen a record doesn't need to see it again even from a differently
 * worded later question. Capped per-dataset so a very long browsing session
 * doesn't grow this unboundedly. */
export interface DatasetBrowseState {
  [datasetId: string]: string[];
}

const BROWSE_STATE_CAP_PER_DATASET = 50;

export async function getDatasetBrowseState(tenantId: string, sessionId: string): Promise<DatasetBrowseState> {
  const client = getRedis();
  if (!client) return {};
  try {
    const raw = await client.get(scopedKey('chat:browse', tenantId, sessionId));
    return raw ? (JSON.parse(raw) as DatasetBrowseState) : {};
  } catch { return {}; }
}

/** Appends newly-shown recordIds for one dataset, deduplicated and capped —
 * caller passes only the NEW ids just surfaced this turn, not the full list. */
export async function appendDatasetBrowseState(
  tenantId: string, sessionId: string, datasetId: string, newRecordIds: string[],
): Promise<void> {
  if (!newRecordIds.length) return;
  const client = getRedis();
  if (!client) return;
  try {
    const state = await getDatasetBrowseState(tenantId, sessionId);
    const existing = state[datasetId] ?? [];
    const merged = Array.from(new Set([...existing, ...newRecordIds])).slice(-BROWSE_STATE_CAP_PER_DATASET);
    state[datasetId] = merged;
    await client.setex(scopedKey('chat:browse', tenantId, sessionId), TTL, JSON.stringify(state));
  } catch { /* non-critical */ }
}

/** Mirrors LeadCaptureState's shape/merge convention. `offeredSlots` is what
 * check_meeting_availability last showed the visitor — book_meeting refuses
 * any startIso not present here (anti-hallucination guard, since the model
 * could otherwise invent a plausible-looking time that was never actually
 * free). `meetingCreated` mirrors leadCreated's idempotency role. */
export interface BookingState {
  offeredSlots?: Array<{ startIso: string; endIso: string; label?: string }>;
  offeredAt?: number;
  meetingCreated?: boolean;
  meetingId?: string;
  /** Department/doctor wizard selections, carried across turns the same way
   * every other BookingState field already is — set once list_departments/
   * list_doctors are called, then merged into check_meeting_availability/
   * book_meeting's own staffId args the same cleanArg(args.X) || state.X
   * pattern already used for name/email/phone. */
  selectedTeamId?: string;
  selectedTeamName?: string;
  selectedStaffId?: string;
  selectedStaffName?: string;
  /** Set by list_doctors.tool.ts when a department has exactly ONE active
   * doctor — the same "single-item-so-treat-as-implicitly-offered" pattern
   * confirmingSlot/resolveSlot() already use for a single offered slot. A
   * real, confirmed gap this fixes: the AI would ask "would you like to see
   * Dr. X?", but nothing durable recorded WHICH doctor was just suggested —
   * only the team was persisted — so a bare "yes" reply had no state to
   * resolve against and re-triggered list_departments/list_doctors from
   * scratch. Cleared once selectedStaffId is actually set. */
  offeredStaffId?: string;
  offeredStaffName?: string;
  /** Set by list_departments.tool.ts the moment it returns a real
   * department list — closes a gap the offeredStaffId fix above didn't
   * cover: the visitor's reply picking a department (e.g. "ss department")
   * comes BEFORE any team/staff field is set, so without this flag the
   * booking-flow-continuation check in base.agent.ts had nothing to key
   * off yet for that one turn, and the full 7-tool set stayed bound. */
  departmentsOffered?: boolean;
  /** Set by the deterministic booking-confirmation shortcut the moment it
   * resolves a specific slot (by time, ordinal, or a bare affirmative with
   * only one slot offered) — so a LATER turn that only supplies contact
   * info (without restating the time) still knows which slot was already
   * confirmed. Cleared once meetingCreated is set. */
  confirmingSlot?: { startIso: string; endIso: string; label?: string };
  /** Set by the deterministic shortcut when 2+ slots were offered and the
   * visitor gave a bare affirmative ("sounds good") that's genuinely
   * ambiguous among them — the shortcut asked "which time?" but (unlike the
   * confirmingSlot branch above) previously never persisted this back to
   * state at all, a real asymmetry: a follow-up turn had no record that a
   * time-clarification question was still pending. Cleared once a specific
   * slot is actually resolved. */
  disambiguating?: boolean;
}

export async function getBookingState(tenantId: string, sessionId: string): Promise<BookingState | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(scopedKey('chat:booking', tenantId, sessionId));
    return raw ? (JSON.parse(raw) as BookingState) : null;
  } catch { return null; }
}

export async function setBookingState(tenantId: string, sessionId: string, state: BookingState): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(scopedKey('chat:booking', tenantId, sessionId), TTL, JSON.stringify(state));
  } catch { /* non-critical */ }
}

/** Short-TTL cache for the backend's /api/internal/tenant-context/:tenantId
 * response (8 Mongo queries on the backend side) — fetched fresh on every
 * single visitor message today with no caching at all. There is deliberately
 * no cross-service invalidation here (the backend, where a tenant admin's
 * save actually happens, has no way to reach into this service's Redis
 * without new cross-service coupling — confirmed backend/ and ai/ use
 * different Upstash credentials, not provably the same keyspace) — a real,
 * disclosed tradeoff: any tenant config change (settings, a website crawl,
 * a catalog import, a tool-model preset change) can take up to this many
 * seconds to actually take effect. 10s keeps that window small enough to be
 * a non-issue for a real admin testing their own change, while still
 * absorbing most of a real visitor conversation's own back-and-forth
 * (messages seconds apart) on one cache hit. Only successful fetches get
 * cached (the caller never passes a null/failed result in), so a transient
 * backend outage can't poison the cache for the TTL window. */
const TENANT_CONTEXT_CACHE_TTL = 10;

export async function getCachedTenantContext<T = unknown>(tenantId: string): Promise<T | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(`ai:tenantctx:${tenantId}`);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch { return null; }
}

export async function setCachedTenantContext(tenantId: string, context: unknown): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(`ai:tenantctx:${tenantId}`, TENANT_CONTEXT_CACHE_TTL, JSON.stringify(context));
  } catch { /* non-critical */ }
}

async function getCachedJSON<T>(key: string): Promise<T | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch { return null; }
}

async function setCachedJSON(key: string, value: unknown, ttlSeconds: number): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(key, ttlSeconds, JSON.stringify(value));
  } catch { /* non-critical */ }
}

/** Same short-TTL, "a tenant's own edit should show up almost immediately"
 * tradeoff as getCachedTenantContext above — applied to the 4 other
 * per-turn backend lookups (catalog search/details, department/doctor
 * lists) that today hit fresh on every tool call. Intentionally keyed and
 * called from inside backend.client.ts's own methods, same place
 * getTenantContext's own caching already lives, so every caller benefits
 * transparently with zero changes needed to the tool files themselves. */
export function getCachedCatalogSearch<T = unknown>(tenantId: string, query?: string, category?: string): Promise<T | null> {
  return getCachedJSON<T>(`ai:catsearch:${tenantId}:${encodeURIComponent(query ?? '')}:${encodeURIComponent(category ?? '')}`);
}

export function setCachedCatalogSearch(tenantId: string, query: string | undefined, category: string | undefined, items: unknown): Promise<void> {
  return setCachedJSON(`ai:catsearch:${tenantId}:${encodeURIComponent(query ?? '')}:${encodeURIComponent(category ?? '')}`, items, TENANT_CONTEXT_CACHE_TTL);
}

export function getCachedCatalogItem<T = unknown>(tenantId: string, sku: string): Promise<T | null> {
  return getCachedJSON<T>(`ai:catitem:${tenantId}:${encodeURIComponent(sku)}`);
}

export function setCachedCatalogItem(tenantId: string, sku: string, item: unknown): Promise<void> {
  return setCachedJSON(`ai:catitem:${tenantId}:${encodeURIComponent(sku)}`, item, TENANT_CONTEXT_CACHE_TTL);
}

export function getCachedWidgetTeams<T = unknown>(tenantId: string): Promise<T | null> {
  return getCachedJSON<T>(`ai:widgetteams:${tenantId}`);
}

export function setCachedWidgetTeams(tenantId: string, teams: unknown): Promise<void> {
  return setCachedJSON(`ai:widgetteams:${tenantId}`, teams, TENANT_CONTEXT_CACHE_TTL);
}

export function getCachedWidgetStaff<T = unknown>(tenantId: string, teamId: string): Promise<T | null> {
  return getCachedJSON<T>(`ai:widgetstaff:${tenantId}:${encodeURIComponent(teamId)}`);
}

export function setCachedWidgetStaff(tenantId: string, teamId: string, staff: unknown): Promise<void> {
  return setCachedJSON(`ai:widgetstaff:${tenantId}:${encodeURIComponent(teamId)}`, staff, TENANT_CONTEXT_CACHE_TTL);
}

/** Tracks an in-progress/completed website crawl so POST /knowledge/crawl can
 * return immediately (a 20-page crawl can take well past a normal HTTP
 * timeout) while GET /knowledge/crawl-status polls this. One entry per
 * tenant — a second crawl started while one is running just overwrites it,
 * there's no queueing here. Shorter TTL than conversation state (1h, not
 * 24h) since this is transient job status, not something worth keeping. */
const CRAWL_STATUS_TTL = 3600;

export interface WebsiteCrawlStatus {
  status: 'running' | 'completed' | 'failed';
  startedAt: number;
  finishedAt?: number;
  pagesCrawled?: number;
  chunksIngested?: number;
  failures?: Array<{ url: string; reason: string }>;
  error?: string;
}

export async function getCrawlStatus(tenantId: string): Promise<WebsiteCrawlStatus | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(`site:crawl:${tenantId}`);
    return raw ? (JSON.parse(raw) as WebsiteCrawlStatus) : null;
  } catch { return null; }
}

export async function setCrawlStatus(tenantId: string, status: WebsiteCrawlStatus): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(`site:crawl:${tenantId}`, CRAWL_STATUS_TTL, JSON.stringify(status));
  } catch { /* non-critical */ }
}

// ── Per-session turn lock ──────────────────────────────────────────────────
// Text (POST /public/widget/chat) and continuous voice (the in-process
// LeadAgentLLM inside the worker) are two independent entry points into
// runBaseAgent() — nothing without this guards against both processing a
// turn for the identical session at the same moment (a visitor speaking and
// typing within the same instant), which could otherwise produce two
// overlapping AI replies or two near-simultaneous tool calls each believing
// they're "the" call for that turn.
const TURN_LOCK_TTL_MS = 60_000; // generous enough for this codebase's own documented worst-case tool-loop retry latency

/** Attempts to atomically claim the lock for this session — `SET ... NX PX`
 * either succeeds (lock acquired, returns true) or fails because another
 * turn already holds it (returns false). If Redis itself is unavailable,
 * fails OPEN (returns true, i.e. "proceed without serialization") — matching
 * this whole file's own established fail-open posture everywhere else, since
 * losing this guard under a Redis outage is a much smaller risk than
 * blocking every single conversation. */
async function tryAcquireTurnLock(tenantId: string, sessionId: string): Promise<boolean> {
  const client = getRedis();
  if (!client) return true;
  try {
    const result = await client.set(scopedKey('ai:turnlock', tenantId, sessionId), '1', 'PX', TURN_LOCK_TTL_MS, 'NX');
    return result === 'OK';
  } catch { return true; }
}

/** Polls briefly (not indefinitely) for the lock to free up — used when a
 * turn is genuinely already in flight for this session, rather than
 * rejecting the second turn outright. If the lock's own TTL expires while
 * this is polling (the original holder crashed without ever reaching its
 * `finally`), a later poll iteration will simply see the key gone and
 * acquire it — recovery is the TTL itself, not any explicit cleanup the
 * crashed process would have needed to run. Returns `{acquired:false}` if
 * still locked after the whole poll window, in which case the caller
 * returns a graceful "still working on your last message" reply instead of
 * proceeding concurrently. `waitedMs` is real, previously-invisible
 * queueing time — logged by the caller so a "why did this take 30-55s"
 * question can be answered from data instead of guessed at (a burst of
 * real, serialized turns each waiting out a slow predecessor looks very
 * different in this number than one genuinely slow LLM call). */
export async function acquireTurnLock(
  tenantId: string, sessionId: string, maxWaitMs = 3000,
): Promise<{ acquired: boolean; waitedMs: number }> {
  const start = Date.now();
  if (await tryAcquireTurnLock(tenantId, sessionId)) return { acquired: true, waitedMs: 0 };
  const pollIntervalMs = 250;
  const deadline = start + maxWaitMs;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    if (await tryAcquireTurnLock(tenantId, sessionId)) return { acquired: true, waitedMs: Date.now() - start };
  }
  return { acquired: false, waitedMs: Date.now() - start };
}

/** Continuous-voice-only suppression for the turn-lock's own spoken fallback
 * ("I'm still working..."). A burst of near-simultaneous STT-final segments
 * (e.g. a visitor pausing mid-sentence) can legitimately collide on the lock
 * several times in a row — speaking the fallback out loud EVERY time is far
 * more disruptive over TTS than the same collision would be over text.
 * `SET NX PX` — the first collision in the window returns true (caller
 * speaks the fallback and this marks the window); every collision within
 * the next few seconds returns false (caller stays silent instead). Fails
 * open (returns true) on a Redis outage, same posture as the turn lock
 * itself — worst case under an outage is reverting to today's "speak every
 * time" behavior, never a hard failure. */
export async function shouldSpeakTurnLockFallback(tenantId: string, sessionId: string, windowMs = 8000): Promise<boolean> {
  const client = getRedis();
  if (!client) return true;
  try {
    const result = await client.set(scopedKey('ai:turnlock:fallback-said', tenantId, sessionId), '1', 'PX', windowMs, 'NX');
    return result === 'OK';
  } catch { return true; }
}

export async function releaseTurnLock(tenantId: string, sessionId: string): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.del(scopedKey('ai:turnlock', tenantId, sessionId));
  } catch { /* non-critical — the TTL will still expire it */ }
}
