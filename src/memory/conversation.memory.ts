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

export async function getHistory(sessionId: string): Promise<ConversationMessage[]> {
  const client = getRedis();
  if (!client) return [];
  try {
    const raw = await client.get(`chat:hist:${sessionId}`);
    return raw ? (JSON.parse(raw) as ConversationMessage[]) : [];
  } catch (err) {
    logger.warn('Could not fetch conversation history', { sessionId, error: (err as Error).message });
    return [];
  }
}

export async function appendMessage(
  sessionId: string,
  message: ConversationMessage
): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    const history = await getHistory(sessionId);
    history.push(message);
    const trimmed = history.slice(-MAX_MESSAGES);
    await client.setex(`chat:hist:${sessionId}`, TTL, JSON.stringify(trimmed));
  } catch (err) {
    logger.warn('Could not save conversation message', { sessionId, error: (err as Error).message });
  }
}

export async function clearHistory(sessionId: string): Promise<void> {
  const client = getRedis();
  if (!client) return;
  await client.del(`chat:hist:${sessionId}`);
}

/* ── CRM session state — remembers which module was last discussed ── */
export interface CRMSessionState {
  mode: 'crm' | 'lead';
  channel?: string;
  module?: string;
  recordsBlock?: string; // Formatted records text injected into last response
}

export async function getCRMState(sessionId: string): Promise<CRMSessionState | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(`chat:crm:${sessionId}`);
    return raw ? (JSON.parse(raw) as CRMSessionState) : null;
  } catch { return null; }
}

export async function setCRMState(sessionId: string, state: CRMSessionState): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(`chat:crm:${sessionId}`, TTL, JSON.stringify(state));
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

export async function getPendingIntent(sessionId: string): Promise<PendingIntent | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(`chat:pending:${sessionId}`);
    return raw ? (JSON.parse(raw) as PendingIntent) : null;
  } catch { return null; }
}

export async function setPendingIntent(sessionId: string, intent: PendingIntent): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(`chat:pending:${sessionId}`, PENDING_TTL, JSON.stringify(intent));
  } catch { /* non-critical */ }
}

export async function clearPendingIntent(sessionId: string): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try { await client.del(`chat:pending:${sessionId}`); } catch { /* non-critical */ }
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
  leadCreated?: boolean;
  leadId?:      string;
}

export async function getLeadCaptureState(sessionId: string): Promise<LeadCaptureState | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(`chat:lead:${sessionId}`);
    return raw ? (JSON.parse(raw) as LeadCaptureState) : null;
  } catch { return null; }
}

export async function setLeadCaptureState(sessionId: string, state: LeadCaptureState): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(`chat:lead:${sessionId}`, TTL, JSON.stringify(state));
  } catch { /* non-critical */ }
}

/** Mirrors LeadCaptureState's shape/merge convention. `offeredSlots` is what
 * check_meeting_availability last showed the visitor — book_meeting refuses
 * any startIso not present here (anti-hallucination guard, since the model
 * could otherwise invent a plausible-looking time that was never actually
 * free). `meetingCreated` mirrors leadCreated's idempotency role. */
export interface BookingState {
  offeredSlots?: Array<{ startIso: string; endIso: string }>;
  offeredAt?: number;
  meetingCreated?: boolean;
  meetingId?: string;
}

export async function getBookingState(sessionId: string): Promise<BookingState | null> {
  const client = getRedis();
  if (!client) return null;
  try {
    const raw = await client.get(`chat:booking:${sessionId}`);
    return raw ? (JSON.parse(raw) as BookingState) : null;
  } catch { return null; }
}

export async function setBookingState(sessionId: string, state: BookingState): Promise<void> {
  const client = getRedis();
  if (!client) return;
  try {
    await client.setex(`chat:booking:${sessionId}`, TTL, JSON.stringify(state));
  } catch { /* non-critical */ }
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
