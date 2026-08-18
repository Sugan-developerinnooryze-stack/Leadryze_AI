import Redis from 'ioredis';
import { config } from '../../config';
import { logger } from '../../utils/logger';
import { backendClient } from '../../services/backend.client';

let redis: Redis | null = null;
let redisAvailable = false;

export function getRedisClient(): Redis | null {
  return redis;
}

export async function connectRateLimiterRedis(): Promise<void> {
  const client = config.redis.url
    ? new Redis(config.redis.url, { retryStrategy: (t) => t > 3 ? null : t * 500, enableOfflineQueue: true })
    : new Redis({
        host: config.redis.host,
        port: config.redis.port,
        password: config.redis.password || undefined,
        tls: config.redis.tls ? { rejectUnauthorized: false } : undefined,
        retryStrategy: (t) => t > 3 ? null : t * 500,
        enableOfflineQueue: true,
      });

  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      client.disconnect();
      logger.warn('AI rate-limiter Redis unavailable — rate limiting disabled');
      resolve();
    }, 5000);

    client.once('ready', () => {
      clearTimeout(timer);
      redisAvailable = true;
      redis = client;
      logger.info('Redis connected (AI rate-limiter)');

      client.on('error', () => { redisAvailable = false; });
      client.on('close', () => { redisAvailable = false; });
      client.on('ready', () => { redisAvailable = true; });
      resolve();
    });

    client.once('error', () => {
      clearTimeout(timer);
      client.disconnect();
      logger.warn('AI rate-limiter Redis unavailable — rate limiting disabled');
      resolve();
    });
  });
}

export async function checkTenantRateLimit(
  tenantId: string
): Promise<{ allowed: boolean; remaining: number; resetIn: number }> {
  if (!redis || !redisAvailable) return { allowed: true, remaining: 999, resetIn: 0 };

  const hourBucket = Math.floor(Date.now() / 3600000);
  const key = `ai:rl:${tenantId}:${hourBucket}`;
  const max = config.guardrails.maxRequestsPerTenantPerHour;

  try {
    const current = await redis.incr(key);
    if (current === 1) await redis.expire(key, 3600);
    const ttl = await redis.ttl(key);

    return {
      allowed: current <= max,
      remaining: Math.max(0, max - current),
      resetIn: ttl,
    };
  } catch {
    return { allowed: true, remaining: 999, resetIn: 0 };
  }
}

/**
 * Tenant monthly LLM token-budget check — public widget only (see
 * isPublicVisitor in base.agent.ts; never applies to the internal,
 * staff-authenticated assistant). The durable source of truth is the
 * backend's AiTokenUsage collection; this caches the month-to-date total in
 * Redis for a short TTL so the common case never adds a backend round-trip
 * to response time. A deliberate, disclosed simplification: the cache isn't
 * bumped the instant a response's tokens are recorded, so a rapid burst of
 * messages right at the 100% boundary could let a handful of extra messages
 * through before the cache next refreshes — acceptable for a soft quota,
 * not a hard security boundary.
 */
const TOKEN_QUOTA_CACHE_TTL_SECONDS = 30;

export async function checkTenantTokenQuota(
  tenantId: string,
  monthlyLimit: number
): Promise<{ blocked: boolean; percentUsed: number }> {
  if (!monthlyLimit || monthlyLimit <= 0) return { blocked: false, percentUsed: 0 };

  const cacheKey = `ai:tokenquota:${tenantId}`;

  if (redis && redisAvailable) {
    try {
      const cached = await redis.get(cacheKey);
      if (cached !== null) {
        const usedTokens = parseInt(cached, 10);
        const percentUsed = usedTokens / monthlyLimit;
        return { blocked: percentUsed >= 1, percentUsed };
      }
    } catch { /* fall through to the backend fetch below */ }
  }

  const usedTokens = await backendClient.getTenantTokenUsageThisMonth(tenantId);
  if (redis && redisAvailable) {
    try { await redis.set(cacheKey, String(usedTokens), 'EX', TOKEN_QUOTA_CACHE_TTL_SECONDS); } catch { /* best-effort cache only */ }
  }
  const percentUsed = usedTokens / monthlyLimit;
  return { blocked: percentUsed >= 1, percentUsed };
}

/** Continuous-voice equivalent of checkTenantTokenQuota() above — a
 * completely separate meter (LiveKit room-minutes), checked once per
 * utterance during a call, same short-TTL Redis cache. Since a session's
 * real usage is only known once it CLOSES (not mid-call), this checks
 * accumulated usage from PREVIOUS completed sessions this month, not the
 * in-progress one — a call already underway when the tenant crosses their
 * budget can finish before the NEXT session gets blocked. Same tradeoff
 * checkTenantTokenQuota() already accepts via its own cache staleness
 * window, just coarser (whole sessions instead of whole messages). */
export async function checkTenantVoiceMinutesQuota(
  tenantId: string,
  monthlyLimit: number
): Promise<{ blocked: boolean; percentUsed: number }> {
  if (!monthlyLimit || monthlyLimit <= 0) return { blocked: false, percentUsed: 0 };

  const cacheKey = `ai:voicequota:${tenantId}`;

  if (redis && redisAvailable) {
    try {
      const cached = await redis.get(cacheKey);
      if (cached !== null) {
        const usedMinutes = parseFloat(cached);
        const percentUsed = usedMinutes / monthlyLimit;
        return { blocked: percentUsed >= 1, percentUsed };
      }
    } catch { /* fall through to the backend fetch below */ }
  }

  const usedMinutes = await backendClient.getTenantVoiceMinutesUsageThisMonth(tenantId);
  if (redis && redisAvailable) {
    try { await redis.set(cacheKey, String(usedMinutes), 'EX', TOKEN_QUOTA_CACHE_TTL_SECONDS); } catch { /* best-effort cache only */ }
  }
  const percentUsed = usedMinutes / monthlyLimit;
  return { blocked: percentUsed >= 1, percentUsed };
}

/**
 * Day-bucket sibling for the PDF/Image Template Analyzer — a vision/PDF call
 * costs meaningfully more per-request than a chat turn, so it gets its own,
 * stricter budget rather than sharing the hourly chat bucket. Same fail-open
 * behavior as checkTenantRateLimit if Redis is unavailable — that gap is why
 * the backend also keeps its own Mongo-backed daily count as a backstop.
 */
export async function checkTemplateAnalysisRateLimit(
  tenantId: string
): Promise<{ allowed: boolean; remaining: number; resetIn: number }> {
  if (!redis || !redisAvailable) return { allowed: true, remaining: 999, resetIn: 0 };

  const dayBucket = Math.floor(Date.now() / 86400000);
  const key = `ai:rl:template-analysis:${tenantId}:${dayBucket}`;
  const max = config.guardrails.maxTemplateAnalysesPerTenantPerDay;

  try {
    const current = await redis.incr(key);
    if (current === 1) await redis.expire(key, 86400);
    const ttl = await redis.ttl(key);

    return {
      allowed: current <= max,
      remaining: Math.max(0, max - current),
      resetIn: ttl,
    };
  } catch {
    return { allowed: true, remaining: 999, resetIn: 0 };
  }
}
