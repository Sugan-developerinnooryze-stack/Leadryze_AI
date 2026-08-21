import OpenAI from 'openai';
import { config } from '../../config';
import { logger } from '../../utils/logger';
import { localModerationFallback } from './local-moderation-fallback';

// Explicit timeout + zero retries: without these, the SDK's own defaults
// (maxRetries:2 with backoff) mean a 429 from OpenAI's moderation endpoint
// burns several seconds retrying INSIDE this call before the catch block
// below ever gets a chance to fall back locally — confirmed live during an
// actual, sustained moderation-API outage this project hit. Failing fast
// here is what makes the local fallback actually fast.
const openai = new OpenAI({ apiKey: config.openai.apiKey, timeout: 6000, maxRetries: 0 });

/** Short-lived, in-memory circuit breaker — real, confirmed latency waste
 * this closes: every single chat turn was paying for a real network round-
 * trip to OpenAI's moderation endpoint, which is currently GUARANTEED to
 * fail (zero OpenAI billing credits, confirmed this session), before ever
 * falling back locally. After a few consecutive real failures, skip the
 * live call entirely for a cooldown window and go straight to the local
 * fallback — self-heals automatically once the cooldown expires (no manual
 * re-enable needed), so this recovers on its own the moment credits are
 * added or OpenAI's endpoint recovers, rather than staying permanently
 * bypassed. Process-local (module-level state, not Redis) — the worst case
 * of a restart resetting this is a few more real attempts, never a
 * correctness issue, since the local fallback is always safe to use. */
const BREAKER_FAILURE_THRESHOLD = 3;
const BREAKER_COOLDOWN_MS = 5 * 60 * 1000;
let consecutiveFailures = 0;
let breakerOpenUntil = 0;

export interface ModerationResult {
  safe: boolean;
  flagged: boolean;
  categories: Record<string, boolean>;
  reason?: string;
  /** True whenever OpenAI's own moderation call errored and the local
   * keyword-based fallback ran in its place — lets callers log/surface
   * "moderation degraded" separately from an actual flagged message. */
  usedFallback?: boolean;
}

export async function moderateContent(text: string): Promise<ModerationResult> {
  if (!config.guardrails.enableModeration || !config.openai.apiKey) {
    return { safe: true, flagged: false, categories: {} };
  }

  if (Date.now() < breakerOpenUntil) {
    const fallback = localModerationFallback(text);
    return { ...fallback, usedFallback: true };
  }

  try {
    const response = await openai.moderations.create({ input: text });
    const result = response.results[0];
    const flaggedCats: Record<string, boolean> = {};

    if (result.flagged) {
      Object.entries(result.categories).forEach(([k, v]) => {
        if (v) flaggedCats[k] = true;
      });
    }

    consecutiveFailures = 0;
    return {
      safe: !result.flagged,
      flagged: result.flagged,
      categories: flaggedCats,
      reason: result.flagged
        ? `Content flagged: ${Object.keys(flaggedCats).join(', ')}`
        : undefined,
    };
  } catch (err) {
    // OpenAI's moderation endpoint itself is down/rate-limited — this is NOT the same as
    // "the message is unsafe." Fail OPEN with a local fallback check rather than blocking
    // every message tenant-wide (the prior fail-closed behavior silently took the whole
    // chatbot down during any moderation-API outage). Prompt-injection, PII, and rate-limit
    // guardrails are unaffected — they're separate stages that keep running regardless.
    consecutiveFailures++;
    if (consecutiveFailures >= BREAKER_FAILURE_THRESHOLD) {
      breakerOpenUntil = Date.now() + BREAKER_COOLDOWN_MS;
      logger.warn('Moderation API failing repeatedly — skipping live calls for a cooldown window', {
        consecutiveFailures, cooldownMs: BREAKER_COOLDOWN_MS,
      });
    }
    logger.warn('Moderation API unavailable — using local fallback check', { error: (err as Error).message });
    const fallback = localModerationFallback(text);
    return { ...fallback, usedFallback: true };
  }
}

const INJECTION_PATTERNS = [
  /ignore\s+(previous|above|prior|all)\s+(instructions?|context|prompt)/i,
  /disregard\s+(your|the)\s+(instructions?|guidelines?|rules?)/i,
  /act\s+as\s+(?:DAN|jailbreak|unrestricted|evil)/i,
  /you\s+are\s+now\s+(?:DAN|jailbroken|unfiltered)/i,
  /<\s*\/?system\s*>/i,
  /\[\s*system\s*\]/i,
  /\|\|system\|\|/i,
];

export function detectPromptInjection(text: string): { safe: boolean; reason?: string } {
  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(text)) {
      return { safe: false, reason: 'Prompt injection attempt detected' };
    }
  }
  return { safe: true };
}
