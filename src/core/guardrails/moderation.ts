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

  try {
    const response = await openai.moderations.create({ input: text });
    const result = response.results[0];
    const flaggedCats: Record<string, boolean> = {};

    if (result.flagged) {
      Object.entries(result.categories).forEach(([k, v]) => {
        if (v) flaggedCats[k] = true;
      });
    }

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
