/**
 * Narrow, single-purpose backstop used ONLY when OpenAI's own moderation
 * endpoint errors/times out/rate-limits (see moderation.ts's catch block).
 *
 * This deliberately does NOT re-implement prompt-injection, PII, spam, or
 * rate-limiting checks — those are separate, always-on guardrail stages
 * (detectPromptInjection, pii-filter.ts, rate-limiter.ts) that already run on
 * every message regardless of whether OpenAI's moderation API is reachable.
 * The only real gap during a moderation-API outage is OpenAI's own harmful-
 * content classification (violence/self-harm/hate/sexual content), so that's
 * all this covers.
 */

export interface LocalModerationResult {
  safe: boolean;
  flagged: boolean;
  categories: Record<string, boolean>;
  reason?: string;
}

const HARMFUL_PATTERNS: Array<{ category: string; pattern: RegExp }> = [
  { category: 'violence', pattern: /\b(kill|murder|shoot|stab|bomb|attack)\s+(you|him|her|them|myself|someone|people)\b/i },
  { category: 'self_harm', pattern: /\b(kill myself|end my life|suicid\w*|self[\s-]?harm|cut myself)\b/i },
  { category: 'hate', pattern: /\b(nigg\w*|kike|spic|chink|faggot|retard\w*)\b/i },
  { category: 'sexual', pattern: /\b(porn\w*|nude photos?|sex(ual)?\s+(video|content|acts?)|explicit images?)\b/i },
];

export function localModerationFallback(text: string): LocalModerationResult {
  const flaggedCats: Record<string, boolean> = {};
  for (const { category, pattern } of HARMFUL_PATTERNS) {
    if (pattern.test(text)) flaggedCats[category] = true;
  }
  const flagged = Object.keys(flaggedCats).length > 0;
  return {
    safe: !flagged,
    flagged,
    categories: flaggedCats,
    reason: flagged ? `Local fallback flagged: ${Object.keys(flaggedCats).join(', ')}` : undefined,
  };
}
