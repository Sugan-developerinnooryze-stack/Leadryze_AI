import type { ToolCallLog } from '../tools/runner';
import type { DatasetItemCard } from './dataset-item-card.types';

/**
 * Deterministic, server-side buying-intent classification — same
 * "deterministic first, no new LLM call" discipline as
 * response-confidence.ts and the query-router's own fast-path. Runs every
 * turn, independent of whether a Lead has been created yet, so a visitor
 * who gives contact info early and only expresses real intent several
 * turns later still gets a correctly-updated score (see base.agent.ts's
 * continuous-enrichment call site).
 *
 * Structured signals, not a flat keyword list — specifically to avoid a
 * false positive like "how much does this machine weigh?" reading as high
 * intent just because it contains a commercial-adjacent word.
 */
export interface BuyingIntentResult {
  buyingIntent: 'low' | 'medium' | 'high';
  leadScore: number;
}

// Signal B — commercial/availability-adjacent language, weaker than
// explicit transactional intent, kept as its own separate signal rather
// than merged with Signal C's keyword list.
const COMMERCIAL_SIGNAL = /\b(price|pricing|cost|how much|available|availability|in stock)\b/i;

// Signal C — explicit transactional intent. The quantity sub-pattern
// ("need 5 units", "20 pieces") scores higher than a bare "I want to buy"
// since it signals a real, specific requirement, not just browsing.
const TRANSACTIONAL_SIGNAL = /\b(buy|purchase|quote|quotation|order|purchasing)\b/i;
const QUANTITY_SIGNAL = /\b(\d+)\s*(units?|pieces?|pcs|nos|numbers?|machines?|items?)\b/i;

/** Extracted separately from the scoring logic — also used by
 * base.agent.ts's requirement-capture accumulator (a message flagged as
 * requirement-shaped gets stored verbatim, no LLM call). */
export function isRequirementShaped(message: string): boolean {
  return QUANTITY_SIGNAL.test(message) || /\b(need|require|looking for|for our|requirement)\b/i.test(message);
}

export function classifyBuyingIntent(
  message: string,
  toolCallsLog: ToolCallLog[],
  hasContactInfo: boolean,
): BuyingIntentResult {
  const datasetHit = toolCallsLog.find(
    (t) => t.name === 'search_dataset' && t.ok && Array.isArray(t.data?.items) && (t.data!.items as DatasetItemCard[]).length > 0,
  );
  // Signal A — a specific-record hit, not just a generic browse/no-match.
  const itemIdentified = !!datasetHit;
  const commercial = COMMERCIAL_SIGNAL.test(message);
  const transactional = TRANSACTIONAL_SIGNAL.test(message);
  const quantity = QUANTITY_SIGNAL.test(message);

  let score: number;
  if (transactional || quantity) {
    // A+C — explicit transactional intent. A stated quantity is a stronger,
    // more specific signal than a bare "I want to buy".
    score = quantity ? 85 : 70;
    if (!itemIdentified) score -= 15; // transactional language with no matched record yet is weaker
  } else if (commercial && itemIdentified) {
    score = 45; // A+B
  } else if (commercial) {
    score = 30; // B alone — asked about price/availability with no specific item matched yet
  } else if (itemIdentified) {
    score = 25; // A alone — a specific item was found, but no commercial language
  } else {
    score = 10; // generic question, no signals
  }

  if (hasContactInfo) score += 10; // Signal D
  score = Math.max(0, Math.min(100, score));

  const buyingIntent: BuyingIntentResult['buyingIntent'] = score >= 70 ? 'high' : score >= 35 ? 'medium' : 'low';
  return { buyingIntent, leadScore: score };
}

/** "Keep the highest-seen" merge — a visitor who asked "how much" then
 * later says "I want to buy" should keep the higher signal, never
 * downgrade back to a weaker one from an earlier, less-committal turn. */
export const INTENT_RANK: Record<BuyingIntentResult['buyingIntent'], number> = { low: 0, medium: 1, high: 2 };
export function mergeBuyingIntent(
  prev: { buyingIntent?: BuyingIntentResult['buyingIntent']; leadScore?: number } | undefined,
  next: BuyingIntentResult,
): { result: BuyingIntentResult; isNewHighWaterMark: boolean } {
  if (!prev?.buyingIntent || INTENT_RANK[next.buyingIntent] >= INTENT_RANK[prev.buyingIntent]) {
    return { result: next, isNewHighWaterMark: true };
  }
  return { result: { buyingIntent: prev.buyingIntent, leadScore: Math.max(prev.leadScore ?? 0, next.leadScore) }, isNewHighWaterMark: false };
}
