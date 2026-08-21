import { QueryPlan, DatasetSchemaContext, FilterCondition } from './types';

/** Deterministic query planning — NO LLM call. Covers the common, safe,
 * unambiguous cases explicitly called out in the plan (a direct numeric
 * comparison — "below ₹5,000" — and a bare count — "how many X do you
 * have") so the majority of real questions never pay for a classifier
 * round-trip. Anything this can't confidently resolve returns null and
 * falls through to the LLM classifier (classifier.ts) — this is
 * deliberately conservative: a wrong deterministic guess is worse than
 * one extra LLM call, so it only ever fires on patterns it's genuinely
 * sure about. */

// Real, confirmed bug this closes: "below ₹5 lakh" was parsed as literally
// ₹5 (the digit group stops at "5"; "lakh" was never recognized as a
// multiplier), matching nothing in a real Indian-priced dataset where the
// cheapest item was ₹2.5 lakh. Indian lakh/crore shorthand is extremely
// common in real business spreadsheets from this market, not an edge case.
const MULTIPLIER_SUFFIXES: Record<string, number> = {
  lakh: 100_000, lac: 100_000, lakhs: 100_000, lacs: 100_000,
  crore: 10_000_000, cr: 10_000_000, crores: 10_000_000,
  thousand: 1_000, million: 1_000_000,
  k: 1_000, m: 1_000_000,
};
const MULTIPLIER_PATTERN = '(?:\\s*(lakhs?|lacs?|crores?|cr|thousand|million|k|m))?';
const NUMBER_PATTERN = new RegExp(`[₹$€£]?\\s*([\\d,]+(?:\\.\\d+)?)${MULTIPLIER_PATTERN}`, 'i');
const UNDER_PATTERN = /\b(under|below|less than|cheaper than)\b.{0,10}?[₹$€£]?\s*[\d,]+/i;
const OVER_PATTERN  = /\b(over|above|more than|costlier than)\b.{0,10}?[₹$€£]?\s*[\d,]+/i;
const BETWEEN_PATTERN = new RegExp(`\\bbetween\\b\\s*[₹$€£]?\\s*([\\d,]+(?:\\.\\d+)?)${MULTIPLIER_PATTERN}\\s*(?:and|-|to)\\s*[₹$€£]?\\s*([\\d,]+(?:\\.\\d+)?)${MULTIPLIER_PATTERN}`, 'i');
const COUNT_PATTERN = /\bhow many\b/i;

function extractNumber(text: string): number | null {
  const m = text.match(NUMBER_PATTERN);
  if (!m) return null;
  const n = parseFloat(m[1].replace(/,/g, ''));
  if (Number.isNaN(n)) return null;
  const suffix = m[2]?.toLowerCase();
  return suffix ? n * (MULTIPLIER_SUFFIXES[suffix] ?? 1) : n;
}

/** Same multiplier-aware extraction as extractNumber(), but reads a
 * specific capture group index (BETWEEN_PATTERN's two numbers each have
 * their own optional multiplier group interleaved, so a single regex
 * exec's groups are indexed positionally, not re-matched per number). */
function applyMultiplier(rawValue: string, suffix: string | undefined): string {
  const n = parseFloat(rawValue.replace(/,/g, ''));
  if (Number.isNaN(n)) return rawValue;
  const mult = suffix ? MULTIPLIER_SUFFIXES[suffix.toLowerCase()] ?? 1 : 1;
  return String(n * mult);
}

export function planQueryFastPath(question: string, schema: DatasetSchemaContext): QueryPlan | null {
  const hasPriceRole = schema.columns.some((c) => c.semanticRole === 'price');
  const isCount = COUNT_PATTERN.test(question);

  const filters: FilterCondition[] = [];

  if (hasPriceRole) {
    const betweenMatch = question.match(BETWEEN_PATTERN);
    if (betweenMatch) {
      // Groups: 1=number1, 2=multiplier1 (optional), 3=number2, 4=multiplier2 (optional).
      filters.push({
        field: 'price', operator: 'between',
        value: applyMultiplier(betweenMatch[1], betweenMatch[2]),
        value2: applyMultiplier(betweenMatch[3], betweenMatch[4]),
      });
    } else if (UNDER_PATTERN.test(question)) {
      const n = extractNumber(question.slice(question.search(UNDER_PATTERN)));
      if (n !== null) filters.push({ field: 'price', operator: '<', value: String(n) });
    } else if (OVER_PATTERN.test(question)) {
      const n = extractNumber(question.slice(question.search(OVER_PATTERN)));
      if (n !== null) filters.push({ field: 'price', operator: '>', value: String(n) });
    }
  }

  if (isCount) {
    return { intent: 'aggregation', aggregation: { type: 'count' }, filters: filters.length ? filters : undefined, source: 'fast_path' };
  }

  if (filters.length > 0) {
    return { intent: 'filter', filters, source: 'fast_path' };
  }

  // Nothing confident found — fall through to the classifier.
  return null;
}
