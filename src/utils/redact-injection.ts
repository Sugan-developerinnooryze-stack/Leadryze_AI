import { detectPromptInjection } from '../core/guardrails/moderation';
import { logger } from './logger';

/**
 * Field-level prompt-injection screening for retrieved content (hardening
 * Gap 7) — deliberately per-FIELD, not per-record like
 * formatRecordsForLLM() (base.agent.ts) does for CRM data. Dataset/
 * product/website content is exactly the free-text, less-curated content
 * most likely to trip the regex denylist as a false positive (a genuine
 * usage warning like "do not use this with X" reads similarly to an
 * injection attempt) — and it's also exactly the content most likely to
 * BE the actual answer the visitor asked for. Losing a whole record's
 * name/price because its description field alone tripped the detector is
 * worse than showing this one field redacted while keeping the rest.
 * Logs every match, same convention as formatRecordsForLLM().
 */
export function screenFieldForInjection(fieldLabel: string, value: string, context: Record<string, unknown> = {}): string {
  const check = detectPromptInjection(value);
  if (check.safe) return value;
  logger.warn('Retrieved field redacted: prompt injection pattern detected', { field: fieldLabel, reason: check.reason, ...context });
  return '[content removed — flagged as unsafe]';
}
