import type { ToolCallLog } from '../tools/runner';

/**
 * Deterministic, server-side confidence classification for a reply — built
 * because trusting the LLM to self-report its own uncertainty proved
 * unreliable in practice (confirmed live: an identical question asked three
 * times, with a real tool result available every time, only used that
 * result correctly once). Rather than trust the model's own admission of
 * doubt, this reads the same real signals already available — did a
 * product-catalog tool actually match something, did RAG's own similarity
 * score clear a real bar — and decides from those instead.
 */
export interface ConfidenceResult {
  source: 'product_catalog' | 'booking' | 'business_profile' | 'website_rag' | 'general';
  confidence: number;
}

// Tuned to Voyage's actual cosine-similarity range, not a naive 0.7 —
// retrieveContext() already filters out anything below 0.45 (see
// rag/pipeline.ts), so a threshold much higher than that floor would flag
// most real Voyage-scored matches as false-positive "low confidence."
export const CONFIDENCE_THRESHOLD = 0.55;

export function classifyResponseConfidence(
  ragTopScore: number,
  toolCallsLog: ToolCallLog[],
  hasProfileMatch: boolean = false
): ConfidenceResult {
  const productHit = toolCallsLog.find(
    (t) => (t.name === 'search_products' || t.name === 'get_product_details') && t.ok && t.data
  );
  if (productHit) return { source: 'product_catalog', confidence: 0.9 };

  // The visitor asked a profile-shaped question ("tell me about this
  // website", "where are you located"...) AND the tenant has real, crawled
  // profile data to answer it with — checked BEFORE the ambient RAG score
  // below, for the exact same reason the booking branch above is: without
  // this, a genuinely grounded profile answer would otherwise be overridden
  // by an unrelated, near-zero RAG similarity score (the original bug this
  // whole feature exists to fix, just for a different question class).
  if (hasProfileMatch) return { source: 'business_profile', confidence: 0.85 };

  // A successful booking-tool call means real, structured data was found or
  // acted on (real availability, a real confirmed meeting) — same
  // "confident, don't second-guess it" category as a product-catalog hit.
  // Without this branch, these fall through to the ambient RAG score below,
  // which is 0 for any tenant that hasn't crawled their website yet —
  // incorrectly overriding a perfectly good booking answer with the
  // low-confidence deflection message.
  const bookingHit = toolCallsLog.find(
    (t) => (t.name === 'check_meeting_availability' || t.name === 'book_meeting') && t.ok
  );
  if (bookingHit) return { source: 'booking', confidence: 0.9 };

  const ragToolHit = toolCallsLog.find(
    (t) => t.name === 'search_website_knowledge' && t.ok && typeof t.data?.topScore === 'number'
  );
  if (ragToolHit) return { source: 'website_rag', confidence: ragToolHit.data!.topScore as number };

  return { source: 'website_rag', confidence: ragTopScore };
}
