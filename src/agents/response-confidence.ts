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
  source: 'product_catalog' | 'website_rag' | 'general';
  confidence: number;
}

// Tuned to Voyage's actual cosine-similarity range, not a naive 0.7 —
// retrieveContext() already filters out anything below 0.45 (see
// rag/pipeline.ts), so a threshold much higher than that floor would flag
// most real Voyage-scored matches as false-positive "low confidence."
export const CONFIDENCE_THRESHOLD = 0.55;

export function classifyResponseConfidence(
  ragTopScore: number,
  toolCallsLog: ToolCallLog[]
): ConfidenceResult {
  const productHit = toolCallsLog.find(
    (t) => (t.name === 'search_products' || t.name === 'get_product_details') && t.ok && t.data
  );
  if (productHit) return { source: 'product_catalog', confidence: 0.9 };

  const ragToolHit = toolCallsLog.find(
    (t) => t.name === 'search_website_knowledge' && t.ok && typeof t.data?.topScore === 'number'
  );
  if (ragToolHit) return { source: 'website_rag', confidence: ragToolHit.data!.topScore as number };

  return { source: 'website_rag', confidence: ragTopScore };
}
