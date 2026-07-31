import { v4 as uuidv4 } from 'uuid';
import { chunkText } from './chunker';
import { embedTexts, embedQuery, VECTOR_SIZE } from './embeddings';
import { ensureCollection, upsertVectors, searchVectors, deleteVectors } from './qdrant.client';
import { config } from '../config';
import { logger } from '../utils/logger';

export interface KnowledgeItem {
  id: string;
  tenantId: string;
  title: string;
  content: string;
  type: 'faq' | 'product' | 'policy' | 'document' | 'page' | 'note';
  metadata?: Record<string, unknown>;
}

export interface RetrievedContext {
  content: string;
  score: number;
  source: string;
  type: string;
}

// Voyage AI cosine similarity scores are lower than OpenAI — 0.45 is appropriate
const SIMILARITY_THRESHOLD = 0.45;

export async function ingestKnowledge(
  item: KnowledgeItem
): Promise<{ chunksIngested: number }> {
  await ensureCollection(config.qdrant.collection, VECTOR_SIZE);

  const chunks = await chunkText(item.content, {
    tenantId: item.tenantId,
    knowledgeId: item.id,
    title: item.title,
    type: item.type,
    ...item.metadata,
  });

  if (!chunks.length) return { chunksIngested: 0 };

  const embeddings = await embedTexts(chunks.map((c) => c.content));

  const points = chunks.map((chunk, i) => ({
    id: uuidv4(),
    vector: embeddings[i],
    payload: {
      content: chunk.content,
      tenantId: item.tenantId,
      knowledgeId: item.id,
      title: item.title,
      type: item.type,
      ...chunk.metadata,
    },
  }));

  await upsertVectors(config.qdrant.collection, points);

  logger.info('Knowledge ingested into RAG', {
    knowledgeId: item.id,
    chunks: chunks.length,
    tenantId: item.tenantId,
    type: item.type,
  });

  return { chunksIngested: chunks.length };
}

export async function retrieveContext(
  query: string,
  tenantId: string,
  limit = 5
): Promise<RetrievedContext[]> {
  await ensureCollection(config.qdrant.collection, VECTOR_SIZE);

  const queryVector = await embedQuery(query);

  const results = await searchVectors(config.qdrant.collection, queryVector, limit, {
    must: [{ key: 'tenantId', match: { value: tenantId } }],
  });

  return results
    .filter((r) => r.score >= SIMILARITY_THRESHOLD)
    .map((r) => ({
      content: String(r.payload.content || ''),
      score: r.score,
      source: String(r.payload.title || 'Knowledge Base'),
      type: String(r.payload.type || 'document'),
    }));
}

export async function deleteKnowledge(tenantId: string, knowledgeId: string): Promise<void> {
  await deleteVectors(config.qdrant.collection, {
    must: [
      { key: 'tenantId', match: { value: tenantId } },
      { key: 'knowledgeId', match: { value: knowledgeId } },
    ],
  });
  logger.info('Knowledge deleted', { knowledgeId, tenantId });
}

export async function buildRAGContext(query: string, tenantId: string): Promise<string> {
  const contexts = await retrieveContext(query, tenantId, 5);
  if (!contexts.length) return '';

  const block = contexts
    .map((c, i) => `[Source ${i + 1} — ${c.source}]\n${c.content}`)
    .join('\n\n---\n\n');

  return `KNOWLEDGE BASE CONTEXT (use this to answer accurately — do not fabricate):\n\n${block}\n\nIf the answer is not found above, say you'll connect the customer with a human team member.`;
}

/** Same as buildRAGContext(), but also surfaces the top match's real
 * similarity score — used by the confidence-classification gate
 * (response-confidence.ts) to decide whether a reply is actually grounded
 * rather than trusting the LLM to self-report its own uncertainty. Built
 * from the exact same retrieveContext() call buildRAGContext() already
 * makes, so this costs no extra embedding call when used in its place. */
export async function buildRAGContextWithConfidence(
  query: string,
  tenantId: string
): Promise<{ context: string; topScore: number }> {
  const contexts = await retrieveContext(query, tenantId, 5);
  if (!contexts.length) return { context: '', topScore: 0 };

  const block = contexts
    .map((c, i) => `[Source ${i + 1} — ${c.source}]\n${c.content}`)
    .join('\n\n---\n\n');

  const context = `KNOWLEDGE BASE CONTEXT (use this to answer accurately — do not fabricate):\n\n${block}\n\nIf the answer is not found above, say you'll connect the customer with a human team member.`;
  return { context, topScore: contexts[0].score };
}
