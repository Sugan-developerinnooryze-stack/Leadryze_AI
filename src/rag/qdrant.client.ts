import { QdrantClient } from '@qdrant/js-client-rest';
import { config } from '../config';
import { logger } from '../utils/logger';

let client: QdrantClient | null = null;

export function getQdrantClient(): QdrantClient {
  if (!client) {
    client = new QdrantClient({
      url: config.qdrant.url,
      ...(config.qdrant.apiKey ? { apiKey: config.qdrant.apiKey } : {}),
    });
    logger.info('Qdrant client initialized', { url: config.qdrant.url });
  }
  return client;
}

export async function ensureCollection(
  collectionName: string,
  vectorSize = 1536
): Promise<void> {
  const qdrant = getQdrantClient();
  let created = false;
  try {
    await qdrant.getCollection(collectionName);
  } catch {
    await qdrant.createCollection(collectionName, {
      vectors: { size: vectorSize, distance: 'Cosine' },
      optimizers_config: { default_segment_number: 2 },
    });
    logger.info('Qdrant collection created', { collectionName, vectorSize });
    created = true;
  }

  // Always ensure payload indexes exist (safe to call on existing collections)
  if (created) {
    await qdrant.createPayloadIndex(collectionName, {
      field_name: 'tenantId',
      field_schema: 'keyword',
    });
    await qdrant.createPayloadIndex(collectionName, {
      field_name: 'type',
      field_schema: 'keyword',
    });
    // Without this, deleteKnowledge()'s tenantId+knowledgeId filter fails
    // outright ("Index required but not found") — every knowledge delete/
    // re-ingest was silently broken until this was added.
    await qdrant.createPayloadIndex(collectionName, {
      field_name: 'knowledgeId',
      field_schema: 'keyword',
    });
    logger.info('Qdrant payload indexes created', { collectionName });
  }
}

export async function upsertVectors(
  collectionName: string,
  points: Array<{ id: string; vector: number[]; payload: Record<string, unknown> }>
): Promise<void> {
  const qdrant = getQdrantClient();
  await qdrant.upsert(collectionName, {
    wait: true,
    points,
  });
}

export async function searchVectors(
  collectionName: string,
  queryVector: number[],
  limit = 5,
  filter?: Record<string, unknown>
): Promise<Array<{ id: string; score: number; payload: Record<string, unknown> }>> {
  const qdrant = getQdrantClient();
  const results = await qdrant.search(collectionName, {
    vector: queryVector,
    limit,
    with_payload: true,
    ...(filter ? { filter } : {}),
  });
  return results.map((r) => ({
    id: String(r.id),
    score: r.score,
    payload: (r.payload as Record<string, unknown>) || {},
  }));
}

export async function deleteVectors(
  collectionName: string,
  filter: Record<string, unknown>
): Promise<void> {
  const qdrant = getQdrantClient();
  await qdrant.delete(collectionName, { wait: true, filter });
}
