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

  // Generic Dataset system reuses this SAME collection (a `type:
  // 'dataset_record'` payload value alongside the existing website-page
  // type, not a second collection) — every field the Dataset system ever
  // filters on (datasetId, datasetVersion, recordId — see
  // dataset-index.service.ts's search/delete filters) needs its own index
  // for the same reason `knowledgeId` did above: Qdrant returns a hard 400
  // ("Index required but not found") for a filter on an unindexed field,
  // not just a slow scan. Called unconditionally (not gated on `created`)
  // since this collection already exists in production — `if (created)`
  // above would never retroactively add these to it. try/catch per index
  // since Qdrant errors on re-creating one that already exists; safe to
  // ignore, matching this file's own graceful degradation style.
  const datasetIndexes: Array<{ field_name: string; field_schema: 'keyword' | 'integer' }> = [
    { field_name: 'datasetId', field_schema: 'keyword' },
    { field_name: 'datasetVersion', field_schema: 'integer' },
    { field_name: 'recordId', field_schema: 'keyword' },
  ];
  for (const idx of datasetIndexes) {
    try {
      await qdrant.createPayloadIndex(collectionName, idx);
    } catch {
      // Already exists — fine.
    }
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
