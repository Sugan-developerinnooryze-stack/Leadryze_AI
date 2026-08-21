import { v5 as uuidv5 } from 'uuid';
import { config } from '../config';
import { ensureCollection, upsertVectors, searchVectors, deleteVectors } from './qdrant.client';
import { embedTexts, VECTOR_SIZE } from './embeddings';
import { logger } from '../utils/logger';

const DATASET_RECORD_TYPE = 'dataset_record';

// A fixed, generated-once namespace UUID (RFC 4122 v5 requires one) —
// arbitrary but must never change, since changing it would silently
// re-derive every future point id and orphan every already-indexed vector.
const DATASET_POINT_NAMESPACE = '6f1a1f2e-6b8a-4b7a-8f2a-2a7c9d0e4b31';

/** Deterministic Qdrant point id (hardening Gap 3) — Qdrant only accepts a
 * real UUID or an unsigned 64-bit integer as a point id, never an
 * arbitrary string, so the original plan's `${datasetVersion}:${recordId}`
 * was never valid. A random uuidv4() per insert (the first working version
 * of this file) IS valid, but isn't itself idempotent — it needed a
 * separate pre-delete-by-payload-filter step before every insert, and that
 * delete was necessarily non-fatal (a fresh version has nothing to
 * delete), so a transient delete failure could leave a duplicate point
 * behind on a retried/resumed batch. A UUID v5 derived from the record's
 * own real identity sidesteps this entirely: re-indexing the same record
 * — whether a retried batch, a resumed import after a crash (Gap 2), or
 * any other re-run — always computes the SAME id, so Qdrant's own upsert
 * naturally overwrites in place. No pre-delete needed. */
function datasetPointId(tenantId: string, datasetId: string, datasetVersion: number, recordId: string): string {
  return uuidv5(`${tenantId}:${datasetId}:${datasetVersion}:${recordId}`, DATASET_POINT_NAMESPACE);
}

export interface DatasetIndexInput {
  recordId: string;
  semanticText: string;
}

export interface DatasetIndexBatchResult {
  indexed: number;
  failed: number;
}

/** Idempotent batch embed + Qdrant upsert for a Dataset ingestion batch —
 * called (possibly repeatedly, on a resumed/retried ingestion) from the
 * backend's dataset.service.ts. Reuses the SAME Qdrant collection every
 * other knowledge type already uses (a `type:'dataset_record'` payload
 * value, not a second collection). Idempotency comes from the point id
 * itself being deterministic (see datasetPointId() above) — re-running
 * this exact batch always upserts the same points, no pre-delete step
 * needed. Batched (one embed call, one upsert call for the whole batch),
 * not per-record. */
export async function indexDatasetBatch(params: {
  tenantId: string;
  datasetId: string;
  datasetVersion: number;
  records: DatasetIndexInput[];
}): Promise<DatasetIndexBatchResult> {
  const { tenantId, datasetId, datasetVersion, records } = params;
  if (records.length === 0) return { indexed: 0, failed: 0 };

  const collection = config.qdrant.collection;
  await ensureCollection(collection, VECTOR_SIZE);

  try {
    const vectors = await embedTexts(records.map((r) => r.semanticText));
    const points = records.map((r, i) => ({
      id: datasetPointId(tenantId, datasetId, datasetVersion, r.recordId),
      vector: vectors[i],
      payload: {
        tenantId, datasetId, datasetVersion, recordId: r.recordId,
        type: DATASET_RECORD_TYPE,
      },
    }));
    await upsertVectors(collection, points);
    return { indexed: records.length, failed: 0 };
  } catch (err) {
    logger.error('Dataset batch indexing failed', { tenantId, datasetId, datasetVersion, count: records.length, error: (err as Error).message });
    return { indexed: 0, failed: records.length };
  }
}

/** Deletes every vector for one dataset version — used by dataset.service.ts's
 * failed-version cleanup (decision #3: a version that never reaches ready
 * is cleaned up, not left as orphaned data) and by a future "delete this
 * dataset" action. */
export async function deleteDatasetVersionVectors(tenantId: string, datasetId: string, datasetVersion: number): Promise<void> {
  await deleteVectors(config.qdrant.collection, {
    must: [
      { key: 'tenantId', match: { value: tenantId } },
      { key: 'datasetId', match: { value: datasetId } },
      { key: 'datasetVersion', match: { value: datasetVersion } },
    ],
  }).catch((err) => {
    logger.warn('Dataset version vector cleanup failed', { tenantId, datasetId, datasetVersion, error: (err as Error).message });
  });
}

/** Semantic search scoped to one tenant's one dataset's ACTIVE version —
 * the security fields (tenantId/datasetId/datasetVersion) are always
 * passed in by the caller (the query-router's semantic executor, which
 * itself derives them server-side, never from the LLM's output — see the
 * query router's own doc comment for why). */
export async function searchDatasetRecords(
  tenantId: string, datasetId: string, datasetVersion: number, queryText: string, limit = 10,
): Promise<Array<{ recordId: string; score: number }>> {
  const [queryVector] = await embedTexts([queryText]);
  const results = await searchVectors(config.qdrant.collection, queryVector, limit, {
    must: [
      { key: 'tenantId', match: { value: tenantId } },
      { key: 'datasetId', match: { value: datasetId } },
      { key: 'datasetVersion', match: { value: datasetVersion } },
      { key: 'type', match: { value: DATASET_RECORD_TYPE } },
    ],
  });
  return results.map((r) => ({ recordId: String(r.payload.recordId), score: r.score }));
}
