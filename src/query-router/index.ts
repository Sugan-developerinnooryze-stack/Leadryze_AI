import { planQueryFastPath } from './fast-path';
import { planQueryWithClassifier } from './classifier';
import { QueryPlan, DatasetSchemaContext } from './types';

export type { QueryPlan, DatasetSchemaContext, FilterCondition } from './types';

/** The single entry point search-dataset.tool.ts calls — tries the
 * deterministic fast path first (decision #6's whole point: zero LLM
 * calls for the common cases), only invoking the LLM classifier when the
 * fast path can't confidently resolve the question. Neither path ever
 * sees or sets tenantId/datasetId/limit — those are injected server-side
 * by the caller of the resulting plan (dataset-query.service.ts on the
 * backend), never by this module. */
export async function planQuery(question: string, schema: DatasetSchemaContext): Promise<QueryPlan> {
  const fast = planQueryFastPath(question, schema);
  if (fast) return fast;
  return planQueryWithClassifier(question, schema);
}
