/**
 * The constrained shape a `search_dataset` question always resolves to,
 * built either by the deterministic fast path or the LLM classifier
 * fallback (see fast-path.ts/classifier.ts) — the model itself NEVER
 * writes a raw Mongo/Qdrant query, only ever populates this narrow object,
 * and even then only the fields below; tenantId/datasetId/activeVersion
 * are always injected server-side by the caller from real auth context
 * (see dataset-query.service.ts on the backend, which is the only place
 * that actually executes a QueryPlan) — this file has no security fields
 * at all, by design.
 *
 * `FilterCondition` mirrors backend's `IFlowCondition` shape
 * (automation-rule.model.ts) field-for-field — one filter-condition
 * vocabulary in spirit across the codebase, even though `ai/` and
 * `backend/` are separate packages with no shared-types import path.
 */
export type FilterOperator = '=' | '!=' | '>' | '<' | '>=' | '<=' | 'contains' | 'between';

export interface FilterCondition {
  /** A semantic role name (price/location/category/date/identifier) if the
   * dataset's schema mapped one, otherwise a raw original column name. */
  field: string;
  operator: FilterOperator;
  value: string;
  /** Only used by 'between' — the upper bound (value is the lower bound). */
  value2?: string;
}

export type QueryIntent = 'exact' | 'filter' | 'semantic' | 'hybrid' | 'aggregation';

export interface QueryPlan {
  intent: QueryIntent;
  filters?: FilterCondition[];
  sort?: { field: string; direction: 'asc' | 'desc' };
  aggregation?: { type: 'count' };
  semanticQuery?: string;
  /** How this plan was produced — surfaced only for the LLM-call-count
   * verification metric (plan decision #6), never shown to the visitor. */
  source: 'fast_path' | 'classifier';
}

/** The column schema a dataset actually has — passed into both the fast
 * path and the classifier so either one only ever references real,
 * existing fields, never a hallucinated one. */
export interface DatasetSchemaContext {
  datasetId: string;
  datasetName: string;
  columns: Array<{
    originalName: string;
    /** The sanitized key DatasetRecord.data actually uses (hardening Gap
     * 5) — tools use this to relabel a result's data keys back to
     * originalName for display; the query router itself never needs to
     * reference it (filters/sort only ever address a role name). */
    normalizedName: string;
    semanticRole?: 'name' | 'category' | 'price' | 'location' | 'date' | 'description' | 'identifier' | 'image';
  }>;
}
