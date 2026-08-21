import { z } from 'zod';
import { llm } from '../core/model-abstraction/llm.provider';
import { QueryPlan, DatasetSchemaContext } from './types';
import { logger } from '../utils/logger';

/** Only reached when fast-path.ts returns null — the deliberately
 * conservative fallback for questions the deterministic pass can't
 * confidently resolve (fuzzy/semantic phrasing, an uncommon comparison
 * shape, a field the fast path doesn't specifically pattern-match). Uses
 * the SAME constrained-output technique extractChatIntent() already
 * proved out in base.agent.ts (llm.generateStructured, primary→fallback),
 * not a new LLM-calling pattern. The model only ever fills in this one
 * schema — it never sees or writes a raw query. */

const FilterConditionSchema = z.object({
  field: z.string(),
  operator: z.enum(['=', '!=', '>', '<', '>=', '<=', 'contains', 'between']),
  value: z.string(),
  value2: z.string().nullable(),
});

const QueryPlanSchema = z.object({
  intent: z.enum(['exact', 'filter', 'semantic', 'hybrid', 'aggregation']),
  filters: z.array(FilterConditionSchema).nullable(),
  sort: z.object({ field: z.string(), direction: z.enum(['asc', 'desc']) }).nullable(),
  aggregation: z.object({ type: z.enum(['count']) }).nullable(),
  semanticQuery: z.string().nullable(),
});

/** search-dataset.tool.ts's caller (tools/runner.ts) hard-kills the WHOLE
 * search_dataset call at 8000ms — but llm.generateStructured()'s own
 * internal primary→fallback sequence bounds EACH attempt at 20000ms, so
 * under degraded provider conditions (confirmed live: both Groq and Gemini
 * rate-limited simultaneously) the classifier's own try/catch below never
 * gets a chance to run its safe semantic-fallback — the outer tool timeout
 * fires first, killing the ENTIRE search with no results at all, when a
 * degraded-but-real semantic search was available the whole time. This
 * inner timeout is deliberately shorter than the tool's 8000ms budget
 * (leaving headroom for the schema-fetch + query-execution HTTP calls that
 * share that same budget) so the classifier reliably falls back to a plain
 * semantic search on the visitor's own words instead of the whole tool
 * call being killed with nothing to show for it. */
const CLASSIFIER_TIMEOUT_MS = 5000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`classifier call timed out after ${ms}ms`)), ms)),
  ]);
}

export async function planQueryWithClassifier(question: string, schema: DatasetSchemaContext): Promise<QueryPlan> {
  const fieldList = schema.columns
    .map((c) => c.semanticRole ? `${c.originalName} (role: ${c.semanticRole})` : c.originalName)
    .join(', ');

  const systemPrompt = `You turn a visitor's question about the dataset "${schema.datasetName}" into a structured query plan — you never answer the question yourself, only classify it.

Available fields: ${fieldList}

intent meanings:
- exact: asking about one specific, named item
- filter: a structured condition on a field with a role (price/date/location/category/identifier) — use the ROLE name as the filter field, not the raw column name
- aggregation: "how many"/count questions — set aggregation.type="count", plus filters if the count is also conditional
- semantic: a fuzzy/descriptive question with no clear field match ("something for high blood pressure") — set semanticQuery to the visitor's own intent in a few words
- hybrid: BOTH a structured filter AND a semantic/fuzzy part

Only use a field name that has a role listed above for filters/sort — if nothing matches, use intent="semantic" instead of guessing a field.`;

  try {
    const result = await withTimeout(
      llm.generateStructured(
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: question },
        ],
        QueryPlanSchema,
      ),
      CLASSIFIER_TIMEOUT_MS,
    );
    return {
      intent: result.intent,
      filters: result.filters?.map((f) => ({ field: f.field, operator: f.operator, value: f.value, value2: f.value2 ?? undefined })) ?? undefined,
      sort: result.sort ?? undefined,
      aggregation: result.aggregation ?? undefined,
      semanticQuery: result.semanticQuery ?? undefined,
      source: 'classifier',
    };
  } catch (err) {
    logger.warn('Dataset query classifier failed — defaulting to semantic search', { error: (err as Error).message, question });
    // A classifier failure should never crash the whole search_dataset
    // call — falling back to a plain semantic search over the visitor's
    // own words is a safe, always-available degradation.
    return { intent: 'semantic', semanticQuery: question, source: 'classifier' };
  }
}
