import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import { planQuery } from '../query-router';
import { DatasetSchemaContext } from '../query-router/types';
import { screenFieldForInjection } from '../utils/redact-injection';
import { DatasetItemCard } from '../agents/dataset-item-card.types';
import { getDatasetBrowseState, appendDatasetBrowseState } from '../memory/conversation.memory';

const schema = z.object({
  question: z.string().min(1).describe(
    "The visitor's question, in their own words — e.g. \"machines under 5 lakhs\", " +
    '"how many branches in Chennai", "something for high blood pressure". ' +
    'Never rewritten into a database query yourself — pass it through as asked.',
  ),
  // .nullish() — see search-products.tool.ts's own comment on why (Groq
  // sometimes fills an unset optional with an explicit null).
  datasetName: z.string().nullish().describe(
    'The name of the specific uploaded dataset the visitor means, ONLY if this tenant has more than one ' +
    'and the visitor named it — omit when there is only one dataset, or the visitor did not say which.',
  ),
});

const MAX_FIELDS_SHOWN = 6;
const FIELD_VALUE_CAP = 120;

/** `labelMap` relabels DatasetRecord.data's sanitized normalizedName keys
 * back to the tenant's real original header text (hardening Gap 5) — a
 * visitor-facing answer should never show a snake_case internal key. Each
 * value is also screened for prompt-injection content (hardening Gap 7)
 * before it ever reaches the model. */
function formatRecordLine(rec: { recordId: string; data: Record<string, unknown> }, labelMap: Map<string, string>): string {
  const entries = Object.entries(rec.data).filter(([, v]) => v !== undefined && v !== null && v !== '').slice(0, MAX_FIELDS_SHOWN);
  const fields = entries
    .map(([k, v]) => {
      const label = labelMap.get(k) ?? k;
      const safeValue = screenFieldForInjection(label, String(v).slice(0, FIELD_VALUE_CAP), { recordId: rec.recordId });
      return `${label}: ${safeValue}`;
    })
    .join(', ');
  return `- [id: ${rec.recordId}] ${fields}`;
}

const MAX_CARDS = 4;
const MAX_KEY_SPECS = 3;

/** Only http(s):// is ever treated as a real image — rejects javascript:/
 * data:/file:/anything else outright (server-side check; the widget
 * independently re-validates before ever setting img.src, defense in
 * depth). A malformed/unparseable string is also rejected, not passed
 * through. */
function isSafeImageUrl(value: unknown): value is string {
  if (typeof value !== 'string' || !value.trim()) return false;
  try {
    return new URL(value.trim()).protocol === 'http:' || new URL(value.trim()).protocol === 'https:';
  } catch {
    return false;
  }
}

/** Guarantees a card's title is never empty regardless of how sparse or
 * unconventional a tenant's column headers are — directly serving the "any
 * Excel" promise instead of quietly breaking on non-product-shaped data.
 * Priority: (1) the name-role column, (2) the identifier-role column (a
 * code/SKU is a reasonable title when there's no name), (3) the first
 * other non-empty field value present on the record, (4) a synthesized
 * "Record #<id>" fallback. */
function resolveDisplayTitle(
  rec: { recordId: string; data: Record<string, unknown> },
  columns: DatasetSchemaContext['columns'],
  usedKeys: Set<string>,
): string {
  const nameCol = columns.find((c) => c.semanticRole === 'name');
  if (nameCol) {
    const v = rec.data[nameCol.normalizedName];
    if (v !== undefined && v !== null && String(v).trim()) { usedKeys.add(nameCol.normalizedName); return String(v).trim(); }
  }
  const idCol = columns.find((c) => c.semanticRole === 'identifier');
  if (idCol) {
    const v = rec.data[idCol.normalizedName];
    if (v !== undefined && v !== null && String(v).trim()) { usedKeys.add(idCol.normalizedName); return String(v).trim(); }
  }
  for (const [k, v] of Object.entries(rec.data)) {
    if (usedKeys.has(k) || v === undefined || v === null || String(v).trim() === '') continue;
    usedKeys.add(k);
    return String(v).trim();
  }
  return `Record #${rec.recordId}`;
}

/** Builds the bounded, structured item-card list for a widget response —
 * built exclusively from these real query results, never from the LLM's
 * own generated text (see the plan's non-negotiable data-integrity rule).
 * Capped at MAX_CARDS regardless of how many results the query returned;
 * `totalMatches` (results.length, itself already capped server-side by the
 * query executor's own STRUCTURED_LIMIT/SEMANTIC_LIMIT) lets the caller
 * honestly convey "showing N of M" rather than implying MAX_CARDS is the
 * complete result set. */
function buildItemCards(
  results: Array<{ recordId: string; data: Record<string, unknown>; datasetId: string; datasetName: string; datasetVersion: number }>,
  columns: DatasetSchemaContext['columns'],
  labelMap: Map<string, string>,
): DatasetItemCard[] {
  const priceCol = columns.find((c) => c.semanticRole === 'price');
  const imageCol = columns.find((c) => c.semanticRole === 'image');

  return results.slice(0, MAX_CARDS).map((rec) => {
    const usedKeys = new Set<string>();
    const title = resolveDisplayTitle(rec, columns, usedKeys);

    let price: string | undefined;
    if (priceCol) {
      const v = rec.data[priceCol.normalizedName];
      if (v !== undefined && v !== null && String(v).trim()) { price = String(v).trim(); usedKeys.add(priceCol.normalizedName); }
    }

    let imageUrl: string | undefined;
    if (imageCol) {
      const v = rec.data[imageCol.normalizedName];
      if (isSafeImageUrl(v)) { imageUrl = v.trim(); usedKeys.add(imageCol.normalizedName); }
      else if (v !== undefined && v !== null) usedKeys.add(imageCol.normalizedName);
    }

    const keySpecs: string[] = [];
    for (const [k, v] of Object.entries(rec.data)) {
      if (keySpecs.length >= MAX_KEY_SPECS) break;
      if (usedKeys.has(k) || v === undefined || v === null || String(v).trim() === '') continue;
      const label = labelMap.get(k) ?? k;
      const safeValue = screenFieldForInjection(label, String(v).slice(0, FIELD_VALUE_CAP), { recordId: rec.recordId });
      keySpecs.push(`${label}: ${safeValue}`);
    }

    return {
      datasetId: rec.datasetId,
      datasetName: rec.datasetName,
      datasetVersion: rec.datasetVersion,
      recordId: rec.recordId,
      title,
      price,
      imageUrl,
      keySpecs: keySpecs.length ? keySpecs : undefined,
    };
  });
}

/**
 * The single entry point for the whole generic Dataset system (plan decision
 * #8) — the model never chooses structured vs. semantic vs. hybrid itself,
 * it only ever asks a question in plain English. Dataset resolution, query
 * planning (fast path or classifier), and execution all happen server-side;
 * this tool is a thin orchestration shim over backendClient + the query
 * router, mirroring search_products' own shape.
 */
export const searchDatasetTool: AgentTool<z.infer<typeof schema>> = {
  name: 'search_dataset',
  description:
    "Search this business's own uploaded data — machines, services, courses, medical items, price lists, or " +
    "any other business-specific data the tenant has uploaded (distinct from the general Product Catalog and " +
    'website content). Handles exact lookups, filters/ranges ("under X", "in Y"), counts ("how many"), and ' +
    'fuzzy/descriptive questions in ONE call — you never need to decide which kind of question it is yourself.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const datasets = await backendClient.listDatasetsForChatbot(ctx.tenantId);
    if (!datasets.length) {
      return { ok: false, summary: 'No business datasets are configured for this tenant.' };
    }

    const nameHint = cleanArg(args.datasetName)?.toLowerCase();
    const target =
      (nameHint ? datasets.find((d) => d.name.toLowerCase().includes(nameHint)) : undefined) ??
      datasets[0];

    const columns = await backendClient.getDatasetSchema(ctx.tenantId, target.datasetId);
    const plan = await planQuery(args.question, { datasetId: target.datasetId, datasetName: target.name, columns });
    const result = await backendClient.executeDatasetQuery(ctx.tenantId, target.datasetId, plan);

    if (plan.aggregation?.type === 'count') {
      return {
        ok: true,
        summary: `According to ${target.name}, there are ${result.count ?? 0} matching record(s).`,
        data: { datasetName: target.name, count: result.count ?? 0 },
      };
    }

    if (!result.results.length) {
      return { ok: true, summary: `No matching records found in ${target.name} for this question.` };
    }

    // "Show more" fix: exclude records already surfaced as cards this
    // session for this dataset, so a repeat/broader query doesn't re-show
    // the exact same top-N results. Only affects this general list path —
    // an exact single-record lookup is a different, unaffected code path.
    const browseState = await getDatasetBrowseState(ctx.tenantId, ctx.sessionId);
    const alreadyShown = new Set(browseState[target.datasetId] ?? []);
    const unseenResults = result.results.filter((r) => !alreadyShown.has(r.recordId));
    const totalMatches = unseenResults.length;

    if (!unseenResults.length) {
      return {
        ok: true,
        summary: `No new matching records in ${target.name} — you've already been shown all ${result.results.length} that matched.`,
        data: { datasetName: target.name, records: [], items: [], totalMatches: 0 },
      };
    }

    const labelMap = backendClient.buildDatasetLabelMap(columns);
    const lines = unseenResults.map((r) => formatRecordLine(r, labelMap)).join('\n');
    const items = buildItemCards(unseenResults, columns, labelMap);
    await appendDatasetBrowseState(ctx.tenantId, ctx.sessionId, target.datasetId, items.map((i) => i.recordId));
    // `degraded` means the real ranked search failed (embedding-provider
    // hiccup, etc.) and this is a plain, unranked listing from the dataset
    // instead — tell the model so it doesn't claim these are precisely
    // matched, but still let it show them (some real, honestly-labeled
    // options beat "found nothing").
    const summaryPrefix = result.degraded
      ? `A precise search couldn't be completed right now, so here are some items from ${target.name} that may be relevant (not ranked to the exact question):`
      : `According to ${target.name}, found ${unseenResults.length} matching record(s):`;
    return {
      ok: true,
      summary: `${summaryPrefix}\n${lines}`,
      data: { datasetName: target.name, records: unseenResults, items, totalMatches, degraded: result.degraded },
    };
  },
};
