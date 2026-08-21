import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import { screenFieldForInjection } from '../utils/redact-injection';

const schema = z.object({
  // .nullish(), not just .optional() — confirmed live elsewhere in this
  // codebase (list_departments's serviceHint) that Groq's function-calling
  // sometimes fills an unset optional string with an explicit `null`
  // rather than omitting it, which .optional() alone rejects, failing the
  // whole tool call before execute() ever runs. See list-departments.tool.ts.
  query: z.string().nullish().describe('What the visitor is looking for, e.g. "butterfly valve" — omit to just browse a category'),
  category: z.string().nullish().describe('A product category the visitor mentioned, if any'),
});

/** Scans a product's normalized specifications for anything price-shaped
 * (key contains "price"/"amount"/"cost"/"fee"/"rate") and surfaces it
 * inline — spec key names are whatever the tenant's own source columns
 * happened to be (e.g. "estimated_price_amount_inr"), never a fixed
 * schema field, so this is a best-effort scan, not an exact-field lookup. */
function priceHint(specifications?: Record<string, string>): string {
  if (!specifications) return '';
  const key = Object.keys(specifications).find((k) => /price|amount|cost|fee|rate/i.test(k));
  return key ? ` [${specifications[key]}]` : '';
}

export const searchProductsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'search_products',
  description:
    "Search this business's real product catalog (not general website content) — use this when a visitor asks to see, " +
    "find, or browse specific products or a product category. Returns real items with their SKU; " +
    'call get_product_details with a SKU afterward for full specifications.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const items = await backendClient.searchCatalogItems(ctx.tenantId, { query: cleanArg(args.query), category: cleanArg(args.category) });
    if (!items.length) {
      return { ok: true, summary: 'No matching products found in the catalog for this query.' };
    }
    // Hardening Gap 7 — a product's title/description are tenant-uploaded,
    // attacker-reachable text; screened per-field, not per-record, so one
    // flagged description doesn't hide a whole product's name/price/SKU.
    const block = items
      .map((p) => {
        const title = screenFieldForInjection('title', p.title, { sku: p.sku });
        const desc = p.shortDescription ? screenFieldForInjection('shortDescription', p.shortDescription, { sku: p.sku }) : undefined;
        return `${title}${p.sku ? ` (SKU: ${p.sku})` : ''}${p.category ? ` — ${p.category}` : ''}${priceHint(p.specifications)}${desc ? `: ${desc}` : ''}`;
      })
      .join('\n');
    return {
      ok: true,
      summary: `Found ${items.length} matching product(s).`,
      data: { products: block },
    };
  },
};
