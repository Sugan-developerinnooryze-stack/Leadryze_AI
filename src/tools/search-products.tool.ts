import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';

const schema = z.object({
  query: z.string().optional().describe('What the visitor is looking for, e.g. "butterfly valve" — omit to just browse a category'),
  category: z.string().optional().describe('A product category the visitor mentioned, if any'),
});

export const searchProductsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'search_products',
  description:
    "Search this business's real product catalog (not general website content) — use this when a visitor asks to see, " +
    "find, or browse specific products or a product category. Returns real items with their SKU; " +
    'call get_product_details with a SKU afterward for full specifications.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const items = await backendClient.searchCatalogItems(ctx.tenantId, { query: args.query, category: args.category });
    if (!items.length) {
      return { ok: true, summary: 'No matching products found in the catalog for this query.' };
    }
    const block = items
      .map((p) => `${p.title}${p.sku ? ` (SKU: ${p.sku})` : ''}${p.category ? ` — ${p.category}` : ''}${p.shortDescription ? `: ${p.shortDescription}` : ''}`)
      .join('\n');
    return {
      ok: true,
      summary: `Found ${items.length} matching product(s).`,
      data: { products: block },
    };
  },
};
