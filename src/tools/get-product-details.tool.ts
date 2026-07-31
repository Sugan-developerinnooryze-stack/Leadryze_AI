import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';

const schema = z.object({
  sku: z.string().min(1).describe('The exact SKU of one of the products just found via search_products — never invent one yourself'),
});

export const getProductDetailsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'get_product_details',
  description:
    'Get full details (specifications, description, PDFs, images) for one specific product by its SKU. ' +
    'Always call search_products first to find the right SKU — never guess a SKU.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const item = await backendClient.getCatalogItemBySku(ctx.tenantId, args.sku);
    if (!item) {
      return { ok: false, summary: `No product found with SKU "${args.sku}" — call search_products again to find the correct SKU.` };
    }
    const specLines = item.specifications
      ? Object.entries(item.specifications).map(([k, v]) => `${k}: ${v}`).join('; ')
      : '';
    return {
      ok: true,
      summary: `${item.title}${item.longDescription ? ` — ${item.longDescription}` : ''}${specLines ? ` (Specifications: ${specLines})` : ''}`,
      data: {
        title: item.title,
        specifications: item.specifications,
        pdfs: item.pdfs,
        images: item.images,
      },
    };
  },
};
