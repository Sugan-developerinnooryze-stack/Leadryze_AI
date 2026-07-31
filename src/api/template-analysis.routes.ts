import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { analyzeTemplateDocument } from '../services/template-analysis.service';
import { checkTemplateAnalysisRateLimit } from '../core/guardrails/rate-limiter';
import { logger } from '../utils/logger';

const router = Router();

const CatalogItemSchema = z.object({ key: z.string(), label: z.string(), elemType: z.string() });
const CatalogGroupSchema = z.object({ label: z.string(), items: z.array(CatalogItemSchema) });

const TextRunSchema = z.object({
  text: z.string(), x: z.number(), y: z.number(), fontSize: z.number(), fontName: z.string(),
});
const StructuredContentSchema = z.object({
  textRuns:   z.array(TextRunSchema),
  imageCount: z.number().int().min(0),
});

// Exactly one of {mimetype+fileBase64} (vision path) or {structuredContent}
// (real-PDF text-extraction path, backend already parsed it) is present per
// request — never both, never neither.
const TemplateAnalysisSchema = z.object({
  tenantId:        z.string().min(1),
  docType:         z.enum(['invoice', 'quotation', 'contract', 'workorder']),
  mimetype:        z.enum(['application/pdf', 'image/jpeg', 'image/png', 'image/gif', 'image/webp']).optional(),
  fileBase64:      z.string().min(1).optional(),
  structuredContent: StructuredContentSchema.optional(),
  variableCatalog: z.array(CatalogGroupSchema),
}).refine(
  (b) => !!b.structuredContent !== !!(b.mimetype && b.fileBase64),
  { message: 'Provide exactly one of structuredContent or (mimetype + fileBase64)' },
);

function validate<T>(schema: z.ZodSchema<T>, data: unknown, res: Response): T | null {
  const result = schema.safeParse(data);
  if (!result.success) {
    res.status(400).json({ success: false, message: 'Validation failed', errors: result.error.flatten().fieldErrors });
    return null;
  }
  return result.data;
}

/**
 * @swagger
 * /template-analysis:
 *   post:
 *     summary: Analyze an uploaded invoice/quotation/contract/workorder (PDF or image) into a draft template JSON
 */
router.post('/template-analysis', async (req: Request, res: Response) => {
  const body = validate(TemplateAnalysisSchema, req.body, res);
  if (!body) return;

  const rl = await checkTemplateAnalysisRateLimit(body.tenantId);
  if (!rl.allowed) {
    return res.status(429).json({ success: false, message: `Template analysis rate limit reached. Try again in ${Math.ceil(rl.resetIn / 3600)}h.` });
  }

  try {
    const result = await analyzeTemplateDocument({
      docType: body.docType,
      mimetype: body.mimetype,
      fileBase64: body.fileBase64,
      structuredContent: body.structuredContent,
      variableCatalog: body.variableCatalog,
    });
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Template analysis route error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: (err as Error).message || 'Analysis failed' });
  }
});

export default router;
