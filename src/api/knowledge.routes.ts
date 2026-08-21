import { Router, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { ingestKnowledge, retrieveContext, deleteKnowledge } from '../rag/pipeline';
import { ingestWebsite } from '../rag/website-ingest.service';
import { indexDatasetBatch, deleteDatasetVersionVectors, searchDatasetRecords } from '../rag/dataset-index.service';
import { getCrawlStatus } from '../memory/conversation.memory';
import { logger } from '../utils/logger';

const router = Router();

const upload = multer({
  dest: path.join(process.cwd(), 'uploads'),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = ['.pdf', '.txt', '.md'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Only PDF, TXT, and MD files are allowed'));
    }
  },
});

/**
 * @swagger
 * /knowledge/ingest:
 *   post:
 *     summary: Ingest text content into the RAG knowledge base
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [tenantId, content, title]
 *             properties:
 *               tenantId: { type: string }
 *               content: { type: string }
 *               title: { type: string }
 *               type: { type: string, enum: [faq, product, policy, document, page, note] }
 *               metadata: { type: object }
 *     responses:
 *       200:
 *         description: Ingestion result
 */
router.post('/knowledge/ingest', async (req: Request, res: Response) => {
  try {
    const { tenantId, content, title, type, metadata } = req.body;

    if (!tenantId || !content || !title) {
      return res.status(400).json({ success: false, message: 'tenantId, content, title required' });
    }

    const result = await ingestKnowledge({
      id: uuidv4(),
      tenantId,
      content,
      title,
      type: type || 'document',
      metadata: metadata || {},
    });

    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Knowledge ingest error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Ingestion failed' });
  }
});

/**
 * @swagger
 * /knowledge/upload:
 *   post:
 *     summary: Upload a PDF or TXT file to the knowledge base
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             required: [tenantId, file]
 *             properties:
 *               tenantId: { type: string }
 *               file:
 *                 type: string
 *                 format: binary
 */
router.post('/knowledge/upload', upload.single('file'), async (req: Request, res: Response) => {
  const filePath = req.file?.path;
  try {
    const { tenantId } = req.body;
    if (!tenantId || !req.file) {
      return res.status(400).json({ success: false, message: 'tenantId and file required' });
    }

    const ext = path.extname(req.file.originalname).toLowerCase();
    let content = '';

    if (ext === '.pdf') {
      const pdfParse = await import('pdf-parse');
      const buffer = fs.readFileSync(req.file.path);
      const parsed = await pdfParse.default(buffer);
      content = parsed.text;
    } else {
      content = fs.readFileSync(req.file.path, 'utf-8');
    }

    if (!content.trim()) {
      return res.status(400).json({ success: false, message: 'File appears to be empty' });
    }

    const result = await ingestKnowledge({
      id: uuidv4(),
      tenantId,
      content,
      title: req.file.originalname,
      type: 'document',
      metadata: { originalName: req.file.originalname, size: req.file.size },
    });

    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Knowledge upload error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Upload processing failed' });
  } finally {
    if (filePath) {
      try { fs.unlinkSync(filePath); } catch {}
    }
  }
});

/**
 * @swagger
 * /knowledge/search:
 *   post:
 *     summary: Search the knowledge base for relevant context
 */
router.post('/knowledge/search', async (req: Request, res: Response) => {
  try {
    const { tenantId, query, limit } = req.body;

    if (!tenantId || !query) {
      return res.status(400).json({ success: false, message: 'tenantId and query required' });
    }

    const results = await retrieveContext(query, tenantId, limit || 5);
    return res.json({ success: true, data: { results, count: results.length } });
  } catch (err) {
    logger.error('Knowledge search error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Search failed' });
  }
});

/**
 * @swagger
 * /knowledge/{knowledgeId}:
 *   delete:
 *     summary: Delete all vectors for a given knowledge document ID
 *     parameters:
 *       - in: path
 *         name: knowledgeId
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: tenantId
 *         required: true
 *         schema: { type: string }
 */
router.delete('/knowledge/:knowledgeId', async (req: Request, res: Response) => {
  try {
    const { knowledgeId } = req.params;
    const { tenantId } = req.query as { tenantId: string };

    if (!tenantId) {
      return res.status(400).json({ success: false, message: 'tenantId query param required' });
    }

    await deleteKnowledge(tenantId, knowledgeId);
    return res.json({ success: true, message: 'Knowledge document deleted' });
  } catch (err) {
    logger.error('Knowledge delete error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Delete failed' });
  }
});

/**
 * @swagger
 * /knowledge/crawl:
 *   post:
 *     summary: Crawl a tenant's own website and ingest its pages into the RAG pipeline
 *     description: |
 *       Returns immediately with { status: 'running' } — a multi-page crawl can
 *       take well past a normal HTTP timeout. Poll GET /knowledge/crawl-status
 *       to see progress/completion. Re-running for the same tenant+URL is
 *       idempotent (re-crawled pages replace their previous chunks, no dupes).
 */
router.post('/knowledge/crawl', async (req: Request, res: Response) => {
  try {
    const { tenantId, startUrl, maxPages } = req.body;
    if (!tenantId || !startUrl) {
      return res.status(400).json({ success: false, message: 'tenantId and startUrl required' });
    }
    try {
      new URL(startUrl);
    } catch {
      return res.status(400).json({ success: false, message: 'startUrl must be a valid absolute URL' });
    }

    // Fire-and-forget — the caller polls /knowledge/crawl-status for the result.
    ingestWebsite({ tenantId, startUrl, maxPages }).catch((err) => {
      logger.error('Website crawl failed', { tenantId, startUrl, error: (err as Error).message });
    });

    return res.json({ success: true, data: { status: 'running' } });
  } catch (err) {
    logger.error('Website crawl kickoff error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Could not start crawl' });
  }
});

/**
 * @swagger
 * /knowledge/crawl-status:
 *   get:
 *     summary: Poll the status of the most recent website crawl for a tenant
 */
router.get('/knowledge/crawl-status', async (req: Request, res: Response) => {
  const tenantId = req.query.tenantId as string;
  if (!tenantId) {
    return res.status(400).json({ success: false, message: 'tenantId query param required' });
  }
  const status = await getCrawlStatus(tenantId);
  return res.json({ success: true, data: status ?? { status: 'idle' } });
});

/**
 * @swagger
 * /knowledge/dataset-index-batch:
 *   post:
 *     summary: Embed + upsert one batch of Dataset records into Qdrant (Generic Dataset system)
 *     description: |
 *       Called synchronously by the backend's dataset ingestion loop, one
 *       batch at a time (not fire-and-forget like /knowledge/crawl — the
 *       backend needs the real indexed/failed counts back to update its own
 *       DatasetVersion progress counters). Idempotent: safe to call again
 *       for the same batch (e.g. resuming an interrupted ingestion) without
 *       creating duplicate vectors.
 */
router.post('/knowledge/dataset-index-batch', async (req: Request, res: Response) => {
  try {
    const { tenantId, datasetId, datasetVersion, records } = req.body as {
      tenantId: string; datasetId: string; datasetVersion: number;
      records: Array<{ recordId: string; semanticText: string }>;
    };
    if (!tenantId || !datasetId || typeof datasetVersion !== 'number' || !Array.isArray(records)) {
      return res.status(400).json({ success: false, message: 'tenantId, datasetId, datasetVersion, records[] required' });
    }
    const result = await indexDatasetBatch({ tenantId, datasetId, datasetVersion, records });
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Dataset index batch error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Dataset batch indexing failed' });
  }
});

/**
 * @swagger
 * /knowledge/dataset-index/{datasetId}/{version}:
 *   delete:
 *     summary: Delete all vectors for one Dataset version (failed-version cleanup)
 */
router.delete('/knowledge/dataset-index/:datasetId/:version', async (req: Request, res: Response) => {
  try {
    const { datasetId, version } = req.params;
    const { tenantId } = req.query as { tenantId: string };
    if (!tenantId) return res.status(400).json({ success: false, message: 'tenantId query param required' });
    await deleteDatasetVersionVectors(tenantId, datasetId, Number(version));
    return res.json({ success: true, message: 'Dataset version vectors deleted' });
  } catch (err) {
    logger.error('Dataset version vector cleanup error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Cleanup failed' });
  }
});

/**
 * @swagger
 * /knowledge/dataset-search:
 *   post:
 *     summary: Semantic search within one tenant's one dataset's active version
 *     description: |
 *       Called by the query router's semantic executor (ai/src/query-router/)
 *       — tenantId/datasetId/datasetVersion are always server-derived by the
 *       caller from auth context, never taken from the LLM's own output.
 */
router.post('/knowledge/dataset-search', async (req: Request, res: Response) => {
  try {
    const { tenantId, datasetId, datasetVersion, query, limit } = req.body as {
      tenantId: string; datasetId: string; datasetVersion: number; query: string; limit?: number;
    };
    if (!tenantId || !datasetId || typeof datasetVersion !== 'number' || !query) {
      return res.status(400).json({ success: false, message: 'tenantId, datasetId, datasetVersion, query required' });
    }
    const results = await searchDatasetRecords(tenantId, datasetId, datasetVersion, query, limit);
    return res.json({ success: true, data: { results } });
  } catch (err) {
    logger.error('Dataset search error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Dataset search failed' });
  }
});

export default router;
