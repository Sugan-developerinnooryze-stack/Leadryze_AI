import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import axios from 'axios';
import swaggerUi from 'swagger-ui-express';
import swaggerJsdoc from 'swagger-jsdoc';
import { requireApiKey } from './middlewares/api-key.middleware';
import chatRoutes from './api/chat.routes';
import knowledgeRoutes from './api/knowledge.routes';
import templateAnalysisRoutes from './api/template-analysis.routes';
import { logger } from './utils/logger';
import { config } from './config';
import { getRedisClient } from './core/guardrails/rate-limiter';
import { getQdrantClient } from './rag/qdrant.client';

const app = express();

app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN || '*', credentials: true }));
// 15mb (not 10mb) — the Template Analyzer forwards a base64-encoded upload
// (up to backend's 5MB multer ceiling, ~33% larger as base64) plus the
// variable catalog JSON.
app.use(express.json({ limit: '15mb' }));
app.use(express.urlencoded({ extended: true }));

app.use((req, _res, next) => {
  logger.info(`${req.method} ${req.path}`, { ip: req.ip });
  next();
});

const swaggerSpec = swaggerJsdoc({
  definition: {
    openapi: '3.0.0',
    info: { title: 'LeadRyze AI Microservice', version: '1.0.0' },
    components: {
      securitySchemes: {
        ApiKeyAuth: { type: 'apiKey', in: 'header', name: 'x-api-key' },
      },
    },
    security: [{ ApiKeyAuth: [] }],
  },
  apis: ['./src/api/*.ts'],
});

// Swagger docs — only accessible in non-production environments
if (config.app.env !== 'production') {
  app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));
}

// Public liveness check — no sensitive info exposed
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'leadryze-ai', timestamp: new Date().toISOString() });
});

// Detailed health — requires internal API key (used by backend admin panel only)
app.get('/health/detail', requireApiKey, async (_req, res) => {
  const checks: Record<string, 'ok' | 'degraded'> = {};

  try {
    const redis = getRedisClient();
    if (redis) { await redis.ping(); checks.redis = 'ok'; } else { checks.redis = 'degraded'; }
  } catch { checks.redis = 'degraded'; }

  try {
    const qdrant = getQdrantClient();
    await qdrant.getCollections();
    checks.qdrant = 'ok';
  } catch { checks.qdrant = 'degraded'; }

  try {
    await axios.get(`${config.backend.url}/health`, { timeout: 3000 });
    checks.backend = 'ok';
  } catch { checks.backend = 'degraded'; }

  checks.llm = config.anthropic.apiKey || config.openai.apiKey || config.groq.apiKey || config.google.apiKey ? 'ok' : 'degraded';

  const keys = {
    ANTHROPIC_API_KEY:     !!config.anthropic.apiKey,
    OPENAI_API_KEY:        !!config.openai.apiKey,
    GROQ_API_KEY:          !!config.groq.apiKey,
    GOOGLE_API_KEY:        !!config.google.apiKey,
    QDRANT_URL:            !!config.qdrant.url && config.qdrant.url !== 'http://localhost:6333',
    QDRANT_API_KEY:        !!config.qdrant.apiKey,
    VOYAGE_API_KEY:        !!config.voyage.apiKey,
    EMBEDDING_PROVIDER:    config.embeddings.provider,
    EMBEDDING_MODEL:       config.embeddings.model,
    LLM_PROVIDER:          config.llm.provider,
    LLM_MODEL:             config.llm.model,
    LLM_FALLBACK_PROVIDER: config.llm.fallbackProvider,
    LLM_FALLBACK_MODEL:    config.llm.fallbackModel,
  };

  const allOk = Object.values(checks).every((v) => v === 'ok');
  res.status(allOk ? 200 : 207).json({
    status: allOk ? 'ok' : 'degraded',
    service: 'leadryze-ai',
    timestamp: new Date().toISOString(),
    checks,
    keys,
  });
});

app.use('/api', requireApiKey, chatRoutes);
app.use('/api', requireApiKey, knowledgeRoutes);
app.use('/api', requireApiKey, templateAnalysisRoutes);

app.use((_req, res) => {
  res.status(404).json({ success: false, message: 'Not found' });
});

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  logger.error('Unhandled error', { error: err.message, stack: err.stack });
  res.status(500).json({ success: false, message: 'Internal server error' });
});

export default app;
