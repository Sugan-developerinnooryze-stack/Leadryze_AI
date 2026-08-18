import { createServer } from 'http';
import app from './app';
import { config } from './config';
import { logger } from './utils/logger';
import { ensureCollection, getQdrantClient } from './rag/qdrant.client';
import { VECTOR_SIZE } from './rag/embeddings';
import { connectRateLimiterRedis } from './core/guardrails/rate-limiter';
import { llm } from './core/model-abstraction/llm.provider';

async function bootstrap() {
  try {
    logger.info('Initializing AI service...');

    try {
      // Delete old collection if it exists with wrong vector size, then recreate
      try {
        const qdrant = getQdrantClient();
        const info = await qdrant.getCollection(config.qdrant.collection);
        const existingSize = (info.config?.params?.vectors as { size?: number })?.size;
        if (existingSize && existingSize !== VECTOR_SIZE) {
          logger.info(`Recreating Qdrant collection (size ${existingSize} → ${VECTOR_SIZE})`);
          await qdrant.deleteCollection(config.qdrant.collection);
        }
      } catch { /* collection doesn't exist yet — that's fine */ }

      await ensureCollection(config.qdrant.collection, VECTOR_SIZE);

      // Ensure tenantId index exists (idempotent — safe to call every startup)
      try {
        const qdrant = getQdrantClient();
        await qdrant.createPayloadIndex(config.qdrant.collection, { field_name: 'tenantId', field_schema: 'keyword' });
        await qdrant.createPayloadIndex(config.qdrant.collection, { field_name: 'type', field_schema: 'keyword' });
      } catch { /* indexes already exist */ }

      logger.info('Qdrant collection ready', { vectorSize: VECTOR_SIZE, provider: config.embeddings.provider });
    } catch (err) {
      logger.warn('Qdrant unavailable — RAG/knowledge-base features disabled', { error: (err as Error).message });
    }

    await connectRateLimiterRedis(); // never throws — logs its own status

    const server = createServer(app);
    const port = config.app.port;
    server.listen(port, () => {
      logger.info(`LeadRyze AI service running on port ${port}`);
      logger.info(`Swagger UI: http://localhost:${port}/api-docs`);
      logger.info(`LLM provider: ${config.llm.provider} / ${config.llm.model}`);
    });

    // Fire-and-forget: never blocks the server from accepting traffic — a
    // dead model should be loud in the logs at boot, not add startup
    // latency (voice/text calls right after a fresh deploy would otherwise
    // wait on it for nothing on the common case where everything is fine).
    llm.checkHealth().catch((err) => {
      logger.error('[LLM HEALTH CHECK] unexpected error running boot-time health check', { error: (err as Error).message });
    });

    const shutdown = async (signal: string) => {
      logger.info(`${signal} received — shutting down AI service`);
      server.close(() => {
        logger.info('HTTP server closed');
        process.exit(0);
      });
      setTimeout(() => process.exit(1), 10000);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('uncaughtException', (err) => {
      logger.error('Uncaught exception', { error: err.message, stack: err.stack });
      process.exit(1);
    });
    process.on('unhandledRejection', (reason) => {
      logger.error('Unhandled rejection', { reason });
      process.exit(1);
    });
  } catch (err) {
    logger.error('Bootstrap failed', { error: (err as Error).message });
    process.exit(1);
  }
}

bootstrap();
