import { createHash } from 'crypto';
import { crawlWebsite } from './web-crawler';
import { ingestKnowledge, deleteKnowledge } from './pipeline';
import { buildWebsiteProfile } from './website-profile-extractor';
import { setCrawlStatus } from '../memory/conversation.memory';
import { backendClient } from '../services/backend.client';
import { logger } from '../utils/logger';

function stableKnowledgeId(tenantId: string, url: string): string {
  return createHash('sha1').update(`${tenantId}:${url}`).digest('hex');
}

/** Maps one schema.org Product JSON-LD entry into the Product Catalog's
 * flexible field shape. Returns null for a non-Product entry (FAQPage,
 * Organization, BreadcrumbList, etc. are ignored here — Product is the only
 * type this pass routes into the structured catalog). */
function mapJsonLdProduct(entry: any): Record<string, unknown> | null {
  if (!entry || typeof entry !== 'object') return null;
  const types = Array.isArray(entry['@type']) ? entry['@type'] : [entry['@type']];
  if (!types.includes('Product')) return null;
  if (!entry.name) return null;

  const images = Array.isArray(entry.image) ? entry.image : entry.image ? [entry.image] : [];
  const specifications: Record<string, unknown> = {};
  const additionalProps = Array.isArray(entry.additionalProperty)
    ? entry.additionalProperty
    : entry.additionalProperty ? [entry.additionalProperty] : [];
  for (const p of additionalProps) {
    if (p && p.name && p.value !== undefined) specifications[String(p.name)] = p.value;
  }

  return {
    title: String(entry.name),
    sku: entry.sku ? String(entry.sku) : undefined,
    category: entry.category ? String(entry.category) : undefined,
    shortDescription: entry.description ? String(entry.description) : undefined,
    images,
    specifications,
  };
}

export interface IngestWebsiteResult {
  pagesCrawled: number;
  chunksIngested: number;
  failures: Array<{ url: string; reason: string }>;
}

/** Crawls a tenant's own website and ingests each page into the SAME RAG
 * pipeline every other knowledge item already uses — no changes to
 * ingestion/storage/retrieval, this only adds a new producer of
 * KnowledgeItems. Re-running this for the same tenant+URL is idempotent: a
 * stable per-URL id means each re-crawl deletes the page's previous chunks
 * before re-inserting, so repeated crawls never accumulate duplicates. */
export async function ingestWebsite(opts: {
  tenantId: string;
  startUrl: string;
  maxPages?: number;
}): Promise<IngestWebsiteResult> {
  const { tenantId, startUrl, maxPages = 20 } = opts;

  await setCrawlStatus(tenantId, { status: 'running', startedAt: Date.now() });
  const catalogSyncStart = Date.now();
  const knowledgeSourceId = await backendClient.startCatalogKnowledgeSource(tenantId, 'website', startUrl);

  try {
    const { pages, failures } = await crawlWebsite(startUrl, { maxDepth: 2, maxPages });

    let chunksIngested = 0;
    for (const page of pages) {
      const id = stableKnowledgeId(tenantId, page.url);
      try {
        await deleteKnowledge(tenantId, id).catch(() => {});
        const result = await ingestKnowledge({
          id,
          tenantId,
          title: page.title,
          content: page.text,
          type: 'page',
          metadata: { sourceUrl: page.url, crawledAt: new Date().toISOString() },
        });
        chunksIngested += result.chunksIngested;
      } catch (err) {
        failures.push({ url: page.url, reason: (err as Error).message });
      }

      // Product JSON-LD → structured catalog (separate from the RAG text
      // above — never embedded into Qdrant, see the plan's STEP 6/3 notes).
      if (knowledgeSourceId) {
        for (const entry of page.jsonLd) {
          const mapped = mapJsonLdProduct(entry);
          if (mapped) {
            await backendClient.upsertCatalogItemFromCrawl(tenantId, knowledgeSourceId, page.url, mapped).catch(() => {});
          }
        }
      }
    }

    // Website Profile — one structured "who we are" doc per tenant, built
    // once per crawl from the SAME already-crawled pages (zero new network
    // calls). Failure here never fails the crawl, same posture as the
    // Product-catalog upsert above.
    if (knowledgeSourceId) {
      try {
        const profileFields = await buildWebsiteProfile(pages);
        await backendClient.upsertWebsiteProfileFromCrawl(tenantId, knowledgeSourceId, profileFields as unknown as Record<string, unknown>);
      } catch (err) {
        logger.warn('Website profile build failed', { tenantId, startUrl, error: (err as Error).message });
      }
    }

    const result: IngestWebsiteResult = { pagesCrawled: pages.length, chunksIngested, failures };
    await setCrawlStatus(tenantId, {
      status: 'completed',
      startedAt: Date.now(),
      finishedAt: Date.now(),
      pagesCrawled: result.pagesCrawled,
      chunksIngested: result.chunksIngested,
      failures: result.failures,
    });
    await backendClient.recordWebsiteCrawlResult(tenantId, result.pagesCrawled);
    if (knowledgeSourceId) {
      await backendClient.finishCatalogKnowledgeSource(knowledgeSourceId, 'completed', Date.now() - catalogSyncStart);
    }
    logger.info('Website ingestion complete', { tenantId, startUrl, ...result, failureCount: failures.length });
    return result;
  } catch (err) {
    await setCrawlStatus(tenantId, {
      status: 'failed',
      startedAt: Date.now(),
      finishedAt: Date.now(),
      error: (err as Error).message,
    });
    if (knowledgeSourceId) {
      await backendClient.finishCatalogKnowledgeSource(knowledgeSourceId, 'failed', Date.now() - catalogSyncStart, (err as Error).message);
    }
    throw err;
  }
}
