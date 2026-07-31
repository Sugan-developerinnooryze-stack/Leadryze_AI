import * as cheerio from 'cheerio';
import { logger } from '../utils/logger';

export interface CrawledPage {
  url: string;
  title: string;
  text: string;
  jsonLd: any[];
}

export interface CrawlFailure {
  url: string;
  reason: string;
}

export interface CrawlOptions {
  maxDepth?: number;
  maxPages?: number;
}

const FETCH_TIMEOUT_MS = 8000;
const MAX_BODY_BYTES = 2 * 1024 * 1024; // 2MB
const USER_AGENT = 'LeadRyzeBot/1.0 (+https://leadryze.ai)';
const BINARY_EXT_RE = /\.(pdf|jpg|jpeg|png|gif|svg|webp|ico|css|js|zip|mp4|mp3|woff2?|ttf|eot|xml|rss)$/i;

async function fetchWithTimeout(url: string): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, {
      signal: controller.signal,
      redirect: 'follow',
      headers: { 'User-Agent': USER_AGENT },
    });
  } finally {
    clearTimeout(timer);
  }
}

/** Minimal, deliberately simple robots.txt check — only looks for a blanket
 * "Disallow: /" under a User-agent: * (or our own UA) block. Real per-path
 * matching is out of scope for this pass; a tenant crawling their own public
 * marketing site is very unlikely to hit a robots.txt that needs more than
 * this to be respected correctly. */
async function isCrawlingBlocked(origin: string): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${origin}/robots.txt`);
    if (!res.ok) return false;
    const body = await res.text();
    const lines = body.split(/\r?\n/).map((l) => l.trim());
    let inRelevantBlock = false;
    for (const line of lines) {
      const [rawKey, ...rest] = line.split(':');
      const key = rawKey?.toLowerCase().trim();
      const value = rest.join(':').trim();
      if (key === 'user-agent') {
        inRelevantBlock = value === '*' || value.toLowerCase().includes('leadryzebot');
        continue;
      }
      if (inRelevantBlock && key === 'disallow' && (value === '/' || value === '')) {
        if (value === '/') return true;
      }
    }
    return false;
  } catch {
    // Unreachable/missing robots.txt — treat as "no restriction", same as
    // most real crawlers' default behavior.
    return false;
  }
}

function extractReadableText(html: string): { title: string; text: string; links: string[] } {
  const $ = cheerio.load(html);
  $('script, style, noscript, svg, iframe, nav, footer, header, form, button').remove();

  const title = $('title').first().text().trim() || $('h1').first().text().trim() || '';
  const main = $('main').length ? $('main') : $('article').length ? $('article') : $('body');
  const text = main
    .text()
    .replace(/[ \t]+/g, ' ')
    .replace(/\n\s*\n+/g, '\n')
    .trim();

  const links: string[] = [];
  $('a[href]').each((_, el) => {
    const href = $(el).attr('href');
    if (href) links.push(href);
  });

  return { title, text, links };
}

/** Reads <script type="application/ld+json"> blocks — must run on its own
 * cheerio.load() of the same html string, separate from
 * extractReadableText()'s own load, since that function removes ALL
 * <script> tags (including these) before extracting visible text. Tolerates
 * malformed JSON per-block (skips it) rather than failing the whole page. */
function extractJsonLd(html: string): any[] {
  const $ = cheerio.load(html);
  const blocks: any[] = [];
  $('script[type="application/ld+json"]').each((_, el) => {
    const raw = $(el).contents().text();
    if (!raw || !raw.trim()) return;
    try {
      const parsed = JSON.parse(raw);
      const entries = Array.isArray(parsed) ? parsed : [parsed];
      for (const entry of entries) {
        if (entry && Array.isArray(entry['@graph'])) blocks.push(...entry['@graph']);
        else if (entry) blocks.push(entry);
      }
    } catch {
      // Malformed JSON-LD on this one block — skip it, not the whole page.
    }
  });
  return blocks;
}

function normalizeUrl(href: string, base: string): string | null {
  try {
    const u = new URL(href, base);
    u.hash = '';
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
    return u.toString();
  } catch {
    return null;
  }
}

/** Same-origin BFS crawl starting from startUrl. Deliberately simple and
 * bounded — this is meant to pull in a tenant's own public marketing/product
 * pages, not to be a general-purpose crawler. */
export async function crawlWebsite(
  startUrl: string,
  opts: CrawlOptions = {}
): Promise<{ pages: CrawledPage[]; failures: CrawlFailure[] }> {
  const maxDepth = opts.maxDepth ?? 2;
  const maxPages = opts.maxPages ?? 20;

  const startOrigin = new URL(startUrl).origin;
  if (await isCrawlingBlocked(startOrigin)) {
    return { pages: [], failures: [{ url: startUrl, reason: 'Blocked by robots.txt' }] };
  }

  const visited = new Set<string>();
  const pages: CrawledPage[] = [];
  const failures: CrawlFailure[] = [];
  let queue: Array<{ url: string; depth: number }> = [{ url: startUrl, depth: 0 }];

  while (queue.length && pages.length < maxPages) {
    const { url, depth } = queue.shift()!;
    if (visited.has(url) || BINARY_EXT_RE.test(url)) continue;
    visited.add(url);

    try {
      const res = await fetchWithTimeout(url);
      const contentType = res.headers.get('content-type') || '';
      if (!res.ok) {
        failures.push({ url, reason: `HTTP ${res.status}` });
        continue;
      }
      if (!contentType.includes('text/html')) {
        failures.push({ url, reason: `Non-HTML content-type: ${contentType}` });
        continue;
      }

      const reader = res.body;
      let html: string;
      if (reader) {
        const buf = await res.arrayBuffer();
        if (buf.byteLength > MAX_BODY_BYTES) {
          failures.push({ url, reason: 'Response too large' });
          continue;
        }
        html = Buffer.from(buf).toString('utf-8');
      } else {
        html = await res.text();
      }

      const { title, text, links } = extractReadableText(html);
      if (text.length >= 40) {
        pages.push({ url, title: title || url, text, jsonLd: extractJsonLd(html) });
      } else {
        failures.push({ url, reason: 'No meaningful text content found' });
      }

      if (depth < maxDepth) {
        for (const href of links) {
          const abs = normalizeUrl(href, url);
          if (!abs) continue;
          if (new URL(abs).origin !== startOrigin) continue;
          if (visited.has(abs) || BINARY_EXT_RE.test(abs)) continue;
          queue.push({ url: abs, depth: depth + 1 });
        }
      }
    } catch (err) {
      failures.push({ url, reason: (err as Error).message });
    }
  }

  logger.info('Website crawl complete', { startUrl, pagesFound: pages.length, failures: failures.length });
  return { pages, failures };
}
