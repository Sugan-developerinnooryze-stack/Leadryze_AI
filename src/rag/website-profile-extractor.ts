import { z } from 'zod';
import { llm } from '../core/model-abstraction/llm.provider';
import { logger } from '../utils/logger';
import type { CrawledPage } from './web-crawler';

export interface WebsiteProfileFields {
  summary?: string;
  services?: string[];
  contact?: { phone?: string; email?: string; address?: string };
  hours?: string;
  staff?: Array<{ name: string; title?: string }>;
  faqs?: Array<{ question: string; answer: string }>;
  fieldSources: Record<string, 'jsonld' | 'llm'>;
}

/** Flattens a schema.org PostalAddress (or a plain string) into one display
 * line — JSON-LD addresses are commonly a nested object, not a string. */
function formatAddress(address: any): string | undefined {
  if (!address) return undefined;
  if (typeof address === 'string') return address;
  const parts = [
    address.streetAddress, address.addressLocality, address.addressRegion,
    address.postalCode, address.addressCountry,
  ].filter(Boolean);
  return parts.length ? parts.join(', ') : undefined;
}

/** Joins schema.org OpeningHoursSpecification entries (or a plain string)
 * into one readable display line. Display-only — never feeds bookable-slot
 * computation, which stays owned by Tenant.widget.booking.hours alone. */
function formatHours(spec: any): string | undefined {
  if (!spec) return undefined;
  if (typeof spec === 'string') return spec;
  const entries = Array.isArray(spec) ? spec : [spec];
  const lines = entries
    .map((e) => {
      const days = Array.isArray(e.dayOfWeek) ? e.dayOfWeek : e.dayOfWeek ? [e.dayOfWeek] : [];
      const dayLabel = days.map((d: string) => String(d).replace(/^.*schema\.org\//, '')).join('/');
      if (!dayLabel || !e.opens || !e.closes) return null;
      return `${dayLabel}: ${e.opens}-${e.closes}`;
    })
    .filter(Boolean);
  return lines.length ? lines.join(', ') : undefined;
}

/** Deterministic, free pass over JSON-LD blocks the crawler already parses
 * for every page (web-crawler.ts's extractJsonLd extracts every @type, but
 * website-ingest.service.ts's mapJsonLdProduct currently discards everything
 * except Product — this reads the same already-parsed Organization/
 * LocalBusiness/FAQPage data instead of letting it go to waste). */
function extractFromJsonLd(pages: CrawledPage[]): { fields: Partial<WebsiteProfileFields>; sources: Record<string, 'jsonld'> } {
  const fields: Partial<WebsiteProfileFields> = {};
  const sources: Record<string, 'jsonld'> = {};
  const staff: Array<{ name: string; title?: string }> = [];
  const faqs: Array<{ question: string; answer: string }> = [];

  for (const page of pages) {
    for (const entry of page.jsonLd) {
      if (!entry || typeof entry !== 'object') continue;
      const types: string[] = Array.isArray(entry['@type']) ? entry['@type'] : [entry['@type']];

      if (types.some((t) => t === 'Organization' || t === 'LocalBusiness')) {
        if (!fields.summary && entry.description) { fields.summary = String(entry.description); sources.summary = 'jsonld'; }
        const phone = entry.telephone;
        const email = entry.email;
        const address = formatAddress(entry.address);
        if (phone || email || address) {
          fields.contact = { ...fields.contact, phone: phone ? String(phone) : fields.contact?.phone, email: email ? String(email) : fields.contact?.email, address: address ?? fields.contact?.address };
          sources.contact = 'jsonld';
        }
        const hours = formatHours(entry.openingHoursSpecification);
        if (hours && !fields.hours) { fields.hours = hours; sources.hours = 'jsonld'; }

        const people = [entry.employee, entry.founder, entry.member].flat().filter(Boolean);
        for (const p of people) {
          if (p?.name) staff.push({ name: String(p.name), title: p.jobTitle ? String(p.jobTitle) : undefined });
        }
      }

      if (types.includes('FAQPage') && Array.isArray(entry.mainEntity)) {
        for (const q of entry.mainEntity) {
          const question = q?.name;
          const answer = q?.acceptedAnswer?.text;
          if (question && answer) faqs.push({ question: String(question), answer: String(answer) });
        }
      }
    }
  }

  if (staff.length) { fields.staff = staff; sources.staff = 'jsonld'; }
  if (faqs.length) { fields.faqs = faqs; sources.faqs = 'jsonld'; }
  return { fields, sources };
}

const PAGE_PRIORITY_RE = /about|contact|team|staff|doctor|service|faq/i;

/** Picks which crawled pages' text to feed the LLM summarization fallback —
 * prioritizes pages whose URL/title suggest they describe the business,
 * falling back to the homepage + first few crawled pages if none match
 * (typical for a small site with no distinctly-named About/Contact page). */
function selectPagesForSummary(pages: CrawledPage[], maxChars: number): string {
  const prioritized = pages.filter((p) => PAGE_PRIORITY_RE.test(p.url) || PAGE_PRIORITY_RE.test(p.title));
  const ordered = prioritized.length ? prioritized : pages.slice(0, 5);

  let text = '';
  for (const page of ordered) {
    const chunk = `\n\n### ${page.title} (${page.url})\n${page.text}`;
    if (text.length + chunk.length > maxChars) break;
    text += chunk;
  }
  return text.trim();
}

const WebsiteProfileSchema = z.object({
  summary: z.string().nullable().describe('A 1-3 sentence overview of what this business/website is and does'),
  services: z.array(z.string()).nullable().describe('Up to 10 services or offerings this business provides'),
  phone: z.string().nullable(),
  email: z.string().nullable(),
  address: z.string().nullable(),
  hours: z.string().nullable().describe('Business/working hours as one readable line, if mentioned anywhere'),
  staff: z.array(z.object({ name: z.string(), title: z.string().nullable() })).nullable().describe('Up to 15 named staff/team members, if mentioned'),
});

/** One-shot, offline structured-output call made once per crawl job — not
 * part of the live, turn-by-turn tool-calling chat path, so it has no
 * relationship to that path's own disclosed multi-tool-call reliability
 * limitation (see llm.provider.ts's PER_CALL_TIMEOUT_MS comment). Only fills
 * fields the JSON-LD pass didn't already supply. */
async function extractViaLLM(
  pages: CrawledPage[],
  alreadyKnown: Partial<WebsiteProfileFields>
): Promise<{ fields: Partial<WebsiteProfileFields>; sources: Record<string, 'llm'> }> {
  const missingSummary = !alreadyKnown.summary;
  const missingServices = !alreadyKnown.services?.length;
  const missingContact = !alreadyKnown.contact?.phone && !alreadyKnown.contact?.email && !alreadyKnown.contact?.address;
  const missingHours = !alreadyKnown.hours;
  const missingStaff = !alreadyKnown.staff?.length;

  if (!missingSummary && !missingServices && !missingContact && !missingHours && !missingStaff) {
    return { fields: {}, sources: {} };
  }

  const excerpt = selectPagesForSummary(pages, 11000);
  if (!excerpt) return { fields: {}, sources: {} };

  try {
    const extraction = await llm.generateStructured(
      [
        {
          role: 'system',
          content:
            'You are extracting a business profile from crawled website text below. Fill in only what is genuinely present — use null for anything not mentioned. Do not guess or fabricate.',
        },
        { role: 'user', content: excerpt },
      ],
      WebsiteProfileSchema
    );

    const fields: Partial<WebsiteProfileFields> = {};
    const sources: Record<string, 'llm'> = {};

    if (missingSummary && extraction.summary) { fields.summary = extraction.summary; sources.summary = 'llm'; }
    if (missingServices && extraction.services?.length) { fields.services = extraction.services; sources.services = 'llm'; }
    if (missingContact && (extraction.phone || extraction.email || extraction.address)) {
      fields.contact = {
        phone: extraction.phone ?? undefined,
        email: extraction.email ?? undefined,
        address: extraction.address ?? undefined,
      };
      sources.contact = 'llm';
    }
    if (missingHours && extraction.hours) { fields.hours = extraction.hours; sources.hours = 'llm'; }
    if (missingStaff && extraction.staff?.length) {
      fields.staff = extraction.staff.map((s) => ({ name: s.name, title: s.title ?? undefined }));
      sources.staff = 'llm';
    }

    return { fields, sources };
  } catch (err) {
    logger.warn('Website profile LLM fallback extraction failed', { error: (err as Error).message });
    return { fields: {}, sources: {} };
  }
}

/** Builds a per-tenant Website Profile from an already-crawled page set —
 * zero new network calls. JSON-LD-derived fields win where present; the LLM
 * fallback only fills genuine gaps (the common case, since most small
 * business sites carry no Organization/LocalBusiness/FAQPage markup). */
export async function buildWebsiteProfile(pages: CrawledPage[]): Promise<WebsiteProfileFields> {
  const { fields: jsonLdFields, sources: jsonLdSources } = extractFromJsonLd(pages);
  const { fields: llmFields, sources: llmSources } = await extractViaLLM(pages, jsonLdFields);

  return {
    summary: jsonLdFields.summary ?? llmFields.summary,
    services: jsonLdFields.services ?? llmFields.services,
    contact: jsonLdFields.contact ?? llmFields.contact,
    hours: jsonLdFields.hours ?? llmFields.hours,
    staff: jsonLdFields.staff ?? llmFields.staff,
    faqs: jsonLdFields.faqs,
    fieldSources: { ...llmSources, ...jsonLdSources },
  };
}
