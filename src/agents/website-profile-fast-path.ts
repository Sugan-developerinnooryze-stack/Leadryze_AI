/**
 * Fast-path handler for "tell me about this website"-style questions — the
 * literal, demonstrated failure this whole feature exists to fix (plain RAG
 * chunk-similarity search can never answer a meta-question like this, since
 * no single crawled page's text chunk *is* "a summary of the whole site").
 * Kept separate from fast-path.ts since this needs a WebsiteProfileSummary
 * object no other fast-path branch does. Deliberately NOT a tool — see the
 * plan's own reasoning: a tool call here would add real exposure to the
 * already-disclosed, unresolved second-tool-call reliability gap
 * (llm.provider.ts's PER_CALL_TIMEOUT_MS comment) for the completely
 * realistic "ask about the business, then book a call" conversation.
 */

import type { WebsiteProfileSummary } from '../services/backend.client';
import type { FastPathResult } from './fast-path';

function normalise(msg: string): string {
  return msg.toLowerCase().trim().replace(/[!?.,]+$/g, '').replace(/\s+/g, ' ');
}

interface ProfileCategory {
  patterns: string[];
  respond: (profile: WebsiteProfileSummary) => string | null; // null = field empty, fall through
}

const CATEGORIES: ProfileCategory[] = [
  {
    // About/overview — the exact demonstrated failure, highest priority.
    patterns: [
      'tell me about this website', 'tell me about your company', 'tell me about your business',
      'tell me about yourselves', 'what does this website do', 'what do you guys do',
      'what is this business about', 'what is this website about', 'what does this company do',
      'tell me more about your company', 'what\'s this business about',
    ],
    respond: (p) => (p.summary ? p.summary : null),
  },
  {
    // Location/contact
    patterns: [
      'where are you located', 'where is this located', "what's your address", 'what is your address',
      'how can i contact you', 'how do i contact you', "what's your phone number", 'what is your phone number',
      'what is your email', "what's your email",
    ],
    respond: (p) => {
      const c = p.contact;
      if (!c || (!c.address && !c.phone && !c.email)) return null;
      const parts: string[] = [];
      if (c.address) parts.push(`We're located at ${c.address}`);
      if (c.phone) parts.push(`you can call us at ${c.phone}`);
      if (c.email) parts.push(`or email ${c.email}`);
      return parts.length ? parts.join(', ') + '.' : null;
    },
  },
  {
    // Hours
    patterns: ['what are your hours', 'what are your business hours', 'when are you open', 'are you open'],
    respond: (p) => (p.hours ? `Our hours are: ${p.hours}` : null),
  },
  {
    // Services
    patterns: ['what services do you offer', 'what services do you provide', 'what do you sell', 'what do you offer'],
    respond: (p) => (p.services?.length ? `We offer: ${p.services.join(', ')}.` : null),
  },
  {
    // Staff/team
    patterns: ['who are your doctors', 'who is on your team', 'who works there', 'who are your staff'],
    respond: (p) =>
      p.staff?.length
        ? `Our team includes: ${p.staff.map((s) => (s.title ? `${s.name} (${s.title})` : s.name)).join(', ')}.`
        : null,
  },
];

/** The real fast-path check — only "handles" the message if the matching
 * category AND the corresponding profile field actually has data. Never
 * match-then-apologize: no profile, or an empty field, falls through to
 * today's existing pipeline unchanged. */
export function checkWebsiteProfileFastPath(
  message: string,
  profile: WebsiteProfileSummary | null | undefined
): FastPathResult {
  if (!profile) return { handled: false };
  const n = normalise(message);

  for (const category of CATEGORIES) {
    if (category.patterns.some((p) => n.includes(p))) {
      const response = category.respond(profile);
      if (response) return { handled: true, response };
    }
  }
  return { handled: false };
}

/** Lighter-weight check used only for the confidence gate (Part C) — does
 * this message merely LOOK like a profile question (regardless of whether
 * checkWebsiteProfileFastPath above fully handled it), for the case where
 * broader natural phrasing didn't match a literal pattern but the LLM still
 * answered from the always-on injected profile context. Combined with "does
 * the tenant actually have a non-empty profile" by the caller. */
export function looksLikeProfileQuestion(message: string): boolean {
  const n = normalise(message);
  return CATEGORIES.some((category) => category.patterns.some((p) => n.includes(p)));
}
