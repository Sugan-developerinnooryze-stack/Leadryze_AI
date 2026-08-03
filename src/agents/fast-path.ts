/**
 * Fast-path handler — returns instant responses for common non-business messages.
 * These never reach the LLM, saving tokens and latency.
 */

import Fuse from 'fuse.js';
import { checkWebsiteProfileFastPath } from './website-profile-fast-path';
import type { WebsiteProfileSummary } from '../services/backend.client';

export interface FastPathResult {
  handled: boolean;
  response?: string;
  /** Which branch fired — only 'qna'/'profile' answers are eligible for the
   * trailing "want to book / anything else" nudge (base.agent.ts); a
   * greeting/farewell/off-topic reply already has its own closing framing
   * and shouldn't get a second one bolted on. */
  category?: 'qna' | 'profile' | 'other';
}

export interface QnAPairInput {
  question: string;
  answer: string;
  category: string;
}

/** Fuzzy-matches the visitor's message against a tenant's own configured
 * Q&A pairs. Unlike the QnA context injected into the system prompt
 * elsewhere (still a full LLM call), a real match here returns the stored
 * answer directly — zero tokens, same as a greeting. Threshold/shape mirrors
 * the existing contact fuzzy-match in base.agent.ts for consistency. */
function matchQnAPair(message: string, qnaPairs: QnAPairInput[]): QnAPairInput | null {
  if (!qnaPairs.length) return null;
  const fuse = new Fuse(qnaPairs, { keys: ['question'], threshold: 0.4, includeScore: true });
  const matches = fuse.search(message);
  return matches.length > 0 ? matches[0].item : null;
}

/* ── Exact-match sets (normalised to lowercase, punctuation stripped) ── */
const GREETINGS = new Set([
  'hi', 'hello', 'hey', 'hiya', 'howdy', 'heya', 'greetings',
  'good morning', 'good afternoon', 'good evening', 'good day',
  'morning', 'evening', 'afternoon',
]);

const FAREWELLS = new Set([
  'bye', 'goodbye', 'see you', 'see ya', 'cya', 'later', 'take care',
  'have a good day', 'have a nice day', 'goodnight', 'good night', 'night',
  'ttyl', 'talk later',
]);

const ACKNOWLEDGMENTS = new Set([
  'ok', 'okay', 'k', 'sure', 'got it', 'alright', 'understood', 'noted',
  'cool', 'great', 'nice', 'sounds good', 'perfect', 'fine', 'roger',
  'will do', 'makes sense', 'i see', 'i understand', 'no problem', 'np',
]);

const GRATITUDE = new Set([
  'thanks', 'thank you', 'ty', 'thx', 'cheers', 'appreciate it',
  'thank u', 'tysm', 'thank you so much', 'many thanks', 'much appreciated',
  'thanks a lot', 'thanks a bunch',
]);

/* ── Partial-match patterns ── */
const HOW_ARE_YOU_PATTERNS = [
  'how are you', 'how r u', 'hru', 'how do you do', 'how are u',
  'how is it going', 'how are things', 'how you doing', "how ya doin",
  "you're good", 'you good', 'all good',
];

const AGENT_NAME_PATTERNS = [
  'what is your name', "what's your name", 'who are you', 'what are you',
  'tell me about yourself', 'introduce yourself', 'your name', 'who am i talking to',
  'are you a bot', 'are you an ai', 'are you human', 'are you real',
  'what can you do', 'what do you do',
];

/* ── Off-topic regex patterns — clearly not business-related ── */
const OFF_TOPIC_PATTERNS: RegExp[] = [
  // General knowledge / geography
  /^(what is|what's|who is|who's|where is|when did|when was|how many people).{0,30}(capital|population|president|prime minister|king|queen|country|city|planet|star|continent|ocean)/i,
  // Math problems
  /^(solve|calculate|compute|what is)\s+[\d\s+\-*/()]+/i,
  /^\d+\s*[\+\-\*\/]\s*\d+/,
  // Creative writing
  /^(write|compose|create|generate)\s+(a|an|me a|me an)\s+(poem|song|story|essay|joke|riddle|letter|speech|code|script)/i,
  /^tell me a (joke|story|riddle|fun fact|random fact)/i,
  // Sports / news / entertainment
  /^(who|what)\s+(won|is winning|is the winner|lost|is the best)\s+(the|a|at)/i,
  /^(latest|recent|today's|current)\s+(news|update|score|results|match)/i,
  // Coding help (unless it's about their CRM/app)
  /^(write|debug|fix|explain)\s+(this)?\s*(code|function|script|program|class|sql|html|css|javascript|python)/i,
  // Recipes / personal advice
  /^(how to (cook|make|bake|prepare)|recipe for)/i,
  /^(what should i (eat|wear|watch|read|do tonight))/i,
  // Philosophical / random
  /^(what is the meaning of life|are you conscious|do you have feelings|can you feel)/i,
];

function normalise(msg: string): string {
  return msg.toLowerCase().trim().replace(/[!?.,]+$/g, '').replace(/\s+/g, ' ');
}

/**
 * Check if message is a common non-business message that should be answered
 * immediately without hitting the LLM.
 *
 * Call this AFTER loading tenant config (so you have agentName & companyName),
 * but BEFORE the LLM call.
 */
export function checkFastPath(
  message: string,
  agentName: string,
  companyName: string,
  hasConnectors: boolean,
  qnaPairs: QnAPairInput[] = [],
  websiteProfile: WebsiteProfileSummary | null = null,
): FastPathResult {
  const n = normalise(message);

  /* ── Greetings ── */
  if (GREETINGS.has(n) || /^(hi|hello|hey)\s*[!.]*$/.test(n)) {
    const canHelp = hasConnectors
      ? `I can answer questions about your products, invoices, deals, accounts, and any CRM data.`
      : `I can help capture leads, answer questions about ${companyName}, and connect you with the right team.`;
    return {
      handled: true,
      response: `Hello! I'm ${agentName}, the AI assistant for ${companyName}. ${canHelp} What would you like to know?`,
    };
  }

  /* ── Farewells ── */
  if (FAREWELLS.has(n)) {
    return {
      handled: true,
      response: `Goodbye! Feel free to come back anytime. Have a great day!`,
    };
  }

  /* ── Gratitude ── */
  if (GRATITUDE.has(n)) {
    return {
      handled: true,
      response: `You're welcome! Is there anything else I can help you with?`,
    };
  }

  /* ── Short acknowledgments ── */
  if (ACKNOWLEDGMENTS.has(n)) {
    return {
      handled: true,
      response: `Got it! Let me know if you have any other questions.`,
    };
  }

  /* ── Tenant's own FAQ (QnAPair) — real business answer, zero LLM cost.
   * qnaPairs already includes any crawled FAQPage entries by the time this
   * runs (merged at the base.agent.ts call site), so a crawled FAQ gets
   * answered via this exact same matcher — no separate matching code. ── */
  const qnaMatch = matchQnAPair(message, qnaPairs);
  if (qnaMatch) {
    return { handled: true, response: qnaMatch.answer, category: 'qna' };
  }

  /* ── Website Profile — "tell me about this website"/location/hours/
   * services/staff. Only fires if the tenant has a crawled profile AND the
   * matching field actually has data — never match-then-apologize. ── */
  const profileMatch = checkWebsiteProfileFastPath(message, websiteProfile);
  if (profileMatch.handled) return { ...profileMatch, category: 'profile' };

  /* ── How are you ── */
  if (HOW_ARE_YOU_PATTERNS.some((p) => n.includes(p))) {
    return {
      handled: true,
      response: `I'm doing great, thanks for asking! Ready to help with ${companyName}'s data. What would you like to know?`,
    };
  }

  /* ── Who are you / what can you do ── */
  if (AGENT_NAME_PATTERNS.some((p) => n.includes(p))) {
    const abilities = hasConnectors
      ? `I have access to your CRM data and can answer questions about products, pricing, invoices, deals, accounts, contacts, and more.`
      : `I help capture leads, answer questions about ${companyName}, and connect visitors with your team.`;
    return {
      handled: true,
      response: `I'm ${agentName}, an AI assistant for ${companyName}. ${abilities} What can I help you with today?`,
    };
  }

  /* ── Very short / empty / meaningless messages ── */
  if (n.length <= 2 || /^[.!?]+$/.test(n)) {
    return {
      handled: true,
      response: `I didn't quite catch that. Could you ask me something about ${companyName}'s products, pricing, or records?`,
    };
  }

  /* ── Off-topic patterns ── */
  if (OFF_TOPIC_PATTERNS.some((re) => re.test(message))) {
    return {
      handled: true,
      response: `I'm specialised in helping with ${companyName}'s business data — products, invoices, CRM records, and customer information. For general questions like that, a search engine would be more helpful. Is there something about ${companyName} I can assist with?`,
    };
  }

  return { handled: false };
}
