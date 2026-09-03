import { extractCapturedData } from '../utils/extract-captured-data';
import { getLeadCaptureState, setLeadCaptureState, type LeadCaptureState } from '../memory/conversation.memory';
import { normalise, stripNameFraming, NON_NAME_WORDS } from './booking-confirmation-shortcut';
import { finalizeWidgetLeadCapture, type WidgetLeadCaptureInput } from './lead-capture-finalize';

export interface RequestQuoteShortcutResult {
  handled: boolean;
  response?: string;
}

/** The widget's own literal button text (see leadryze-widget/src/ui.ts's
 * renderItemCards() — `I'd like a quote for ${item.title}`) — matched first
 * and most reliably, since it's the single most common way this flow
 * actually starts (a click, not typed text). Captures the item name. */
const ITEM_QUOTE_PATTERN = /^i'?d like (?:a|the) quote for (.+?)[.!?]?$/i;

/** Natural variants a visitor might type themselves, or the site's own
 * top-nav "Request a Quote" button (no specific item attached) — a bare
 * trigger, same conservative-list idea as booking's own
 * BARE_BOOKING_INTENT_PHRASES (deliberately small, not the broad
 * CATALOG_ONLY_SIGNALS keyword set from base.agent.ts, which is for tool-
 * narrowing and would over-fire here). */
const BARE_QUOTE_OR_DEMO_PHRASES = [
  'request a quote', 'request quote', 'get a quote', 'send me a quote',
  'i want a quote', "i'd like a quote", 'quote please',
  'request a demo', 'request demo', 'schedule a demo', 'book a demo',
  'i want a demo', "i'd like a demo", 'demo please',
];

/** Real, confirmed production gap this closes: "Request Quote"/"Request
 * Demo" is the highest-intent, highest-stakes click in the entire funnel —
 * a visitor who just clicked it has ALREADY decided they want to be
 * contacted — yet it was previously routed through the exact same general
 * LLM tool-bound pipeline as any other message, fully dependent on
 * Groq/OpenRouter being fast and available. A real, live-observed failure:
 * a genuine provider timeout turned "I'd like a quote for X" into "I'm
 * having trouble processing that right now" — the visitor's interest was
 * never captured at all, not even degraded-but-functional. This shortcut
 * makes name+email collection and Lead creation for a quote/demo request
 * fully deterministic — zero LLM dependency anywhere in this file — mirroring
 * booking-confirmation-shortcut.ts's own "don't leave a common, patterned,
 * high-stakes input to unreliable LLM judgment" reasoning, extended to the
 * conversion moment that matters most for actual lead generation. */
export async function tryRequestQuoteShortcut(
  input: WidgetLeadCaptureInput,
): Promise<RequestQuoteShortcutResult> {
  if (!input.visitorId) return { handled: false };

  const state: LeadCaptureState = (await getLeadCaptureState(input.tenantId, input.sessionId)) ?? {};
  const message = input.message;
  const itemMatch = message.match(ITEM_QUOTE_PATTERN);
  const isBareTrigger = !itemMatch && BARE_QUOTE_OR_DEMO_PHRASES.some((p) => normalise(message).includes(p));
  const alreadyAwaiting = !!state.awaitingQuoteContactFor;

  // Not a quote/demo trigger, and not a reply to a question this shortcut
  // already asked on a prior turn — nothing for this shortcut to do.
  if (!itemMatch && !isBareTrigger && !alreadyAwaiting) return { handled: false };

  const itemName = itemMatch ? itemMatch[1].trim() : (state.awaitingQuoteContactFor || undefined);

  const captured = extractCapturedData(message);
  let firstName = state.firstName;
  let lastName  = state.lastName;
  const email   = captured.email || state.email;
  const phone   = captured.phone || state.phone;
  let newlyExtractedName = false;
  if (captured.name && !firstName) {
    const parts = captured.name.split(/\s+/).filter(Boolean);
    firstName = parts[0];
    if (parts.length > 1) lastName = parts.slice(1).join(' ');
    newlyExtractedName = true;
  }
  // Same bare-name fallback booking's continueBookingFlow() already uses —
  // once we've asked "could I get your name and email?", a real reply is
  // often a bare "Rahul, rahul@x.com" with no "my name is" framing at all.
  if (!firstName && alreadyAwaiting) {
    const firstSegment = stripNameFraming(message).split(/[,;]| and /i)[0].trim();
    if (/^[A-Z][a-zA-Z'-]+(\s+[A-Z][a-zA-Z'-]+)?$/i.test(firstSegment) && !NON_NAME_WORDS.has(normalise(firstSegment))) {
      const parts = firstSegment.split(/\s+/).filter(Boolean);
      firstName = parts[0];
      if (parts.length > 1) lastName = parts.slice(1).join(' ');
      newlyExtractedName = true;
    }
  }

  // Interruptible, same as booking's own equivalent check: we already asked
  // this exact question on a PRIOR turn and this reply contains nothing that
  // looks like an attempted answer — it's a different question entirely, so
  // don't repeat the same canned prompt over it. Leave awaitingQuoteContactFor
  // untouched so the paused flow resumes the next time the visitor actually
  // answers.
  if (alreadyAwaiting && !itemMatch && !isBareTrigger && !captured.email && !captured.phone && !newlyExtractedName) {
    return { handled: false };
  }

  const itemPhrase = itemName ? ` for ${itemName}` : '';

  // A Request Quote/Demo click IS, by definition, explicit transactional
  // intent — the highest-signal action a visitor can take, matching
  // buying-intent.ts's own "explicit transactional" scoring band (70-90).
  // Set directly here rather than relying on classifyBuyingIntent() to
  // infer it: finalizeWidgetLeadCapture() classifies off THAT turn's own
  // message, and the turn where contact info actually arrives (e.g. "John
  // Smith, john@x.com") carries none of the original "I'd like a quote for
  // X" signal on its own — real, confirmed gap, live-verified: without this,
  // a genuine quote request landed as a 'cold'/score-20 Lead. mergeBuyingIntent's
  // "keep the highest-seen" rule (buying-intent.ts) means this floor
  // survives regardless of what that turn's own classification computes.
  const quoteIntentState: Pick<LeadCaptureState, 'buyingIntent' | 'leadScore'> = {
    buyingIntent: 'high', leadScore: Math.max(state.leadScore ?? 0, 85),
  };

  // Email is a hard requirement — not "email or phone" — since the backend's
  // widget-lead-capture endpoint now requires it too (the automatic customer
  // confirmation email, lead-capture-finalize.ts's downstream Phase 4, can't
  // reach a lead with no email). Phone stays optional/best-effort, captured
  // when offered but never blocking.
  if (!firstName || !email) {
    await setLeadCaptureState(input.tenantId, input.sessionId, {
      ...state,
      ...quoteIntentState,
      firstName: firstName || state.firstName,
      lastName:  lastName  || state.lastName,
      email:     email     || state.email,
      phone:     phone     || state.phone,
      awaitingQuoteContactFor: itemName || state.awaitingQuoteContactFor,
    });
    const missing: string[] = [];
    if (!firstName) missing.push('your name');
    if (!email) missing.push('your email address');
    return {
      handled: true,
      response: `Sure! To send you a quote${itemPhrase}, could I get ${missing.join(' and ')}?`,
    };
  }

  // Everything needed is known — capture/enrich the Lead directly, no LLM
  // call anywhere in this path. requirement records the specific ask
  // verbatim (deterministic, matches isRequirementShaped()'s own "store the
  // real message" convention elsewhere in this codebase) so the sales team
  // sees exactly what was requested, not just a generic "interested" note.
  const nextState: LeadCaptureState = {
    ...state,
    ...quoteIntentState,
    firstName, lastName, email, phone,
    requirement: itemName ? `Requested a quote for ${itemName}` : state.requirement,
    awaitingQuoteContactFor: undefined,
  };
  await setLeadCaptureState(input.tenantId, input.sessionId, nextState);
  await finalizeWidgetLeadCapture(input, nextState);

  return {
    handled: true,
    response: `Thanks, ${firstName}! I've passed your request${itemPhrase} to our sales team — they'll reach out to you shortly with a quote.`,
  };
}
