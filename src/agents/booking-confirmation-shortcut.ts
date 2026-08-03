import { cleanArg } from '../utils/clean-arg';
import { extractCapturedData } from '../utils/extract-captured-data';
import { backendClient } from '../services/backend.client';
import { performBooking } from '../tools/book-meeting.tool';
import { setBookingState, type BookingState, type LeadCaptureState } from '../memory/conversation.memory';

export interface BookingShortcutResult {
  handled: boolean;
  response?: string;
}

/** A short affirmative that only counts as a slot confirmation when exactly
 * one slot was offered — with 2+ slots this is genuinely ambiguous (which
 * one?), handled as a clarifying question instead of a guess. */
const BARE_AFFIRMATIVES = new Set([
  'yes', 'yes please', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure', 'sounds good',
  'perfect', 'that works', 'that works for me', 'book it', 'confirm', 'confirmed',
  "let's do it", 'lets do it', 'go ahead', 'please book it', 'book that',
]);

const ORDINAL_WORDS: Record<string, number> = {
  first: 0, '1st': 0, one: 0,
  second: 1, '2nd': 1, two: 1,
  third: 2, '3rd': 2, three: 2,
  fourth: 3, '4th': 3, four: 3,
  fifth: 4, '5th': 4, five: 4,
};

function normalise(msg: string): string {
  return msg.toLowerCase().trim().replace(/[!.,]+$/g, '').replace(/\s+/g, ' ');
}

/** Extracts the trailing "h:mm AM/PM" token from a slot label formatted by
 * availability.service.ts's own formatLabel() (e.g. "Mon, Aug 3, 1:00 PM"),
 * normalised to 24h "HH:MM" for comparison. */
function labelTimeTo24h(label: string): string | null {
  const m = label.match(/(\d{1,2}):(\d{2})\s*([AP]M)/i);
  if (!m) return null;
  let hour = parseInt(m[1], 10);
  const minute = m[2];
  const ampm = m[3].toUpperCase();
  if (ampm === 'PM' && hour !== 12) hour += 12;
  if (ampm === 'AM' && hour === 12) hour = 0;
  return `${String(hour).padStart(2, '0')}:${minute}`;
}

/** Extracts a time mention from the visitor's own message (e.g. "1pm",
 * "1:00 pm", "13:00"), normalised the same way as labelTimeTo24h(). */
function messageTimeTo24h(message: string): string | null {
  const ampmMatch = message.match(/\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b/i);
  if (ampmMatch) {
    let hour = parseInt(ampmMatch[1], 10);
    const minute = ampmMatch[2] ?? '00';
    const isPm = ampmMatch[3].toLowerCase() === 'p';
    if (isPm && hour !== 12) hour += 12;
    if (!isPm && hour === 12) hour = 0;
    return `${String(hour).padStart(2, '0')}:${minute}`;
  }
  const militaryMatch = message.match(/\b([01]?\d|2[0-3]):([0-5]\d)\b/);
  if (militaryMatch) {
    return `${militaryMatch[1].padStart(2, '0')}:${militaryMatch[2]}`;
  }
  return null;
}

type ResolvedSlot = NonNullable<BookingState['offeredSlots']>[number];

function resolveSlot(message: string, offered: ResolvedSlot[]): ResolvedSlot | 'ambiguous' | null {
  const n = normalise(message);

  const wantedTime = messageTimeTo24h(message);
  if (wantedTime) {
    const matches = offered.filter((s) => s.label && labelTimeTo24h(s.label) === wantedTime);
    if (matches.length === 1) return matches[0];
  }

  for (const [word, idx] of Object.entries(ORDINAL_WORDS)) {
    if (n.includes(word) && offered[idx]) return offered[idx];
  }

  if (offered.length === 1 && BARE_AFFIRMATIVES.has(n)) return offered[0];
  if (offered.length > 1 && BARE_AFFIRMATIVES.has(n)) return 'ambiguous';

  return null;
}

/** Deterministic booking-confirmation shortcut — checked before the LLM
 * tool-loop / confidence-gate machinery ever runs for a turn. Exists because
 * book_meeting is nearly always the SECOND tool call in a booking
 * conversation (after check_meeting_availability), and this session found
 * repeatedly that Groq/Gemini's small models are unreliable at forming a
 * second tool call — the model either returns malformed pseudo-syntax or
 * times out, `runToolLoop()` falls through to a plain unbound reply with no
 * tool calls executed, and the confidence gate (seeing an empty toolCallsLog
 * plus whatever low, irrelevant score the ALWAYS-RUNNING ambient RAG
 * pre-fetch happened to find) incorrectly fires its "I don't have a
 * confident answer" deflection on what was really just a booking
 * confirmation. This shortcut makes the confirmation step deterministic —
 * zero LLM dependency, so it can never hit either failure mode. */
export async function tryBookingConfirmationShortcut(
  message: string,
  ctx: { tenantId: string; sessionId: string; visitorId?: string; pageUrl?: string },
  bookingState: BookingState,
  leadCaptureState: LeadCaptureState,
): Promise<BookingShortcutResult> {
  if (!bookingState.offeredSlots?.length || bookingState.meetingCreated) {
    return { handled: false };
  }

  let slot = bookingState.confirmingSlot ?? null;
  if (!slot) {
    const resolved = resolveSlot(message, bookingState.offeredSlots);
    if (resolved === 'ambiguous') {
      return { handled: true, response: 'Which time works best for you — could you tell me the specific time?' };
    }
    if (!resolved) return { handled: false };
    slot = resolved;
  }

  const captured = extractCapturedData(message);
  let firstName = cleanArg(leadCaptureState.firstName);
  let lastName  = cleanArg(leadCaptureState.lastName);
  const email   = cleanArg(captured.email) || cleanArg(leadCaptureState.email);
  const phone   = cleanArg(captured.phone) || cleanArg(leadCaptureState.phone);
  if (captured.name && !firstName) {
    const parts = captured.name.split(/\s+/).filter(Boolean);
    firstName = parts[0];
    if (parts.length > 1) lastName = parts.slice(1).join(' ');
  }
  // extractCapturedData() only recognises a name framed as "my name is
  // X"/"call me X" — too narrow for THIS specific moment, where we just
  // asked "could I get your name and an email or phone number?" and a real
  // visitor's natural reply is a bare "Rahul, rahul@x.com" with no framing
  // phrase at all. Only applied when we're already mid-confirmation
  // (confirmingSlot was set, i.e. this exact question was just asked) so it
  // never over-triggers on an ordinary, unrelated message elsewhere.
  if (!firstName && bookingState.confirmingSlot) {
    const firstSegment = message.split(/[,;]| and /i)[0].trim();
    if (/^[A-Z][a-zA-Z'-]+(\s+[A-Z][a-zA-Z'-]+)?$/.test(firstSegment)) {
      const parts = firstSegment.split(/\s+/).filter(Boolean);
      firstName = parts[0];
      if (parts.length > 1) lastName = parts.slice(1).join(' ');
    }
  }

  if (!firstName || (!email && !phone)) {
    await setBookingState(ctx.sessionId, { ...bookingState, confirmingSlot: slot });
    return {
      handled: true,
      response: `Great, I can pencil you in for ${slot.label ?? 'that time'}! Could I get your name and an email or phone number to confirm it?`,
    };
  }

  const result = await performBooking({
    tenantId: ctx.tenantId, sessionId: ctx.sessionId, visitorId: ctx.visitorId, pageUrl: ctx.pageUrl,
    slot, firstName, lastName, email, phone, topic: leadCaptureState.service, staffId: bookingState.selectedStaffId,
  });

  if (result.ok) {
    let response = `You're all set! I've booked you for ${slot.label ?? 'the requested time'}${result.staffName ? ` with ${result.staffName}` : ''}. We'll be in touch shortly.`;
    if (!leadCaptureState.service) {
      response += " One more thing — what's the reason for your visit, so our team can prepare?";
    }
    return { handled: true, response };
  }

  if (result.slotTaken) {
    const fresh = await backendClient.getWidgetAvailability(ctx.tenantId, { staffId: bookingState.selectedStaffId });
    if (fresh.length) {
      const shown = fresh.slice(0, 5);
      await setBookingState(ctx.sessionId, {
        ...bookingState,
        offeredSlots: shown.map((s) => ({ startIso: s.startIso, endIso: s.endIso, label: s.label })),
        offeredAt: Date.now(),
        confirmingSlot: undefined,
      });
      return {
        handled: true,
        response: `Sorry, that time was just taken. Here's what's still available: ${shown.map((s) => s.label).join('; ')}.`,
      };
    }
    return { handled: true, response: "Sorry, that time was just taken and nothing else is available right now — please check back shortly." };
  }

  return { handled: true, response: "Sorry, I couldn't complete that booking — please try again in a moment." };
}
