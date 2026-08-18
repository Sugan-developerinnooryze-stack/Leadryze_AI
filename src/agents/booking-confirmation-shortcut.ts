import { cleanArg } from '../utils/clean-arg';
import { extractCapturedData } from '../utils/extract-captured-data';
import { backendClient } from '../services/backend.client';
import { performBooking, assessBookingReadiness } from '../tools/book-meeting.tool';
import { listDoctorsTool } from '../tools/list-doctors.tool';
import { checkMeetingAvailabilityTool } from '../tools/check-availability.tool';
import { setBookingState, setLeadCaptureState, type BookingState, type LeadCaptureState } from '../memory/conversation.memory';
import { labelTimeTo24h, messageTimeTo24h } from '../utils/time-parse';

export interface BookingShortcutResult {
  handled: boolean;
  response?: string;
}

/** The literal quick-reply button text and its common near-variants — a
 * bare "I want to book, no specifics yet" message. Deliberately a small,
 * conservative list (not the full BOOKING_ONLY_SIGNALS keyword set from
 * base.agent.ts, which is intentionally broad for tool-narrowing and would
 * over-fire here) — see tryAvailabilityRequestShortcut's own comment for
 * why this needs to be reliable specifically for the widget's own button. */
const BARE_BOOKING_INTENT_PHRASES = [
  'book an appointment', 'book appointment', 'schedule an appointment',
  'i need an appointment', 'i want to book', 'i want an appointment',
  'make an appointment', 'set up an appointment', 'book a meeting',
];

/** A short affirmative that only counts as a slot confirmation when exactly
 * one slot was offered — with 2+ slots this is genuinely ambiguous (which
 * one?), handled as a clarifying question instead of a guess. */
const BARE_AFFIRMATIVES = new Set([
  'yes', 'yes please', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure', 'sounds good',
  'perfect', 'that works', 'that works for me', 'book it', 'confirm', 'confirmed',
  "let's do it", 'lets do it', 'go ahead', 'please book it', 'book that',
]);

/** Short filler/interjection/negative words that are shaped exactly like a
 * bare first name (1-2 capitalized-or-not alphabetic words) but never
 * actually are one — "hmm"/"nice"/"no" would otherwise be wrongly accepted
 * by the bare-name fallback below. Includes BARE_AFFIRMATIVES too, since
 * "sure"/"ok"/"perfect" are just as name-shaped and just as wrong to accept
 * as a name once resolveSlot() itself isn't the one consuming them (that
 * only happens when a slot isn't already confirmingSlot — see below). */
const NON_NAME_WORDS = new Set([
  ...BARE_AFFIRMATIVES,
  'hi', 'hello', 'hey', 'hmm', 'hm', 'um', 'uh', 'no', 'nope', 'nah', 'maybe',
  'nice', 'cool', 'wow', 'great', 'thanks', 'thank you', 'welcome',
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

/** Strips a leading acknowledgement word ("yeah,"/"sure,") and/or a naming
 * frame ("it's"/"this is"/"my name's"/"i'm") from the FRONT of a message,
 * and a trailing "here"/"speaking" from the end — so the bare-name fallback
 * below can recognise a real, natural spoken answer like "Yeah, it's Kumar"
 * or "Kumar here", not just a completely bare "Kumar". A real, confirmed gap
 * this closes: a visitor asked "could I get your name?" almost never replies
 * with the name alone — the ONLY phrasing the old strict regex accepted —
 * and voice in particular never produces a bare one-word answer, so the
 * fallback silently declined to handle nearly every real reply, falling
 * through to the general LLM path, which has no equivalent persistence at
 * all (it can verbally repeat the name back without ever saving it). */
function stripNameFraming(msg: string): string {
  return msg
    .replace(/^\s*(?:yeah|yes|yep|yup|sure|okay|ok|well|so|alright)[,.]?\s*/i, '')
    .replace(/^\s*(?:it'?s|this is|that'?s|my name'?s|my name is|the name'?s|the name is|name'?s|name is|i'?m called|i'?m|i am|call me)\s*/i, '')
    .replace(/\s+(?:here|speaking)\s*$/i, '')
    .trim();
}

/** Splits a message into individual clauses on sentence/clause-ending
 * punctuation (. ! ? ,) so a reply like "Yeah. Sounds good." or "Sure, that
 * works." — each really just one bare affirmative plus a throwaway second
 * clause — is recognised. normalise() alone only strips TRAILING
 * punctuation, so "Yeah. Sounds good." normalises to the single blob "yeah.
 * sounds good", which matches no whole-string BARE_AFFIRMATIVES entry and
 * silently falls through to the fragile general LLM path (the confirmed
 * root cause of a real production bug — see hasBareAffirmativeClause()
 * below). Each clause is still required to be an EXACT match against
 * BARE_AFFIRMATIVES, so a genuinely unrelated multi-clause message doesn't
 * false-positive just because it happens to contain punctuation. */
function splitClauses(msg: string): string[] {
  return msg
    .split(/[.!?,]+/)
    .map((clause) => normalise(clause))
    .filter(Boolean);
}

/** True when ANY clause of the message is, on its own, an exact
 * BARE_AFFIRMATIVES match — the fix for the punctuation bug above. */
function hasBareAffirmativeClause(msg: string): boolean {
  return splitClauses(msg).some((clause) => BARE_AFFIRMATIVES.has(clause));
}

type ResolvedSlot = NonNullable<BookingState['offeredSlots']>[number];

const MONTH_NAMES: Record<string, number> = {
  jan: 0, january: 0, feb: 1, february: 1, mar: 2, march: 2, apr: 3, april: 3,
  may: 4, jun: 5, june: 5, jul: 6, july: 6, aug: 7, august: 7, sep: 8, sept: 8, september: 8,
  oct: 9, october: 9, nov: 10, november: 10, dec: 11, december: 11,
};

/** Extracts an explicit day qualifier from the message and returns the
 * calendar date (YYYY-MM-DD) it refers to — null if the message names none.
 * Handles "today"/"tomorrow", a weekday name, and an explicit "Month Day"
 * mention ("Aug 18", "August 18th") — the last one added after a real,
 * confirmed gap: once the AI names a date back to the visitor, their own
 * natural follow-up is often to repeat that exact "Aug 18" phrasing rather
 * than say "tomorrow" again, and that wasn't recognised at all — falling to
 * the general LLM path, which was then observed live losing the date
 * entirely on a retry after a real booking conflict. Same UTC-based day
 * math the rest of this deterministic path already uses (no tenant
 * timezone available here) — good enough to tell one real day from another,
 * which is the actual bug this closes. */
function extractDayQualifierDate(message: string): string | null {
  const n = message.toLowerCase();
  const now = new Date();
  const toKey = (d: Date) => d.toISOString().slice(0, 10);
  if (/\btoday\b/.test(n)) return toKey(now);
  if (/\btomorrow\b/.test(n)) return toKey(new Date(now.getTime() + 86_400_000));
  const weekdays = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  for (let i = 0; i < weekdays.length; i++) {
    if (n.includes(weekdays[i])) {
      let delta = (i - now.getUTCDay() + 7) % 7;
      if (delta === 0) delta = 7; // naming today's own weekday means NEXT week's, not today
      return toKey(new Date(now.getTime() + delta * 86_400_000));
    }
  }
  const monthDayMatch = n.match(/\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b/);
  if (monthDayMatch) {
    const month = MONTH_NAMES[monthDayMatch[1]];
    const day = parseInt(monthDayMatch[2], 10);
    if (month !== undefined && day >= 1 && day <= 31) {
      const thisYear = now.getUTCFullYear();
      let candidate = new Date(Date.UTC(thisYear, month, day));
      // Already passed this year (allowing a day of slack for "today") —
      // a visitor naming a date almost always means the next time it comes
      // around, not one that's already gone by.
      if (candidate.getTime() < now.getTime() - 86_400_000) {
        candidate = new Date(Date.UTC(thisYear + 1, month, day));
      }
      return toKey(candidate);
    }
  }
  return null;
}

function resolveSlot(message: string, offered: ResolvedSlot[]): ResolvedSlot | 'ambiguous' | null {
  const n = normalise(message);

  const wantedTime = messageTimeTo24h(message);
  if (wantedTime) {
    // Real, confirmed bug this fixes: matching by time alone let "tomorrow
    // 11 am" silently match TODAY's real, already-offered 11 AM slot just
    // because the time happened to coincide, silently ignoring the
    // visitor's own date correction. When the message names a specific day,
    // an offered slot must actually be ON that day to count as a match —
    // otherwise this correctly returns no match, so the turn falls through
    // to the general LLM path, which re-checks availability for the day the
    // visitor actually meant instead of booking the wrong one.
    const wantedDate = extractDayQualifierDate(message);
    const matches = offered.filter((s) =>
      s.label && labelTimeTo24h(s.label) === wantedTime &&
      (!wantedDate || s.startIso.slice(0, 10) === wantedDate)
    );
    if (matches.length === 1) return matches[0];
  }

  for (const [word, idx] of Object.entries(ORDINAL_WORDS)) {
    if (n.includes(word) && offered[idx]) return offered[idx];
  }

  if (offered.length === 1 && hasBareAffirmativeClause(message)) return offered[0];
  if (offered.length > 1 && hasBareAffirmativeClause(message)) return 'ambiguous';

  return null;
}

/** Shared "given a resolved slot, either ask for what's still missing or
 * complete the booking" logic — extracted from tryBookingConfirmationShortcut
 * so tryAvailabilityRequestShortcut can reuse the exact same deterministic
 * completion path for a one-shot "Thursday 4pm, my name is X, email..."
 * message instead of only fetching the day's slots and dropping the rest of
 * what the visitor already said. One source of truth for "what happens
 * after a slot is resolved," not two copies that could drift apart. */
async function continueBookingFlow(
  slot: ResolvedSlot,
  message: string,
  ctx: {
    tenantId: string; sessionId: string; visitorId?: string; pageUrl?: string;
    bookingRequireTeam?: boolean; bookingRequireService?: boolean;
    bookingContactRequirement?: 'email_only' | 'phone_only' | 'email_or_phone' | 'email_and_phone';
  },
  bookingState: BookingState,
  leadCaptureState: LeadCaptureState,
  hadPendingConfirmation: boolean,
): Promise<BookingShortcutResult> {
  const captured = extractCapturedData(message);
  let firstName = cleanArg(leadCaptureState.firstName);
  let lastName  = cleanArg(leadCaptureState.lastName);
  const email   = cleanArg(captured.email) || cleanArg(leadCaptureState.email);
  const phone   = cleanArg(captured.phone) || cleanArg(leadCaptureState.phone);
  let newlyExtractedName = false;
  if (captured.name && !firstName) {
    const parts = captured.name.split(/\s+/).filter(Boolean);
    firstName = parts[0];
    if (parts.length > 1) lastName = parts.slice(1).join(' ');
    newlyExtractedName = true;
  }
  // extractCapturedData() only recognises a name framed as "my name is
  // X"/"call me X" — too narrow for THIS specific moment, where we just
  // asked "could I get your name and an email or phone number?" and a real
  // visitor's natural reply is a bare "Rahul, rahul@x.com" with no framing
  // phrase at all. Only applied when we're already mid-confirmation
  // (confirmingSlot was set, i.e. this exact question was just asked) so it
  // never over-triggers on an ordinary, unrelated message elsewhere.
  if (!firstName && hadPendingConfirmation) {
    const firstSegment = stripNameFraming(message).split(/[,;]| and /i)[0].trim();
    if (/^[A-Z][a-zA-Z'-]+(\s+[A-Z][a-zA-Z'-]+)?$/i.test(firstSegment) && !NON_NAME_WORDS.has(normalise(firstSegment))) {
      const parts = firstSegment.split(/\s+/).filter(Boolean);
      firstName = parts[0];
      if (parts.length > 1) lastName = parts.slice(1).join(' ');
      newlyExtractedName = true;
    }
  }

  // Interruptible booking: we already asked this exact "name + contact"
  // question on a PRIOR turn (confirmingSlot was already set coming in), and
  // this message contains nothing that looks like an attempted answer — no
  // email, no phone, no name-shaped segment. It's not a reply to the
  // question at all (e.g. "what services do you provide?"), so don't repeat
  // the same canned prompt at it. Bail out WITHOUT touching bookingState —
  // confirmingSlot stays exactly as it is, so the paused booking resumes
  // automatically the next time the visitor actually answers.
  if (hadPendingConfirmation && !captured.email && !captured.phone && !newlyExtractedName) {
    return { handled: false };
  }

  const readiness = assessBookingReadiness({
    slot, firstName, email, phone,
    teamId: bookingState.selectedTeamId || bookingState.selectedStaffId || bookingState.offeredStaffId,
    service: leadCaptureState.service,
    requireTeam: ctx.bookingRequireTeam,
    requireService: ctx.bookingRequireService,
    contactRequirement: ctx.bookingContactRequirement,
  });
  if (!readiness.ready || !firstName) {
    // Persist whatever WAS extracted this turn — even though it's still
    // incomplete — merged over what's already known. Without this, info
    // given across separate turns (name in one message, email in the next)
    // is silently discarded and the same "please give me your name and
    // contact" question repeats forever, since each turn's extraction was
    // only ever compared against an empty LeadCaptureState.
    await setBookingState(ctx.tenantId, ctx.sessionId, { ...bookingState, confirmingSlot: slot });
    await setLeadCaptureState(ctx.tenantId, ctx.sessionId, {
      ...leadCaptureState,
      firstName: firstName || leadCaptureState.firstName,
      lastName: lastName || leadCaptureState.lastName,
      email: email || leadCaptureState.email,
      phone: phone || leadCaptureState.phone,
    });
    const stillNeeded: string[] = [];
    if (readiness.missing.includes('team')) stillNeeded.push('which department you\'d like');
    if (readiness.missing.includes('service')) stillNeeded.push('what this is for');
    if (readiness.missing.includes('name')) stillNeeded.push('your name');
    if (readiness.missing.includes('contact')) stillNeeded.push('an email or phone number');
    return {
      handled: true,
      response: `Great, I can pencil you in for ${slot.label ?? 'that time'}! Could I get ${stillNeeded.join(' and ')} to confirm it?`,
    };
  }

  // Real, confirmed bug this fixes: readiness being satisfied went straight
  // into performBooking() below with no persist step at all — if that
  // attempt then failed (slot taken, no staff available, etc.), the name/
  // email/phone the visitor just gave existed only as local variables for
  // this one turn and evaporated. The next turn re-read empty state from
  // Redis and asked the visitor to repeat themselves. Persisting here,
  // unconditionally, closes that gap the same way the "still incomplete"
  // branch above already does — a failed booking attempt must never lose
  // contact info that was already given.
  await setLeadCaptureState(ctx.tenantId, ctx.sessionId, {
    ...leadCaptureState,
    firstName: firstName || leadCaptureState.firstName,
    lastName: lastName || leadCaptureState.lastName,
    email: email || leadCaptureState.email,
    phone: phone || leadCaptureState.phone,
  });

  const result = await performBooking({
    tenantId: ctx.tenantId, sessionId: ctx.sessionId, visitorId: ctx.visitorId, pageUrl: ctx.pageUrl,
    slot, firstName, lastName, email, phone, topic: leadCaptureState.service, staffId: bookingState.selectedStaffId,
  });

  if (result.ok) {
    // Same "never name an assignment the visitor didn't choose" rule
    // already applied to the general LLM path's prompt — this deterministic
    // shortcut has its own hardcoded confirmation text and was missed when
    // that rule was first added, so it kept naming the staff member even
    // for a tenant where the visitor never picked anyone.
    const mentionStaff = ctx.bookingRequireTeam === true && result.staffName;
    let response = `You're all set! I've booked you for ${slot.label ?? 'the requested time'}${mentionStaff ? ` with ${result.staffName}` : ''}. We'll be in touch shortly.`;
    if (!leadCaptureState.service) {
      response += " One more thing — what's the reason for your visit, so our team can prepare?";
    }
    return { handled: true, response };
  }

  if (result.slotTaken) {
    // Real, confirmed bug this fixes: re-offering alternatives after a
    // last-instant conflict used to drop the date the visitor was actually
    // booking for entirely, silently reverting to today's slots even when
    // the visitor was booking tomorrow (or any other day) — the exact same
    // "date lost" failure class as the original bug, just in this recovery
    // path instead of the initial availability check. Keep the visitor on
    // the day they were already booking.
    const fresh = await backendClient.getWidgetAvailability(ctx.tenantId, {
      staffId: bookingState.selectedStaffId,
      teamId: !bookingState.selectedStaffId ? bookingState.selectedTeamId : undefined,
      date: slot.startIso.slice(0, 10),
    });
    if (fresh.length) {
      const shown = fresh.slice(0, 5);
      await setBookingState(ctx.tenantId, ctx.sessionId, {
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

/** Deterministic department/team-selection resolution — the "bookingStage"
 * fix: a real, confirmed gap where a bare reply like "ss department" or
 * "ss", picking one of the departments list_departments just offered, was
 * previously left entirely to the general LLM path. Per this file's own
 * already-disclosed "second tool call is unreliable" limitation, that path
 * would sometimes just re-ask the same department question instead of
 * progressing — the backend now decides "which department was picked"
 * directly from the real, already-fetched team list, instead of leaving
 * that judgment to the model. Delegates everything downstream (persisting
 * selectedTeamId, fetching staff, single-staff auto-offer) to the existing
 * list_doctors tool logic — one source of truth, not a second copy. */
export async function tryDepartmentSelectionShortcut(
  message: string,
  ctx: { tenantId: string; sessionId: string; visitorId?: string; pageUrl?: string; bookingRequireTeam?: boolean },
  bookingState: BookingState,
): Promise<BookingShortcutResult> {
  // Explicit, not just incidental: with bookingRequireTeam:false the hard
  // tool-gate in base.agent.ts already stops list_departments from ever
  // being called, so departmentsOffered never gets set and this would bail
  // out below anyway — but a team/staff already resolved earlier (e.g. from
  // before the setting changed) still means a genuine department reply
  // should be handled, so only short-circuit when nothing is resolved yet.
  if (ctx.bookingRequireTeam === false && !bookingState.selectedTeamId && !bookingState.selectedStaffId) {
    return { handled: false };
  }
  if (!bookingState.departmentsOffered || bookingState.selectedTeamId || bookingState.meetingCreated) {
    return { handled: false };
  }

  const teams = await backendClient.getWidgetTeams(ctx.tenantId);
  if (!teams.length) return { handled: false };

  const normalized = normalise(message);
  const matches = teams.filter((t) => t.name && normalized.includes(t.name.toLowerCase()));
  // No match, or genuinely ambiguous (e.g. one team's name is a substring
  // of another's) — let the general LLM path interpret it instead of
  // guessing wrong deterministically.
  if (matches.length !== 1) return { handled: false };

  const chosen = matches[0];
  const doctorsResult = await listDoctorsTool.execute({ teamId: chosen.teamId }, {
    tenantId: ctx.tenantId, sessionId: ctx.sessionId, visitorId: ctx.visitorId, pageUrl: ctx.pageUrl,
    companyName: '', timezone: 'UTC',
  });
  if (!doctorsResult.ok) return { handled: false };

  const staff = (doctorsResult.data?.staff as Array<{ staffId: string; name: string }> | undefined) ?? [];
  if (staff.length === 0) {
    return { handled: true, response: `Got it, ${chosen.name}. What day and time would work best for you?` };
  }
  if (staff.length === 1) {
    return { handled: true, response: `Got it, ${chosen.name}. I'll get you booked with ${staff[0].name} — what day and time would work best for you?` };
  }
  return {
    handled: true,
    response: `Got it, ${chosen.name}. We have ${staff.map((s) => s.name).join(', ')} — who would you like to see, or is anyone fine?`,
  };
}

/** Deterministic availability lookup for a visitor naming a new day
 * ("Wednesday", "tomorrow", "today") — same "don't make a common, patterned
 * input depend on a real LLM call" reasoning as the shortcut above, extended
 * to the FIRST step of booking. A real, confirmed production incident: a
 * visitor's "Wednesday 2pm" hit Groq's own 10s timeout ceiling on both the
 * primary and fallback call, producing the generic handoff message — even
 * though the real backend data for that exact day/time was correct (verified
 * directly against the live availability endpoint). Naming a day is exactly
 * as recognisable and low-risk to handle deterministically as a bare "yes"
 * or a department name already is — this closes that gap the same way,
 * reusing check_meeting_availability's own execute() directly (one source of
 * truth for the real slot logic, not a second copy), so it's exactly as
 * correct as the LLM-driven path and immune to a Groq timeout. Only fires
 * for a genuinely NEW date not already offered — an already-offered date's
 * slots are tryBookingConfirmationShortcut's job (picking one), not this
 * one's (fetching them in the first place). */
export async function tryAvailabilityRequestShortcut(
  message: string,
  ctx: {
    tenantId: string; sessionId: string; visitorId?: string; pageUrl?: string;
    bookingRequireTeam?: boolean; bookingRequireService?: boolean;
    bookingContactRequirement?: 'email_only' | 'phone_only' | 'email_or_phone' | 'email_and_phone';
  },
  bookingState: BookingState,
  leadCaptureState: LeadCaptureState,
): Promise<BookingShortcutResult> {
  if (bookingState.meetingCreated) return { handled: false };

  const wantedDate = extractDayQualifierDate(message);
  // Real, confirmed production incident: the literal "Book an appointment"
  // quick-reply button — the single most common message in the entire
  // funnel — is not reliably handled by ANY currently-available model.
  // Live-tested repeatedly: sometimes it correctly calls
  // check_meeting_availability, sometimes it just replies with a generic
  // "what brings you here today?" (zero tool calls), sometimes it asks
  // about department/team even when this tenant has that explicitly turned
  // off. Same "don't leave a common, patterned input to an unreliable LLM
  // judgment call" reasoning as the day-qualifier case below, extended to
  // the case where the visitor hasn't named a day at all yet — only
  // handled when nothing about the booking has started yet (no slots/
  // department already offered), so this never overrides a turn that's
  // actually further along in the conversation.
  // Gated on bookingRequireTeam !== true: a tenant that DOES require picking
  // a department/doctor first must not skip straight to time slots — that
  // question still belongs to the (unaffected, already-working)
  // tryDepartmentSelectionShortcut / general LLM path.
  const isBareBookingIntent = !wantedDate && ctx.bookingRequireTeam !== true
    && BARE_BOOKING_INTENT_PHRASES.some((p) => normalise(message).includes(p))
    && !bookingState.offeredSlots?.length && !bookingState.departmentsOffered
    && !bookingState.selectedTeamId && !bookingState.selectedStaffId;
  if (!wantedDate && !isBareBookingIntent) return { handled: false };

  if (wantedDate) {
    const alreadyOfferedForDate = (bookingState.offeredSlots ?? []).some((s) => s.startIso.slice(0, 10) === wantedDate);
    if (alreadyOfferedForDate) return { handled: false };
  }

  // Extracted BEFORE the tool call now (not after) so it can be passed
  // through as preferredTime — real, confirmed gap this closes: a visitor
  // asking for "3pm" on a busy-morning day used to just get the day's first
  // 5 (unrelated, morning) slots back with zero acknowledgment of the 3pm
  // they actually asked for. The tool now ranks its shown slots by
  // closeness to this time instead of blind chronological order.
  const wantedTime = messageTimeTo24h(message);
  const result = await checkMeetingAvailabilityTool.execute(
    { preferredDate: wantedDate ?? undefined, preferredTime: wantedTime ?? undefined, staffId: bookingState.selectedStaffId },
    { tenantId: ctx.tenantId, sessionId: ctx.sessionId, visitorId: ctx.visitorId, pageUrl: ctx.pageUrl, companyName: '', timezone: 'UTC' },
  );
  if (!result.ok) return { handled: false };

  const freshSlots = (result.data?.slots as ResolvedSlot[] | undefined) ?? [];
  // Real gap this closes: a one-shot message naming BOTH a day and a
  // specific time ("Thursday 4pm, my name is X, email...") used to only
  // get the day's slot list back, silently dropping the time/name/contact
  // the visitor already gave — they'd have to repeat themselves. If the
  // requested time is among the (now time-ranked) slots just fetched,
  // resolve it immediately and continue the SAME shared flow
  // tryBookingConfirmationShortcut uses, instead of stopping at "here's the
  // day's availability."
  const timeMatch = wantedTime
    ? freshSlots.find((s) => s.label && labelTimeTo24h(s.label) === wantedTime)
    : undefined;
  if (!timeMatch) return { handled: true, response: result.summary };

  const freshBookingState: BookingState = {
    ...bookingState,
    offeredSlots: freshSlots, offeredAt: Date.now(), confirmingSlot: undefined,
  };
  return continueBookingFlow(timeMatch, message, ctx, freshBookingState, leadCaptureState, false);
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
  ctx: {
    tenantId: string; sessionId: string; visitorId?: string; pageUrl?: string;
    bookingRequireTeam?: boolean; bookingRequireService?: boolean;
    bookingContactRequirement?: 'email_only' | 'phone_only' | 'email_or_phone' | 'email_and_phone';
  },
  bookingState: BookingState,
  leadCaptureState: LeadCaptureState,
): Promise<BookingShortcutResult> {
  if (!bookingState.offeredSlots?.length || bookingState.meetingCreated) {
    return { handled: false };
  }

  // Captured BEFORE slot resolution below (which may itself set
  // confirmingSlot for the first time this turn) — this is what
  // distinguishes "we already asked this exact question on a prior turn" from
  // "we're asking it for the first time right now", see the interruptibility
  // check further down.
  const hadPendingConfirmation = !!bookingState.confirmingSlot;

  let slot = bookingState.confirmingSlot ?? null;
  if (!slot) {
    const resolved = resolveSlot(message, bookingState.offeredSlots);
    if (resolved === 'ambiguous') {
      // Real gap found in live testing: unlike the confirmingSlot branch
      // below, this used to return without ever touching bookingState —
      // no record that a time-clarification question was left pending, an
      // asymmetry with how every other branch here persists state.
      // Persisting it doesn't change behavior on its own turn, but keeps
      // BookingState honest for anything else that inspects it (the
      // Conversation Inspector, future logic) rather than silently omitting
      // this branch.
      await setBookingState(ctx.tenantId, ctx.sessionId, { ...bookingState, disambiguating: true });
      return { handled: true, response: 'Which time works best for you — could you tell me the specific time?' };
    }
    if (!resolved) return { handled: false };
    slot = resolved;
    // Clears the pending-disambiguation flag now that a specific slot
    // actually resolved — the readiness-gate/booking logic below persists
    // confirmingSlot itself once it knows whether contact info is complete,
    // so this only needs to clear the now-stale disambiguating flag.
    if (bookingState.disambiguating) {
      bookingState = { ...bookingState, disambiguating: false };
    }
  }

  return continueBookingFlow(slot, message, ctx, bookingState, leadCaptureState, hadPendingConfirmation);
}
