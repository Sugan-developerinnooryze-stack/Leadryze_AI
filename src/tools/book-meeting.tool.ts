import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import {
  getBookingState, setBookingState,
  getLeadCaptureState, setLeadCaptureState,
} from '../memory/conversation.memory';

const schema = z.object({
  startIso: z.string().describe('The exact startIso of one of the times just offered by check_meeting_availability — never a time you invented yourself'),
  // .nullish() on every optional string field — see list-departments.tool.ts's
  // serviceHint comment for why .optional() alone isn't enough (Groq
  // sometimes emits explicit null instead of omitting an unset field).
  firstName: z.string().nullish().describe('The visitor\'s first name, if they just gave it'),
  lastName: z.string().nullish(),
  email: z.string().nullish().describe('The visitor\'s email, if they just gave it'),
  phone: z.string().nullish().describe('The visitor\'s phone number, if they just gave it'),
  topic: z.string().nullish().describe('What they want to discuss, if mentioned'),
  staffId: z.string().nullish().describe('The exact staffId of the team member the visitor chose from list_doctors, if this business has departments — omit otherwise'),
});

export interface PerformBookingParams {
  tenantId: string; sessionId: string; visitorId?: string; pageUrl?: string;
  slot: { startIso: string; endIso: string; label?: string };
  firstName: string; lastName?: string; email?: string; phone?: string; topic?: string; staffId?: string;
  // Booking-parity fields — carried over from LeadCaptureState so a visitor
  // who reaches high buying intent and books directly gets the same rich
  // Lead the plain "Request Quote" capture path produces.
  leadScore?: number; buyingIntent?: 'low' | 'medium' | 'high';
  interestedItems?: Array<{ datasetId: string; recordId: string; title: string; datasetVersion: number }>;
  requirement?: string; conversationSummary?: string;
}

export interface PerformBookingResult {
  ok: boolean;
  summary: string;
  meetingId?: string;
  staffName?: string;
  alreadyCreated?: boolean;
  slotTaken?: boolean;
}

export interface BookingReadinessInput {
  slot: { startIso: string } | null | undefined;
  firstName?: string;
  email?: string;
  phone?: string;
  /** Only checked when requireTeam is true — a real selected/offered team
   * (not just "departments exist"), same shape as BookingState.selectedTeamId. */
  teamId?: string;
  /** Only checked when requireService is true — LeadCaptureState.service /
   * book_meeting's own topic arg. */
  service?: string;
  requireTeam?: boolean;
  requireService?: boolean;
  /** Defaults to 'email_or_phone' — today's exact prior behavior — when
   * omitted, so every existing call site (and any tenant that's never
   * configured this) keeps working unchanged. */
  contactRequirement?: 'email_only' | 'phone_only' | 'email_or_phone' | 'email_and_phone';
}

export interface BookingReadinessResult {
  ready: boolean;
  missing: Array<'slot' | 'name' | 'contact' | 'team' | 'service'>;
}

/** Shared, pure readiness gate — the exact "is there enough to actually book
 * this?" check, previously duplicated almost verbatim in this tool's own
 * execute() below and in booking-confirmation-shortcut.ts. A real slot plus
 * a first name plus contact info is the deterministic bar for calling
 * performBooking() — nothing here is LLM-dependent, so both call sites now
 * share one source of truth instead of two copies that could silently
 * drift apart. team/service checks are opt-in per tenant (Tenant.widget.
 * booking.requireTeam/requireService) — omitted entirely when not
 * required, so a tenant that's never configured them sees no behavior
 * change. */
/** Real, confirmed exploit this closes: book_meeting's firstName/email/phone
 * args come straight from the LLM's own tool call, and nothing stopped the
 * model from inventing a plausible-looking value (e.g. reusing an earlier
 * department name, "ss", as a "firstName", plus a fabricated phone number)
 * just to force its own required-field checklist to pass — creating a real
 * Lead+Meeting with fake contact info the visitor never gave. A value is
 * only trusted if the visitor's own raw text for THIS turn actually
 * contains it (case-insensitive) — matching what extractCapturedData
 * already assumes elsewhere: real captured data is always traceable back to
 * something the visitor literally typed. Already-known values (carried in
 * from a PRIOR turn's LeadCaptureState) are exempt — they were corroborated
 * when they were first captured, and a later turn omitting the raw text
 * (e.g. "today at 2pm" after "My name is Bala" two turns ago) must not
 * un-know a real name/email/phone. */
function isFreshValueCorroborated(value: string, rawMessage: string): boolean {
  if (!rawMessage) return false;
  const needle = value.trim().split(/\s+/)[0].toLowerCase();
  return needle.length > 1 && rawMessage.toLowerCase().includes(needle);
}

export function assessBookingReadiness(input: BookingReadinessInput): BookingReadinessResult {
  const missing: BookingReadinessResult['missing'] = [];
  if (!input.slot) missing.push('slot');
  if (!input.firstName) missing.push('name');
  if (input.requireTeam && !input.teamId) missing.push('team');
  if (input.requireService && !input.service) missing.push('service');

  const contactRequirement = input.contactRequirement ?? 'email_or_phone';
  const hasContact =
    contactRequirement === 'email_only' ? !!input.email :
    contactRequirement === 'phone_only' ? !!input.phone :
    contactRequirement === 'email_and_phone' ? !!input.email && !!input.phone :
    !!input.email || !!input.phone; // 'email_or_phone'
  if (!hasContact) missing.push('contact');

  return { ready: missing.length === 0, missing };
}

/** The real booking logic, shared by the LLM-facing tool below AND the
 * deterministic booking-confirmation shortcut (booking-confirmation-
 * shortcut.ts) — extracted so both call the exact same code rather than
 * duplicating it. Callers are expected to have already resolved `slot` from
 * BookingState.offeredSlots and merged contact info with LeadCaptureState;
 * this function only does the actual booking + state persistence. */
export async function performBooking(params: PerformBookingParams): Promise<PerformBookingResult> {
  const { tenantId, sessionId, visitorId, pageUrl, slot, firstName, lastName, email, phone, topic, staffId,
    leadScore, buyingIntent, interestedItems, requirement, conversationSummary } = params;

  const result = await backendClient.bookWidgetMeeting({
    tenantId, sessionId, visitorId, sourceUrl: pageUrl,
    startIso: slot.startIso, endIso: slot.endIso, firstName, lastName, email, phone, topic, staffId,
    leadScore, buyingIntent, interestedItems, requirement, conversationSummary,
  });

  if (!result.success) {
    return {
      ok: false,
      summary: `Could not book that time: ${result.error || 'please try another slot.'}`,
      slotTaken: result.reason === 'slot_taken',
    };
  }

  const booking = (await getBookingState(tenantId, sessionId)) ?? {};
  await setBookingState(tenantId, sessionId, { ...booking, meetingCreated: true, meetingId: result.meetingId, confirmingSlot: undefined });
  // Keep LeadCaptureState in sync so maybeCaptureWidgetLead() (which runs
  // unconditionally after this) sees leadCreated:true and never attempts a
  // second, competing Lead for the same session. lastSent* fields mirror
  // what was just sent above, so a later chat turn's enrichment path
  // correctly detects "no meaningful new signal yet" rather than
  // re-sending the same values immediately.
  const lead = (await getLeadCaptureState(tenantId, sessionId)) ?? {};
  await setLeadCaptureState(tenantId, sessionId, {
    ...lead, firstName, lastName, email, phone, service: lead.service || topic || undefined,
    leadCreated: true, leadId: result.leadId,
    lastSentBuyingIntent: buyingIntent ?? lead.buyingIntent,
    lastSentRequirement: requirement ?? lead.requirement,
    lastSentItemCount: interestedItems?.length ?? lead.interestedItems?.length,
  });

  return {
    ok: true,
    summary: result.alreadyCreated
      ? 'This meeting was already booked.'
      : `Booked${result.staffName ? ` with ${result.staffName}` : ''} — tell the visitor it's confirmed.`,
    meetingId: result.meetingId,
    staffName: result.staffName,
    alreadyCreated: result.alreadyCreated,
  };
}

export const bookMeetingTool: AgentTool<z.infer<typeof schema>> = {
  name: 'book_meeting',
  description:
    'Confirm and book a real meeting — ONLY call this after the visitor has already picked one of the exact times check_meeting_availability just returned. ' +
    'Never call this, and never ask the visitor for their name/email/phone, before a specific time has been chosen. ' +
    'Once a time is chosen: if name or (email or phone) is still missing, ask for it first, then call this.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const booking = (await getBookingState(ctx.tenantId, ctx.sessionId)) ?? {};

    if (booking.meetingCreated) {
      return { ok: true, summary: 'This meeting is already booked — no further action needed.' };
    }

    const offered = booking.offeredSlots ?? [];
    const slot = offered.find((s) => s.startIso === args.startIso);
    if (!slot) {
      return { ok: false, summary: 'That time was not one of the times just offered — call check_meeting_availability again and pick one of those exact times.' };
    }

    // Merge with whatever's already known for this session — never let a
    // turn that omits a field erase one captured earlier (same rule
    // maybeCaptureWidgetLead already uses for LeadCaptureState).
    const lead = (await getLeadCaptureState(ctx.tenantId, ctx.sessionId)) ?? {};
    const rawMessage = ctx.rawMessage ?? '';
    // A value freshly supplied by the model this turn (not already known
    // from a prior turn) must be corroborated by the visitor's own raw text
    // — otherwise it's discarded rather than trusted, so a hallucinated
    // name/contact blocks the booking (falls into "still need...") instead
    // of silently creating a fake record. See isFreshValueCorroborated.
    const rawFirstName = cleanArg(args.firstName);
    const rawEmail     = cleanArg(args.email);
    const rawPhone     = cleanArg(args.phone);
    const firstName = lead.firstName || (rawFirstName && isFreshValueCorroborated(rawFirstName, rawMessage) ? rawFirstName : undefined);
    const lastName  = cleanArg(args.lastName)  || lead.lastName;
    const email     = lead.email || (rawEmail && isFreshValueCorroborated(rawEmail, rawMessage) ? rawEmail : undefined);
    const phone     = lead.phone || (rawPhone && isFreshValueCorroborated(rawPhone, rawMessage) ? rawPhone : undefined);
    // topic merges into the same "service" signal maybeCaptureWidgetLead
    // already tracks — a booking's stated reason shouldn't be a dead end
    // that only lands in the Meeting's notes.
    const topic     = cleanArg(args.topic)     || lead.service;

    const readiness = assessBookingReadiness({
      slot, firstName, email, phone,
      teamId: booking.selectedTeamId || cleanArg(args.staffId) || booking.selectedStaffId || undefined, // a resolved staff member implies a team was already resolved too
      service: topic,
      requireTeam: ctx.bookingRequireTeam,
      requireService: ctx.bookingRequireService,
      contactRequirement: ctx.bookingContactRequirement,
    });
    if (!readiness.ready || !firstName) {
      // Field-specific, not a blanket "name and email/phone" — a real,
      // confirmed cause of the model re-asking for info it already has
      // (e.g. asking for the name again after "Bala" was already given,
      // because the missing PIECE was actually just the email/phone). Also
      // explicitly echoes what's already known so the model has no excuse
      // to re-ask for it.
      const missing: string[] = [];
      if (readiness.missing.includes('team')) missing.push('which department/team');
      if (readiness.missing.includes('service')) missing.push('what service this is for');
      if (readiness.missing.includes('name')) missing.push("the visitor's name");
      if (readiness.missing.includes('contact')) missing.push('an email or phone number');
      return {
        ok: false,
        summary:
          `Still need ${missing.join(' and ')} before this can be booked. ` +
          `Already known: name=${firstName ?? 'not yet given'}. ` +
          `Ask ONLY for what's missing — do not ask again for anything already known.`,
      };
    }

    const staffId = cleanArg(args.staffId) || booking.selectedStaffId;

    const result = await performBooking({
      tenantId: ctx.tenantId, sessionId: ctx.sessionId, visitorId: ctx.visitorId, pageUrl: ctx.pageUrl,
      slot, firstName, lastName, email, phone, topic, staffId,
      leadScore: lead.leadScore, buyingIntent: lead.buyingIntent,
      interestedItems: lead.interestedItems, requirement: lead.requirement, conversationSummary: lead.conversationSummary,
    });

    return { ok: result.ok, summary: result.summary };
  },
};
