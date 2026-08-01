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
  firstName: z.string().optional().describe('The visitor\'s first name, if they just gave it'),
  lastName: z.string().optional(),
  email: z.string().optional().describe('The visitor\'s email, if they just gave it'),
  phone: z.string().optional().describe('The visitor\'s phone number, if they just gave it'),
  topic: z.string().optional().describe('What they want to discuss, if mentioned'),
});

export const bookMeetingTool: AgentTool<z.infer<typeof schema>> = {
  name: 'book_meeting',
  description:
    'Confirm and book a real meeting at one of the times offered by check_meeting_availability. ' +
    'Requires the visitor\'s name and (email or phone) — ask for whichever is still missing before calling this.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const booking = (await getBookingState(ctx.sessionId)) ?? {};

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
    const lead = (await getLeadCaptureState(ctx.sessionId)) ?? {};
    const firstName = cleanArg(args.firstName) || lead.firstName;
    const lastName  = cleanArg(args.lastName)  || lead.lastName;
    const email     = cleanArg(args.email)     || lead.email;
    const phone     = cleanArg(args.phone)     || lead.phone;

    if (!firstName || (!email && !phone)) {
      return { ok: false, summary: 'Still need the visitor\'s name and an email or phone number before this can be booked — ask for whichever is missing.' };
    }

    const result = await backendClient.bookWidgetMeeting({
      tenantId: ctx.tenantId, sessionId: ctx.sessionId, visitorId: ctx.visitorId, sourceUrl: ctx.pageUrl,
      startIso: slot.startIso, endIso: slot.endIso, firstName, lastName, email, phone, topic: args.topic,
    });

    if (!result.success) {
      return { ok: false, summary: `Could not book that time: ${result.error || 'please try another slot.'}` };
    }

    await setBookingState(ctx.sessionId, { ...booking, meetingCreated: true, meetingId: result.meetingId });
    // Keep LeadCaptureState in sync so maybeCaptureWidgetLead() (which runs
    // unconditionally after this) sees leadCreated:true and never attempts a
    // second, competing Lead for the same session.
    await setLeadCaptureState(ctx.sessionId, { ...lead, firstName, lastName, email, phone, leadCreated: true, leadId: result.leadId });

    return {
      ok: true,
      summary: result.alreadyCreated
        ? 'This meeting was already booked.'
        : `Booked${result.staffName ? ` with ${result.staffName}` : ''} — tell the visitor it's confirmed.`,
    };
  },
};
