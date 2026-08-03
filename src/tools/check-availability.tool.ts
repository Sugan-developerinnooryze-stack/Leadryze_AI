import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import { getBookingState, setBookingState } from '../memory/conversation.memory';

const schema = z.object({
  preferredDate: z.string().optional().describe('A specific date the visitor asked about, as YYYY-MM-DD, if any'),
  timeOfDay: z.enum(['morning', 'afternoon', 'any']).optional().describe('Narrows to morning or afternoon slots if the visitor asked for one'),
  days: z.number().int().min(1).max(30).optional().describe('How many days out to look — defaults to the tenant\'s own booking horizon'),
  staffId: z.string().optional().describe('The exact staffId of the doctor the visitor chose from list_doctors, if this business has departments — omit otherwise'),
});

export const checkMeetingAvailabilityTool: AgentTool<z.infer<typeof schema>> = {
  name: 'check_meeting_availability',
  description:
    'Check real available meeting/call times for this business. Call this whenever a visitor wants to schedule ' +
    'a call, meeting, or demo, or asks what times are open. Never invent or guess times yourself — always call this first.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const booking = (await getBookingState(ctx.sessionId)) ?? {};
    const staffId = cleanArg(args.staffId) || booking.selectedStaffId;

    const slots = await backendClient.getWidgetAvailability(ctx.tenantId, { days: args.days, timeOfDay: args.timeOfDay, staffId });
    if (!slots.length) {
      return { ok: true, summary: 'No available meeting times were found — booking may not be enabled for this business, or nothing is free in the horizon checked.' };
    }

    const filtered = args.preferredDate
      ? slots.filter((s) => s.startIso.startsWith(args.preferredDate!))
      : slots;
    const shown = (filtered.length ? filtered : slots).slice(0, 5);

    await setBookingState(ctx.sessionId, {
      ...booking,
      selectedStaffId: staffId || booking.selectedStaffId,
      offeredSlots: shown.map((s) => ({ startIso: s.startIso, endIso: s.endIso, label: s.label })),
      offeredAt: Date.now(),
      confirmingSlot: undefined,
    });

    return {
      ok: true,
      summary: `Available times: ${shown.map((s) => s.label).join('; ')}.`,
      data: { slots: shown },
    };
  },
};
