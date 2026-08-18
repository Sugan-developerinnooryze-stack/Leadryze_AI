import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import { getBookingState, setBookingState } from '../memory/conversation.memory';
import { labelTimeTo24h, timeToMinutes } from '../utils/time-parse';

const schema = z.object({
  // .nullish() on the string fields — see list-departments.tool.ts's
  // serviceHint comment for why .optional() alone isn't enough.
  preferredDate: z.string().nullish().describe('A specific date the visitor asked about, as YYYY-MM-DD, if any'),
  // Real, confirmed gap this closes: a visitor asking for "3pm" on a day
  // that opens at 9am used to get shown the day's first 5 morning slots —
  // nowhere near 3pm — with no acknowledgment of the actual time asked for.
  // When given, the shown slots are ranked by closeness to this time
  // instead of blindly chronological.
  preferredTime: z.string().nullish().describe('The specific time of day the visitor asked about, as 24h "HH:MM", if any (e.g. "15:00" for "3pm") — omit if they only named a date or no time at all'),
  timeOfDay: z.enum(['morning', 'afternoon', 'any']).optional().describe('Narrows to morning or afternoon slots if the visitor asked for one'),
  days: z.number().int().min(1).max(30).optional().describe('How many days out to look — defaults to the tenant\'s own booking horizon'),
  staffId: z.string().nullish().describe('The exact staffId of the team member the visitor chose from list_doctors, if this business has departments — omit otherwise'),
});

export const checkMeetingAvailabilityTool: AgentTool<z.infer<typeof schema>> = {
  name: 'check_meeting_availability',
  description:
    'Check real available meeting/call times for this business. Call this whenever a visitor wants to schedule ' +
    'a call, meeting, or demo, or asks what times are open. Never invent or guess times yourself — always call this first.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const booking = (await getBookingState(ctx.tenantId, ctx.sessionId)) ?? {};
    const staffId = cleanArg(args.staffId) || booking.selectedStaffId;
    // No specific doctor chosen yet, but a department/team is already known
    // (from list_doctors) — union that whole team's availability instead of
    // the tenant-wide/no-staff path, so a visitor is only ever offered times
    // SOMEONE on the team can actually take, not just whoever the tenant-wide
    // check happens to see.
    const teamId = !staffId ? booking.selectedTeamId : undefined;
    const preferredDate = cleanArg(args.preferredDate);

    // Real, confirmed bug this fixes: a slot already confirmingSlot (the
    // visitor picked a time and is now just being asked for name/contact
    // info) could be silently discarded by a redundant/confused re-call of
    // this same tool on a later turn — e.g. the general LLM path re-invoking
    // it while confused about missing contact info. The old "stillOffered"
    // check below only preserved confirmingSlot if the exact same slot
    // happened to reappear in a freshly refetched, UNSCOPED batch — which it
    // almost never does, since an unscoped fetch defaults to today, not
    // whatever day was actually confirmed. When there's no explicit
    // date/time in this call (i.e. nothing suggesting the visitor actually
    // wants to look at something different), don't touch availability or
    // state at all — just remind the model what's still pending.
    if (booking.confirmingSlot && !preferredDate && !cleanArg(args.preferredTime)) {
      return {
        ok: true,
        summary: `A time is already pending confirmation: ${booking.confirmingSlot.label ?? booking.confirmingSlot.startIso}. Don't re-offer other times — just ask for whatever's still missing (name/email/phone) to confirm THIS slot, unless the visitor explicitly asked for a different day.`,
      };
    }

    // Real, confirmed bug this fixes: preferredDate used to only be applied
    // as a client-side .filter() AFTER already fetching an unscoped batch of
    // slots — if that date had zero matches (very likely, see
    // availability.service.ts's own fix comment), the code silently fell
    // back to showing whatever it got, presenting a DIFFERENT day's slots as
    // if they satisfied the request. Now the date is sent to the backend so
    // it's actually scanned for; if genuinely nothing is free that day, say
    // so explicitly and offer real next-available times instead of
    // silently substituting an unrequested day.
    let slots = await backendClient.getWidgetAvailability(ctx.tenantId, { days: args.days, timeOfDay: args.timeOfDay, staffId, teamId, date: preferredDate });
    let notePrefix = '';
    if (preferredDate && !slots.length) {
      slots = await backendClient.getWidgetAvailability(ctx.tenantId, { days: args.days, timeOfDay: args.timeOfDay, staffId, teamId });
      notePrefix = `Nothing is available on ${preferredDate} — here are the next real available times instead: `;
    }
    if (!slots.length) {
      return { ok: true, summary: 'No available meeting times were found — booking may not be enabled for this business, or nothing is free in the horizon checked.' };
    }

    const preferredTime = cleanArg(args.preferredTime);
    let exactTimeAvailable = false;
    let shown = slots.slice(0, 5);
    if (preferredTime) {
      const wantedMinutes = timeToMinutes(preferredTime);
      // Rank the SAME day's slots by closeness to the requested time instead
      // of blindly taking the day's earliest ones — a visitor who asked for
      // 3pm should see times near 3pm, not five unrelated morning slots with
      // no acknowledgment that 3pm was ever mentioned. Scoped to the first
      // real date present in the results (today's/the requested day's own
      // slots) so this doesn't pull a same-time slot from a totally
      // different, later day into the list.
      const targetDate = slots[0].startIso.slice(0, 10);
      const sameDay = slots.filter((s) => s.startIso.slice(0, 10) === targetDate);
      const ranked = [...sameDay].sort((a, b) => {
        const da = a.label ? Math.abs(timeToMinutes(labelTimeTo24h(a.label) ?? '00:00') - wantedMinutes) : 9999;
        const db = b.label ? Math.abs(timeToMinutes(labelTimeTo24h(b.label) ?? '00:00') - wantedMinutes) : 9999;
        return da - db;
      });
      if (ranked.length) {
        shown = ranked.slice(0, 5).sort((a, b) => a.startIso.localeCompare(b.startIso));
        exactTimeAvailable = ranked.some((s) => s.label && labelTimeTo24h(s.label) === preferredTime);
      }
    }
    if (preferredTime && !exactTimeAvailable) {
      const [h, m] = preferredTime.split(':').map(Number);
      const friendly = `${((h % 12) || 12)}:${String(m).padStart(2, '0')} ${h >= 12 ? 'PM' : 'AM'}`;
      notePrefix = `${notePrefix}${friendly} isn't available — here's what's closest: `;
    }

    // Only clear an in-progress confirmation if this genuinely looks like a
    // different query — a real, confirmed bug: this tool was previously
    // called unconditionally on EVERY invocation, so a redundant/confused
    // re-call mid-confirmation (a real, observed model-reliability gap)
    // silently wiped out a booking that was already being confirmed. If the
    // slot the visitor was already confirming is still among the newly
    // returned slots, keep it — this is very likely a redundant re-call, not
    // a real change of query.
    const stillOffered = booking.confirmingSlot
      ? shown.some((s) => s.startIso === booking.confirmingSlot!.startIso)
      : false;

    await setBookingState(ctx.tenantId, ctx.sessionId, {
      ...booking,
      selectedStaffId: staffId || booking.selectedStaffId,
      offeredSlots: shown.map((s) => ({ startIso: s.startIso, endIso: s.endIso, label: s.label })),
      offeredAt: Date.now(),
      confirmingSlot: stillOffered ? booking.confirmingSlot : undefined,
    });

    return {
      ok: true,
      summary: `${notePrefix}Available times: ${shown.map((s) => s.label).join('; ')}.`,
      data: { slots: shown },
    };
  },
};
