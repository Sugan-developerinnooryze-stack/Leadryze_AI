import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { getBookingState, setBookingState } from '../memory/conversation.memory';
import { cleanArg } from '../utils/clean-arg';

const schema = z.object({
  // Deliberately optional — Groq's own function-calling layer rejects a call
  // missing a required property BEFORE it ever reaches execute() below,
  // which was confirmed live to trigger a full fallback-model retry +
  // timeout cascade (20-35+ seconds) every time the model called this with
  // no args at all. Making it optional lets that call through; execute()
  // below falls back to session state, then to the unambiguous single-team
  // case, rather than crashing the turn.
  // .nullish() — see list-departments.tool.ts's serviceHint comment for why
  // .optional() alone isn't enough (Groq sometimes emits explicit null).
  teamId: z.string().nullish().describe('The exact teamId of the department the visitor chose, from list_departments — omit only if you truly don\'t have one yet'),
});

// Tool name kept as 'list_doctors' — referenced by name in several other
// files (registry.ts's TOOL_HINT_NAMES, response-confidence.ts's booking
// signal check) — renaming is a bigger, separate change. Only the
// description text below is what the LLM actually reads, so that's what's
// made generic here; LeadRyze serves salons/law firms/real estate/sales
// teams, not just medical practices.
export const listDoctorsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'list_doctors',
  description:
    'List the active staff/team members within one department the visitor already chose from list_departments. ' +
    'Call this after the visitor picks a department, before checking availability.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const teams = await backendClient.getWidgetTeams(ctx.tenantId);

    let teamId = cleanArg(args.teamId);
    let team = teamId ? teams.find((t) => t.teamId === teamId) : undefined;

    if (!team) {
      // No valid teamId given — try what's already known for this session,
      // then fall back to the unambiguous case (exactly one department).
      const existing = (await getBookingState(ctx.tenantId, ctx.sessionId)) ?? {};
      const knownTeam = existing.selectedTeamId ? teams.find((t) => t.teamId === existing.selectedTeamId) : undefined;
      if (knownTeam) {
        teamId = knownTeam.teamId;
        team = knownTeam;
      } else if (teams.length === 1) {
        teamId = teams[0].teamId;
        team = teams[0];
      } else if (teams.length === 0) {
        return { ok: true, summary: 'No departments are configured for this business — proceed straight to checking availability, no department/team-member question needed.' };
      } else {
        return {
          ok: false,
          summary: `A department must be chosen first — call list_departments and ask the visitor which of these: ${teams.map((t) => t.name).join(', ')}.`,
        };
      }
    }

    const staff = await backendClient.getWidgetStaff(ctx.tenantId, teamId!);

    const booking = (await getBookingState(ctx.tenantId, ctx.sessionId)) ?? {};
    await setBookingState(ctx.tenantId, ctx.sessionId, { ...booking, selectedTeamId: teamId, selectedTeamName: team?.name });

    if (!staff.length) {
      return { ok: true, summary: 'No active staff found in that department — proceed straight to checking availability for the business as a whole.' };
    }

    // Real, confirmed gap fixed here: with exactly one doctor, the model's
    // own reply naturally suggests them ("would you like to see Dr. X?"),
    // but nothing durable recorded WHICH doctor that was — only the team
    // got persisted. A visitor's bare "yes" on a later turn then had no
    // state to resolve against. Same "single-item-so-treat-as-implicitly-
    // offered" pattern already used for a single offered time slot.
    if (staff.length === 1) {
      const booking = (await getBookingState(ctx.tenantId, ctx.sessionId)) ?? {};
      await setBookingState(ctx.tenantId, ctx.sessionId, {
        ...booking, selectedTeamId: teamId, selectedTeamName: team?.name,
        offeredStaffId: staff[0].staffId, offeredStaffName: staff[0].name,
      });
    }

    return {
      ok: true,
      summary: `Staff in ${team?.name ?? 'that department'}: ${staff.map((s) => s.name).join(', ')}. Ask the visitor who they'd like, then call check_meeting_availability with that staffId.`,
      data: { staff },
    };
  },
};
