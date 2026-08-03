import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { getBookingState, setBookingState } from '../memory/conversation.memory';

const schema = z.object({
  teamId: z.string().describe('The exact teamId of the department the visitor chose, from list_departments — never invent one'),
});

export const listDoctorsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'list_doctors',
  description:
    'List the active staff/doctors within one department the visitor already chose from list_departments. ' +
    'Call this after the visitor picks a department, before checking availability.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const teams = await backendClient.getWidgetTeams(ctx.tenantId);
    const team = teams.find((t) => t.teamId === args.teamId);

    const staff = await backendClient.getWidgetStaff(ctx.tenantId, args.teamId);

    const booking = (await getBookingState(ctx.sessionId)) ?? {};
    await setBookingState(ctx.sessionId, { ...booking, selectedTeamId: args.teamId, selectedTeamName: team?.name });

    if (!staff.length) {
      return { ok: true, summary: 'No active staff found in that department — proceed straight to checking availability for the business as a whole.' };
    }
    return {
      ok: true,
      summary: `Staff in ${team?.name ?? 'that department'}: ${staff.map((s) => s.name).join(', ')}. Ask the visitor who they'd like, then call check_meeting_availability with that staffId.`,
      data: { staff },
    };
  },
};
