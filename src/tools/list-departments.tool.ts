import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import { getBookingState, setBookingState } from '../memory/conversation.memory';

const schema = z.object({
  // .nullish() (not just .optional()) — confirmed live to matter: Groq's
  // function-calling sometimes fills an unset optional string arg with an
  // explicit `null` rather than omitting it, which .optional() alone
  // rejects outright (Zod's `optional` only allows `undefined`), causing
  // this whole tool call to fail Zod validation before execute() ever runs
  // — the visitor's real reply then falls through to unrelated fallback
  // logic instead of the department list ever being shown.
  serviceHint: z.string().nullish().describe(
    'The service/specialty the visitor already mentioned wanting (e.g. "cardiology"), if any — pre-filters the department list to the matching one instead of showing every department.'
  ),
});

export const listDepartmentsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'list_departments',
  description:
    'List the departments/specialties a visitor can choose between before booking a meeting. ' +
    'Call this FIRST when a visitor wants to book, before check_meeting_availability. Pass serviceHint if the visitor already said what they need. ' +
    'If it returns no departments, this business has none configured — proceed straight to checking availability.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const teams = await backendClient.getWidgetTeams(ctx.tenantId, cleanArg(args.serviceHint));
    if (!teams.length) {
      return { ok: true, summary: 'No departments are configured for this business — proceed straight to checking availability, no department/team-member question needed.' };
    }

    // Real, confirmed gap fixed here: without this, the visitor's very next
    // reply picking a department (e.g. "ss department") had no durable
    // signal that a booking flow was already in progress — toolHint fell
    // back to unbounded, and the model could wander into
    // search_website_knowledge for a plain department name instead of
    // continuing with list_doctors.
    const booking = (await getBookingState(ctx.tenantId, ctx.sessionId)) ?? {};
    await setBookingState(ctx.tenantId, ctx.sessionId, { ...booking, departmentsOffered: true });

    return {
      ok: true,
      summary: `Departments: ${teams.map((t) => t.name).join(', ')}. Ask the visitor which one, then call list_doctors.`,
      data: { teams },
    };
  },
};
