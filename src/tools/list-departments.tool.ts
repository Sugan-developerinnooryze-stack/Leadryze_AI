import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';

const schema = z.object({});

export const listDepartmentsTool: AgentTool<z.infer<typeof schema>> = {
  name: 'list_departments',
  description:
    'List the departments/specialties a visitor can choose between before booking a meeting. ' +
    'Call this FIRST when a visitor wants to book, before check_meeting_availability. ' +
    'If it returns no departments, this business has none configured — proceed straight to checking availability.',
  schema,
  surfaces: ['public_widget'],
  async execute(_args, ctx) {
    const teams = await backendClient.getWidgetTeams(ctx.tenantId);
    if (!teams.length) {
      return { ok: true, summary: 'No departments are configured for this business — proceed straight to checking availability, no department/doctor question needed.' };
    }
    return {
      ok: true,
      summary: `Departments: ${teams.map((t) => t.name).join(', ')}. Ask the visitor which one, then call list_doctors.`,
      data: { teams },
    };
  },
};
