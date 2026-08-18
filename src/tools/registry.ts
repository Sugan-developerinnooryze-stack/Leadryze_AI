import { DynamicStructuredTool } from '@langchain/core/tools';
import { AgentTool, ToolSurface } from './tool.types';
import { searchWebsiteKnowledgeTool } from './search-website-knowledge.tool';
import { checkMeetingAvailabilityTool } from './check-availability.tool';
import { bookMeetingTool } from './book-meeting.tool';
import { searchProductsTool } from './search-products.tool';
import { getProductDetailsTool } from './get-product-details.tool';
import { listDepartmentsTool } from './list-departments.tool';
import { listDoctorsTool } from './list-doctors.tool';

// Add each new tool here as it's built. `internal_staff` intentionally has
// no tools in this phase — the existing keyword-matching CRM path in
// base.agent.ts stays exactly as it is; it must never become one of these.
const ALL_TOOLS: AgentTool[] = [
  searchWebsiteKnowledgeTool,
  checkMeetingAvailabilityTool,
  bookMeetingTool,
  searchProductsTool,
  getProductDetailsTool,
  listDepartmentsTool,
  listDoctorsTool,
];

export type ToolHint = 'booking' | 'catalog' | 'website';

// Narrows the bound tool set for a clearly single-purpose turn (never blocks
// it) so that turn doesn't also pay for describing tools it has no use for —
// e.g. a booking-shaped message doesn't need catalog/RAG tools described.
// See isBookingOnlyMessage()/isCatalogOnlyMessage()/isWebsiteOnlyMessage() in
// base.agent.ts for the detectors that choose a hint.
const TOOL_HINT_NAMES: Record<ToolHint, Set<string>> = {
  booking: new Set(['check_meeting_availability', 'book_meeting', 'list_departments', 'list_doctors']),
  catalog: new Set(['search_products', 'get_product_details']),
  website: new Set(['search_website_knowledge']),
};

// Real, confirmed gap this closes: a tenant with bookingRequireTeam:false
// gets zero department-related PROMPT text, but list_departments/
// list_doctors stayed bound and technically callable regardless — nothing
// but the prompt stopped the model from calling them anyway, which Groq's
// small model has been observed to do. base.agent.ts passes this set as
// excludeNames whenever the tenant wants department questions skipped AND
// no team/staff is already resolved for this session (so an in-progress
// pick from before the setting changed doesn't get stranded mid-flow).
export const DEPARTMENT_TOOL_NAMES = new Set(['list_departments', 'list_doctors']);

export function getToolsForSurface(
  surface: ToolSurface,
  opts?: { toolHint?: ToolHint; excludeNames?: Set<string> },
): AgentTool[] {
  let forSurface = ALL_TOOLS.filter((t) => t.surfaces.includes(surface));
  if (opts?.excludeNames?.size) {
    forSurface = forSurface.filter((t) => !opts.excludeNames!.has(t.name));
  }
  if (opts?.toolHint) {
    const names = TOOL_HINT_NAMES[opts.toolHint];
    return forSurface.filter((t) => names.has(t.name));
  }
  return forSurface;
}

// Wraps each AgentTool in a DynamicStructuredTool purely so every provider
// adapter (ChatOpenAI/ChatAnthropic/ChatGroq/ChatGoogleGenerativeAI) converts
// the zod schema into its own function-calling format correctly. The stub
// `func` is never invoked — real execution happens in runner.ts, which needs
// a ToolContext (tenantId/sessionId/visitorId) that LangChain's own dispatch
// has no way to carry.
export function toBindableTools(tools: AgentTool[]): DynamicStructuredTool[] {
  return tools.map(
    (t) =>
      new DynamicStructuredTool({
        name: t.name,
        description: t.description,
        schema: t.schema,
        func: async () => '',
      })
  );
}
