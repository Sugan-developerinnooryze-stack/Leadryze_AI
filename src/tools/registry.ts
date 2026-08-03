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

// Names bound when a message reads as booking-only — narrows the tool set
// (never blocks it) so a clearly booking-shaped turn doesn't also pay for
// describing catalog/RAG tools it has no use for.
const BOOKING_TOOL_NAMES = new Set(['check_meeting_availability', 'book_meeting', 'list_departments', 'list_doctors']);

export function getToolsForSurface(surface: ToolSurface, opts?: { bookingOnly?: boolean }): AgentTool[] {
  const forSurface = ALL_TOOLS.filter((t) => t.surfaces.includes(surface));
  if (opts?.bookingOnly) {
    return forSurface.filter((t) => BOOKING_TOOL_NAMES.has(t.name));
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
