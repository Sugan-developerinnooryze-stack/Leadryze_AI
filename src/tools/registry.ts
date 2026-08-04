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

export function getToolsForSurface(surface: ToolSurface, opts?: { toolHint?: ToolHint }): AgentTool[] {
  const forSurface = ALL_TOOLS.filter((t) => t.surfaces.includes(surface));
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
