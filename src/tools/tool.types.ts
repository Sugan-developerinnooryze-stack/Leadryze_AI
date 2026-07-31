import { z } from 'zod';

// Which agent surface a tool may be bound to. Explicit allow-list per tool,
// no implicit default — a tool must opt in to 'public_widget' to ever be
// reachable by an anonymous website visitor.
export type ToolSurface = 'public_widget' | 'internal_staff';

export interface ToolContext {
  tenantId: string;
  sessionId: string;
  visitorId?: string;
  pageUrl?: string;
  companyName: string;
  timezone: string;
}

export interface ToolResult {
  ok: boolean;
  summary: string;                    // one line the LLM can read directly
  data?: Record<string, unknown>;     // JSON-serialisable, never a raw CRM/Mongoose doc
}

export interface AgentTool<A = any> {
  name: string;                       // snake_case, stable — this is an LLM-visible API
  description: string;                // when to call it, in plain English
  schema: z.ZodObject<any>;
  surfaces: ToolSurface[];
  execute(args: A, ctx: ToolContext): Promise<ToolResult>;
}
