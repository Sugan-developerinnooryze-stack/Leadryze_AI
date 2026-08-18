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
  /** The visitor's own raw text for THIS turn — lets a tool verify an
   * LLM-supplied argument (e.g. book_meeting's firstName/email/phone) was
   * actually said by the visitor rather than invented by the model to force
   * a required field through. Optional since most tools don't need it. */
  rawMessage?: string;
  /** Tenant-configurable booking requirements (Tenant.widget.booking) —
   * threaded through from ResolvedTenantConfig so book_meeting's readiness
   * gate can enforce them without a redundant tenant-config fetch. */
  bookingRequireTeam?: boolean;
  bookingRequireService?: boolean;
  bookingContactRequirement?: 'email_only' | 'phone_only' | 'email_or_phone' | 'email_and_phone';
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
