import { runBaseAgent } from './base.agent';
import { resolveTenantConfig } from '../services/context.builder';
import { buildLeadCapturePrompt } from '../prompts/system.prompts';

export interface LeadAgentInput {
  tenantId: string;
  sessionId: string;
  message: string;
  /** Fallback values used if backend is unreachable */
  companyName?: string;
  agentName?: string;
  language?: string;
  customInstructions?: string;
  /** Present only for the public website widget's own conversations — see
   * chat.routes.ts's own comment on why this gates the lead-capture flow. */
  visitorId?: string;
  pageUrl?: string;
}

export interface LeadAgentOutput {
  response: string;
  escalate: boolean;
  capturedData: {
    name?: string;
    email?: string;
    phone?: string;
    [key: string]: string | undefined;
  };
}

export async function runLeadAgent(input: LeadAgentInput): Promise<LeadAgentOutput> {
  // Resolve tenant config from backend — pulls companyName, agentName, language,
  // and CRM/customer context that will be injected into the system prompt.
  const tenantConfig = await resolveTenantConfig(input.tenantId, {
    companyName:        input.companyName,
    agentName:          input.agentName,
    language:           input.language,
    customInstructions: input.customInstructions,
  });

  const systemPrompt = buildLeadCapturePrompt({
    companyName:        tenantConfig.companyName,
    agentName:          tenantConfig.agentName,
    language:           tenantConfig.language,
    customInstructions: input.customInstructions,
    // crmContext and customerContext are already injected inside runBaseAgent
    // via tenantConfig — passing them here would duplicate them
  });

  return runBaseAgent({
    tenantId:    input.tenantId,
    sessionId:   input.sessionId,
    message:     input.message,
    systemPrompt,
    companyName: tenantConfig.companyName,
    agentName:   tenantConfig.agentName,
    language:    tenantConfig.language,
    visitorId:   input.visitorId,
    pageUrl:     input.pageUrl,
  });
}
