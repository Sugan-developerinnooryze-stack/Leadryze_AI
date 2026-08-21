import { runBaseAgent } from './base.agent';
import { resolveTenantConfig } from '../services/context.builder';
import { buildLeadCapturePrompt } from '../prompts/system.prompts';
import { DatasetItemCard } from './dataset-item-card.types';

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
  /** See AgentInput's own comment (base.agent.ts) — defaults to 'text' when
   * omitted, so every existing caller (chat.routes.ts, voice.routes.ts) is
   * unaffected; only the continuous-voice worker sets this explicitly. */
  channel?: 'text' | 'push_to_talk' | 'continuous_voice';
  /** See AgentInput's own comment (base.agent.ts) — only ever set by
   * LeadAgentLLMStream, never by any other caller. */
  abortSignal?: AbortSignal;
  /** See AgentInput's own comment (base.agent.ts) — only ever set by
   * LeadAgentLLMStream, never by any other caller. */
  onChunk?: (delta: string) => void;
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
  items?: DatasetItemCard[];
  totalMatches?: number;
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
    channel:     input.channel,
    abortSignal: input.abortSignal,
    onChunk:     input.onChunk,
  });
}
