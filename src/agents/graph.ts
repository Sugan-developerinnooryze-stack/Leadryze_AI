import { StateGraph, Annotation, END } from '@langchain/langgraph';
import { llm } from '../core/model-abstraction/llm.provider';
import { buildRAGContext } from '../rag/pipeline';
import { backendClient } from '../services/backend.client';
import { formatCRMContextForQuery } from '../services/context.builder';
import { logger } from '../utils/logger';

export type AgentIntent =
  | 'lead_capture'
  | 'crm_query'
  | 'booking'
  | 'escalation'
  | 'followup'
  | 'general';

/* ── State schema using modern Annotation API ───────────────────── */
const AgentStateAnnotation = Annotation.Root({
  tenantId:      Annotation<string>({ reducer: (_x, y) => y ?? _x, default: () => '' }),
  sessionId:     Annotation<string>({ reducer: (_x, y) => y ?? _x, default: () => '' }),
  customerId:    Annotation<string | undefined>({ reducer: (_x, y) => y ?? _x, default: () => undefined }),
  message:       Annotation<string>({ reducer: (_x, y) => y ?? _x, default: () => '' }),
  companyName:   Annotation<string>({ reducer: (_x, y) => y ?? _x, default: () => '' }),
  agentName:     Annotation<string>({ reducer: (_x, y) => y ?? _x, default: () => 'Aria' }),
  language:      Annotation<string>({ reducer: (_x, y) => y ?? _x, default: () => 'English' }),
  intent:        Annotation<AgentIntent | undefined>({ reducer: (_x, y) => y ?? _x, default: () => undefined }),
  ragContext:    Annotation<string | undefined>({ reducer: (_x, y) => y ?? _x, default: () => undefined }),
  crmData:       Annotation<string | undefined>({ reducer: (_x, y) => y ?? _x, default: () => undefined }),
  nextAgent:     Annotation<string | undefined>({ reducer: (_x, y) => y ?? _x, default: () => undefined }),
  finalResponse: Annotation<string | undefined>({ reducer: (_x, y) => y ?? _x, default: () => undefined }),
});

export type AgentState = typeof AgentStateAnnotation.State;

/* ── Node 1: Classify intent using LLM ─────────────────────────── */
async function classifyIntent(state: AgentState): Promise<Partial<AgentState>> {
  logger.info('Graph: classifying intent', { tenantId: state.tenantId });

  const classifyMessages = [
    {
      role: 'system' as const,
      content: `Classify the user message into exactly ONE intent category.
Return ONLY the category name, nothing else.

Categories:
- lead_capture: User wants info about the company, products, services; new/potential customer
- crm_query: Internal user asking about their CRM data ("how many accounts", "show deals", "top customers")
- booking: User wants to schedule an appointment or meeting
- escalation: User is angry, has a complaint, needs urgent help, or requests a human agent
- followup: Follow-up or checking status of a previous interaction
- general: Anything else`,
    },
    { role: 'user' as const, content: state.message },
  ];

  try {
    const result = await llm.generate(classifyMessages, { maxTokens: 20 });
    const raw = result.content.trim().toLowerCase().replace(/[^a-z_]/g, '');
    const validIntents: AgentIntent[] = ['lead_capture', 'crm_query', 'booking', 'escalation', 'followup', 'general'];
    const intent: AgentIntent = validIntents.includes(raw as AgentIntent)
      ? (raw as AgentIntent)
      : 'lead_capture';
    logger.info('Graph: intent classified', { intent, tenantId: state.tenantId });
    return { intent };
  } catch {
    return { intent: 'lead_capture' };
  }
}

/* ── Node 2: Retrieve context based on intent ──────────────────── */
async function retrieveContext(state: AgentState): Promise<Partial<AgentState>> {
  logger.info('Graph: retrieving context', { intent: state.intent, tenantId: state.tenantId });

  const updates: Partial<AgentState> = {};

  // Always try RAG knowledge base
  try {
    const rag = await buildRAGContext(state.message, state.tenantId);
    if (rag) updates.ragContext = rag;
  } catch {
    /* RAG unavailable — continue */
  }

  // For CRM queries, fetch real records from the backend
  if (state.intent === 'crm_query') {
    try {
      const ctx = await backendClient.getTenantContext(state.tenantId);
      if (ctx && Object.keys(ctx.crmModules).length) {
        const msgLower = state.message.toLowerCase();
        let matchedChannel = '';
        let matchedModule = '';

        for (const [channel, modules] of Object.entries(ctx.crmModules)) {
          for (const { module } of modules) {
            if (msgLower.includes(module.toLowerCase()) || msgLower.includes(channel.toLowerCase())) {
              matchedChannel = channel;
              matchedModule = module;
              break;
            }
          }
          if (matchedChannel) break;
        }

        if (matchedChannel && matchedModule) {
          const records = await backendClient.getCRMRecords(
            state.tenantId, matchedChannel, matchedModule, { limit: 20 }
          );
          if (records.length) {
            const lines = records
              .slice(0, 10)
              .map((r) => `  - ${r.displayName || r.externalId}`)
              .join('\n');
            updates.crmData = `${matchedModule} from ${matchedChannel} (${records.length} records):\n${lines}`;
          }
        } else {
          updates.crmData = formatCRMContextForQuery(ctx);
        }
      }
    } catch (err) {
      logger.warn('Graph: CRM fetch failed', { error: (err as Error).message });
    }
  }

  return updates;
}

/* ── Node 3: Route to the correct specialist agent ─────────────── */
async function routeToSpecialist(state: AgentState): Promise<Partial<AgentState>> {
  const agentMap: Record<AgentIntent, string> = {
    lead_capture: 'lead',
    crm_query:    'crm',
    booking:      'booking',
    escalation:   'escalation',
    followup:     'followup',
    general:      'lead',
  };
  const nextAgent = agentMap[state.intent ?? 'lead_capture'] ?? 'lead';
  logger.info('Graph: routing to agent', { nextAgent, intent: state.intent });
  return { nextAgent };
}

/* ── Node 4: Execute the correct agent and produce final response ─ */
async function executeAction(state: AgentState): Promise<Partial<AgentState>> {
  logger.info('Graph: executing action', { agent: state.nextAgent, tenantId: state.tenantId });

  const systemParts: string[] = [
    `You are ${state.agentName} for ${state.companyName}. Language: ${state.language}.`,
  ];

  if (state.ragContext) systemParts.push(`\nKNOWLEDGE BASE:\n${state.ragContext}`);
  if (state.crmData)    systemParts.push(`\nCRM DATA:\n${state.crmData}`);

  switch (state.nextAgent) {
    case 'crm':
      systemParts.push('\nAnswer the CRM data question using ONLY the data above. Be specific with numbers. Do NOT fabricate data.');
      break;
    case 'booking':
      systemParts.push('\nHelp the user schedule an appointment. Ask for preferred time if not provided. Confirm warmly once agreed.');
      break;
    case 'escalation':
      systemParts.push('\nThe user needs urgent help. Show empathy, assure them a human will assist shortly. Ask for their name and best contact number.');
      break;
    case 'followup':
      systemParts.push('\nWrite a warm, brief follow-up response. Reference any prior context if available.');
      break;
    default:
      systemParts.push('\nCapture the customer\'s need, provide helpful info from the knowledge base, and collect their name and contact details.');
  }

  const messages = [
    { role: 'system' as const, content: systemParts.join('\n') },
    { role: 'user' as const, content: state.message },
  ];

  try {
    const result = await llm.generate(messages);
    return { finalResponse: result.content };
  } catch (err) {
    logger.error('Graph: execution failed', { error: (err as Error).message });
    return {
      finalResponse: "I'm having a moment — let me connect you with our team directly.",
    };
  }
}

/* ── Compile the graph ──────────────────────────────────────────── */
const workflow = new StateGraph(AgentStateAnnotation)
  .addNode('classifyIntent',    classifyIntent)
  .addNode('retrieveContext',   retrieveContext)
  .addNode('routeToSpecialist', routeToSpecialist)
  .addNode('executeAction',     executeAction)
  .addEdge('__start__',         'classifyIntent')
  .addEdge('classifyIntent',    'retrieveContext')
  .addEdge('retrieveContext',   'routeToSpecialist')
  .addEdge('routeToSpecialist', 'executeAction')
  .addEdge('executeAction',     END);

export const appGraph = workflow.compile();
