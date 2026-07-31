import { llm, LLMMessage } from '../core/model-abstraction/llm.provider';
import { logger } from '../utils/logger';

export interface MarketingAgentInput {
  tenantId: string;
  campaignGoal: string;
  targetAudience: string;
  companyName: string;
  channel: 'email' | 'whatsapp' | 'sms';
  tone: 'professional' | 'friendly' | 'urgent' | 'casual';
  productOrService?: string;
}

export interface MarketingAgentOutput {
  subject?: string;
  body: string;
  callToAction: string;
  variants: string[];
}

export async function runMarketingAgent(
  input: MarketingAgentInput
): Promise<MarketingAgentOutput> {
  const charLimit =
    input.channel === 'email'
      ? 'Full email copy with subject line.'
      : `Max 160 characters for ${input.channel}.`;

  const messages: LLMMessage[] = [
    {
      role: 'system',
      content: `You are a conversion copywriter for ${input.companyName}. Write ${input.channel} marketing copy.
Tone: ${input.tone}. Audience: ${input.targetAudience}. ${charLimit}
${input.productOrService ? `Product/Service: ${input.productOrService}` : ''}
Return JSON: { subject?: string, body: string, callToAction: string, variants: string[] }`,
    },
    {
      role: 'user',
      content: `Campaign goal: ${input.campaignGoal}. Generate optimised ${input.channel} copy.`,
    },
  ];

  try {
    const result = await llm.generate(messages);
    const jsonMatch = result.content.match(/\{[\s\S]*\}/);
    if (jsonMatch) return JSON.parse(jsonMatch[0]) as MarketingAgentOutput;

    return { body: result.content, callToAction: 'Contact us today', variants: [] };
  } catch (err) {
    logger.error('Marketing agent error', { error: (err as Error).message });
    return {
      body: `Hi! ${input.companyName} has something exciting for you. Reply to learn more.`,
      callToAction: 'Reply now',
      variants: [],
    };
  }
}
