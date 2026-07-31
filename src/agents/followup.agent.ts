import { llm, LLMMessage } from '../core/model-abstraction/llm.provider';
import { buildFollowupPrompt } from '../prompts/system.prompts';
import { logger } from '../utils/logger';

export interface FollowupAgentInput {
  tenantId: string;
  customerId: string;
  customerName: string;
  lastInteraction: string;
  channel: 'whatsapp' | 'email' | 'sms';
  companyName: string;
  daysSinceContact: number;
}

export interface FollowupAgentOutput {
  message: string;
  subject?: string;
  shouldSend: boolean;
  reason: string;
}

export async function runFollowupAgent(input: FollowupAgentInput): Promise<FollowupAgentOutput> {
  if (input.daysSinceContact < 1) {
    return { message: '', shouldSend: false, reason: 'Too soon to follow up' };
  }
  if (input.daysSinceContact > 30) {
    return { message: '', shouldSend: false, reason: 'Lead too cold — needs manual review' };
  }

  const systemPrompt = buildFollowupPrompt(
    { companyName: input.companyName },
    input.customerName,
    input.lastInteraction
  );

  const messages: LLMMessage[] = [
    { role: 'system', content: systemPrompt },
    {
      role: 'user',
      content: `Generate a ${input.channel} follow-up for ${input.customerName}. Days since last contact: ${input.daysSinceContact}.`,
    },
  ];

  try {
    const result = await llm.generate(messages);
    const lines = result.content.split('\n').filter(Boolean);
    let subject: string | undefined;
    let message = result.content;

    if (input.channel === 'email' && lines[0]?.toLowerCase().startsWith('subject:')) {
      subject = lines[0].replace(/^subject:\s*/i, '');
      message = lines.slice(1).join('\n').trim();
    }

    logger.info('Followup generated', { tenantId: input.tenantId, customerId: input.customerId });
    return { message, subject, shouldSend: true, reason: `Day ${input.daysSinceContact} followup` };
  } catch (err) {
    logger.error('Followup agent error', { error: (err as Error).message });
    return { message: '', shouldSend: false, reason: 'AI generation failed' };
  }
}
