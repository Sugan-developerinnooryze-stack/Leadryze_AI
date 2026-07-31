import { llm, LLMMessage } from '../core/model-abstraction/llm.provider';
import { logger } from '../utils/logger';

export interface FeedbackAgentInput {
  tenantId: string;
  customerName: string;
  service: string;
  completedDate: string;
  companyName: string;
  channel: 'whatsapp' | 'sms' | 'email';
}

export interface FeedbackAgentOutput {
  message: string;
  subject?: string;
}

export async function runFeedbackAgent(input: FeedbackAgentInput): Promise<FeedbackAgentOutput> {
  const messages: LLMMessage[] = [
    {
      role: 'system',
      content: `Write a short, warm feedback request for ${input.companyName} after a ${input.service} on ${input.completedDate}. The message should feel genuine, not like a generic survey blast. Include a simple 1-5 star or link placeholder. Return JSON: { message: string, subject?: string }`,
    },
    { role: 'user', content: `Send feedback request to ${input.customerName} via ${input.channel}.` },
  ];

  try {
    const result = await llm.generate(messages);
    const match = result.content.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    return { message: result.content };
  } catch (err) {
    logger.error('Feedback agent error', { error: (err as Error).message });
    return {
      message: `Hi ${input.customerName}, thank you for choosing ${input.companyName}! We'd love your feedback on your recent ${input.service}. Rate us 1-5 ⭐ — it means a lot!`,
    };
  }
}
