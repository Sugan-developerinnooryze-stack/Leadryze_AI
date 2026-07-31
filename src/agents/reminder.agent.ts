import { llm, LLMMessage } from '../core/model-abstraction/llm.provider';
import { logger } from '../utils/logger';

export interface ReminderAgentInput {
  tenantId: string;
  customerName: string;
  appointmentDate: string;
  appointmentTime: string;
  service: string;
  companyName: string;
  channel: 'whatsapp' | 'sms' | 'email';
  hoursBeforeAppointment: number;
}

export interface ReminderAgentOutput {
  message: string;
  subject?: string;
}

export async function runReminderAgent(input: ReminderAgentInput): Promise<ReminderAgentOutput> {
  const messages: LLMMessage[] = [
    {
      role: 'system',
      content: `You write appointment reminder messages for ${input.companyName}. Keep it friendly, brief, and include key details. Return JSON: { message: string, subject?: string }`,
    },
    {
      role: 'user',
      content: `Reminder for ${input.customerName}: ${input.service} appointment on ${input.appointmentDate} at ${input.appointmentTime}. Channel: ${input.channel}. This is ${input.hoursBeforeAppointment}h before.`,
    },
  ];

  try {
    const result = await llm.generate(messages);
    const match = result.content.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);

    return { message: result.content };
  } catch (err) {
    logger.error('Reminder agent error', { error: (err as Error).message });
    return {
      message: `Hi ${input.customerName}, just a reminder of your ${input.service} appointment with ${input.companyName} on ${input.appointmentDate} at ${input.appointmentTime}. See you then!`,
    };
  }
}
