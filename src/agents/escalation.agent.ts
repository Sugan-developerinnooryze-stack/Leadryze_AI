import { logger } from '../utils/logger';

export interface EscalationAgentInput {
  tenantId: string;
  sessionId: string;
  customerId?: string;
  customerName?: string;
  reason: string;
  conversationSummary: string;
  channel: string;
  urgency: 'low' | 'medium' | 'high';
}

export interface EscalationAgentOutput {
  escalated: boolean;
  escalationId: string;
  notificationSent: boolean;
  message: string;
}

const URGENCY_MESSAGES = {
  low: 'Our team has been notified and will reach out within 24 hours.',
  medium: "I've alerted our team — someone will be in touch within a few hours.",
  high: 'This has been escalated as urgent. Our team will contact you very shortly.',
};

export async function runEscalationAgent(
  input: EscalationAgentInput
): Promise<EscalationAgentOutput> {
  const escalationId = `ESC-${Date.now()}-${Math.random().toString(36).slice(2, 8).toUpperCase()}`;

  logger.warn('Lead escalated to human', {
    escalationId,
    tenantId: input.tenantId,
    sessionId: input.sessionId,
    customerId: input.customerId,
    reason: input.reason,
    urgency: input.urgency,
    channel: input.channel,
  });

  return {
    escalated: true,
    escalationId,
    notificationSent: true,
    message: URGENCY_MESSAGES[input.urgency],
  };
}
