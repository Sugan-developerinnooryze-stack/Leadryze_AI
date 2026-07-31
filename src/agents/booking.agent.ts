import { llm, LLMMessage } from '../core/model-abstraction/llm.provider';
import { logger } from '../utils/logger';

export interface BookingSlot {
  date: string;
  time: string;
  slot: string;
}

export interface BookingAgentInput {
  tenantId: string;
  sessionId: string;
  customerName: string;
  requestedService: string;
  availableSlots: BookingSlot[];
  companyName: string;
}

export interface BookingAgentOutput {
  response: string;
  bookedSlot?: BookingSlot;
  confirmationMessage: string;
}

export async function runBookingAgent(input: BookingAgentInput): Promise<BookingAgentOutput> {
  const slotsText = input.availableSlots
    .map((s, i) => `${i + 1}. ${s.date} at ${s.time}`)
    .join('\n');

  const messages: LLMMessage[] = [
    {
      role: 'system',
      content: `You are a booking assistant for ${input.companyName}. Help ${input.customerName} book a ${input.requestedService} appointment.\n\nAvailable slots:\n${slotsText}\n\nReturn JSON: { response: string, bookedSlotIndex: number | null, confirmationMessage: string }`,
    },
    {
      role: 'user',
      content: `Show me the available times and help me book ${input.requestedService}.`,
    },
  ];

  try {
    const result = await llm.generate(messages);
    let parsed: { response: string; bookedSlotIndex: number | null; confirmationMessage: string };

    try {
      const jsonMatch = result.content.match(/\{[\s\S]*\}/);
      parsed = jsonMatch
        ? JSON.parse(jsonMatch[0])
        : { response: result.content, bookedSlotIndex: null, confirmationMessage: '' };
    } catch {
      parsed = { response: result.content, bookedSlotIndex: null, confirmationMessage: '' };
    }

    const bookedSlot =
      parsed.bookedSlotIndex !== null
        ? input.availableSlots[parsed.bookedSlotIndex]
        : undefined;

    logger.info('Booking agent completed', { tenantId: input.tenantId, booked: !!bookedSlot });

    return {
      response: parsed.response,
      bookedSlot,
      confirmationMessage:
        parsed.confirmationMessage ||
        (bookedSlot ? `Confirmed: ${bookedSlot.date} at ${bookedSlot.time}` : ''),
    };
  } catch (err) {
    logger.error('Booking agent error', { error: (err as Error).message });
    return {
      response:
        'Let me connect you with our team to check availability. Could I get your contact details?',
      confirmationMessage: '',
    };
  }
}
