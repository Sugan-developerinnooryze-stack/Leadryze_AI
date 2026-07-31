import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { runLeadAgent } from '../agents/lead.agent';
import { runFollowupAgent } from '../agents/followup.agent';
import { runBookingAgent } from '../agents/booking.agent';
import { runEscalationAgent } from '../agents/escalation.agent';
import { runMarketingAgent } from '../agents/marketing.agent';
import { logger } from '../utils/logger';

const router = Router();

/* ── Zod schemas ─────────────────────────────────────────────────── */
const ChatSchema = z.object({
  tenantId:           z.string().min(1),
  sessionId:          z.string().min(1),
  message:            z.string().min(1).max(4000),
  companyName:        z.string().optional(),
  agentName:          z.string().optional(),
  language:           z.string().optional(),
  customInstructions: z.string().max(2000).optional(),
  /** Only ever present when the caller is the public website widget (via
   * the backend's public-widget proxy) — the internal, staff-authenticated
   * proxy (backend/src/modules/ai/ai.routes.ts) never sends these. Used as
   * the gating signal for base.agent.ts's own widget-lead-capture-to-CRM
   * flow, so an internal staff member casually chatting with their own
   * tenant's AI assistant never accidentally creates a spurious Lead. */
  visitorId:          z.string().optional(),
  pageUrl:            z.string().optional(),
});

const FollowupSchema = z.object({
  tenantId:         z.string().min(1),
  customerId:       z.string().min(1),
  customerName:     z.string().min(1),
  companyName:      z.string().optional(),
  lastInteraction:  z.string().optional(),
  channel:          z.enum(['whatsapp', 'email', 'sms']).optional(),
  daysSinceContact: z.number().int().min(0).max(365).optional(),
});

const BookingSchema = z.object({
  tenantId:         z.string().min(1),
  sessionId:        z.string().optional(),
  customerName:     z.string().min(1),
  companyName:      z.string().optional(),
  requestedService: z.string().optional(),
  availableSlots:   z.array(z.object({
    date: z.string(),
    time: z.string(),
    slot: z.string(),
  })).min(1).max(20),
});

const EscalationSchema = z.object({
  tenantId:            z.string().min(1),
  sessionId:           z.string().min(1),
  reason:              z.string().min(1).max(1000),
  customerId:          z.string().optional(),
  customerName:        z.string().optional(),
  conversationSummary: z.string().max(2000).optional(),
  channel:             z.string().optional(),
  urgency:             z.enum(['low', 'medium', 'high']).optional(),
});

const MarketingSchema = z.object({
  tenantId:         z.string().min(1),
  campaignGoal:     z.string().min(1).max(500),
  companyName:      z.string().optional(),
  targetAudience:   z.string().optional(),
  channel:          z.enum(['email', 'whatsapp', 'sms']).optional(),
  tone:             z.enum(['professional', 'friendly', 'urgent', 'casual']).optional(),
  productOrService: z.string().optional(),
});

/* ── Helper: validate and respond with Zod errors ─────────────────── */
function validate<T>(schema: z.ZodSchema<T>, data: unknown, res: Response): T | null {
  const result = schema.safeParse(data);
  if (!result.success) {
    res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors: result.error.flatten().fieldErrors,
    });
    return null;
  }
  return result.data;
}

/**
 * @swagger
 * /chat:
 *   post:
 *     summary: Process an inbound lead message through the AI agent
 *     description: |
 *       Auto-fetches tenant config (company name, agent name, language, CRM data) from the backend.
 *       companyName/agentName/language are optional — used as fallback if backend is unavailable.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [tenantId, sessionId, message]
 *             properties:
 *               tenantId:           { type: string }
 *               sessionId:          { type: string }
 *               message:            { type: string, maxLength: 4000 }
 *               companyName:        { type: string, description: "Fallback if backend unavailable" }
 *               agentName:          { type: string }
 *               language:           { type: string }
 *               customInstructions: { type: string, maxLength: 2000 }
 *     responses:
 *       200:
 *         description: AI response with captured lead data
 */
router.post('/chat', async (req: Request, res: Response) => {
  const body = validate(ChatSchema, req.body, res);
  if (!body) return;

  try {
    const result = await runLeadAgent(body);
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Chat route error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Internal error' });
  }
});

/**
 * @swagger
 * /followup:
 *   post:
 *     summary: Generate a follow-up message for a lead
 *     description: |
 *       Generates a personalised follow-up message for a customer.
 *       The AI uses the tenant's CRM data and templates if available.
 *       daysSinceContact must be between 1 and 30 — outside that range the AI decides not to send.
 */
router.post('/followup', async (req: Request, res: Response) => {
  const body = validate(FollowupSchema, req.body, res);
  if (!body) return;

  try {
    const result = await runFollowupAgent({
      tenantId:         body.tenantId,
      customerId:       body.customerId,
      customerName:     body.customerName,
      lastInteraction:  body.lastInteraction || '',
      channel:          body.channel ?? 'whatsapp',
      companyName:      body.companyName || 'our company',
      daysSinceContact: body.daysSinceContact ?? 1,
    });
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Followup route error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Internal error' });
  }
});

/**
 * @swagger
 * /booking:
 *   post:
 *     summary: AI booking assistant — presents slots and confirms appointment
 */
router.post('/booking', async (req: Request, res: Response) => {
  const body = validate(BookingSchema, req.body, res);
  if (!body) return;

  try {
    const result = await runBookingAgent({
      tenantId:         body.tenantId,
      sessionId:        body.sessionId || `booking-${Date.now()}`,
      customerName:     body.customerName,
      requestedService: body.requestedService || 'appointment',
      availableSlots:   body.availableSlots,
      companyName:      body.companyName || 'our company',
    });
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Booking route error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Internal error' });
  }
});

/**
 * @swagger
 * /escalate:
 *   post:
 *     summary: Escalate a lead conversation to a human agent
 */
router.post('/escalate', async (req: Request, res: Response) => {
  const body = validate(EscalationSchema, req.body, res);
  if (!body) return;

  try {
    const result = await runEscalationAgent({
      tenantId:            body.tenantId,
      sessionId:           body.sessionId,
      customerId:          body.customerId,
      customerName:        body.customerName,
      reason:              body.reason,
      conversationSummary: body.conversationSummary || '',
      channel:             body.channel || 'unknown',
      urgency:             body.urgency ?? 'medium',
    });
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Escalation route error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Internal error' });
  }
});

/**
 * @swagger
 * /marketing:
 *   post:
 *     summary: Generate marketing copy for a campaign
 */
router.post('/marketing', async (req: Request, res: Response) => {
  const body = validate(MarketingSchema, req.body, res);
  if (!body) return;

  try {
    const result = await runMarketingAgent({
      tenantId:         body.tenantId,
      campaignGoal:     body.campaignGoal,
      targetAudience:   body.targetAudience || 'general audience',
      companyName:      body.companyName || 'our company',
      channel:          body.channel ?? 'email',
      tone:             body.tone ?? 'professional',
      productOrService: body.productOrService,
    });
    return res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Marketing route error', { error: (err as Error).message });
    return res.status(500).json({ success: false, message: 'Internal error' });
  }
});

export default router;
