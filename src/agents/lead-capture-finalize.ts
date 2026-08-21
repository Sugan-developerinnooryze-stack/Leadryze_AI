import { backendClient } from '../services/backend.client';
import { setLeadCaptureState, type LeadCaptureState } from '../memory/conversation.memory';
import { classifyBuyingIntent, mergeBuyingIntent, isRequirementShaped, INTENT_RANK } from './buying-intent';
import { DatasetItemCard } from './dataset-item-card.types';
import { ToolCallLog } from '../tools/runner';
import { logger } from '../utils/logger';

/** A small, structural subset of base.agent.ts's own AgentInput — deliberately
 * NOT importing that type directly, since base.agent.ts imports
 * finalizeWidgetLeadCapture (below) from THIS module, and request-quote-
 * shortcut.ts also imports it from here — importing AgentInput back from
 * base.agent.ts would create a real circular module dependency. Any real
 * AgentInput value already satisfies this shape structurally, so callers
 * pass their own `input` straight through with no conversion needed. */
export interface WidgetLeadCaptureInput {
  tenantId: string;
  sessionId: string;
  message: string;
  visitorId?: string;
  pageUrl?: string;
}

/** The deterministic "given whatever's already known about this visitor,
 * accumulate signal and persist/create/enrich the Lead" tail. Lives in its
 * own module (same reasoning as extract-captured-data.ts) so both
 * base.agent.ts's maybeCaptureWidgetLead() (LLM-assisted extraction first)
 * and request-quote-shortcut.ts's 100% deterministic "Request Quote"/
 * "Request Demo" flow can call the exact same lead creation/enrichment
 * logic without either module depending on the other. This function itself
 * NEVER calls an LLM — that's the whole point: a caller that already has
 * firstName/email/phone from a deterministic source (a regex, a quick-reply
 * button's own literal text) can create/enrich a real Lead with a hard
 * guarantee it won't fail just because Groq/OpenRouter are having a slow
 * moment. Real, confirmed gap this closes: a "Request Quote" click
 * previously went through the SAME general LLM path as any other message,
 * so a provider timeout meant the visitor's contact info was never captured
 * at all, not even a degraded answer. */
export async function finalizeWidgetLeadCapture(
  input: WidgetLeadCaptureInput,
  state: LeadCaptureState,
  toolCallsLog: ToolCallLog[] = [],
  items?: DatasetItemCard[],
): Promise<void> {
  if (!input.visitorId) return;
  try {
    // Buying-intent / requirement / interested-items accumulation — runs
    // every turn, regardless of whether a Lead has been created yet, so the
    // FIRST creation already carries real signal instead of defaults, AND a
    // Lead created early still gets enriched as intent increases later
    // (see the enrichment branch below) — a real fix from review: this used
    // to be a frozen creation-time snapshot, never updated afterward, so a
    // visitor who gave contact info early and only expressed real buying
    // intent several turns later never had that reflected, and a tenant's
    // hot-lead automation would never fire.
    const hasContactInfo = !!(state.email || state.phone);
    const thisTurn = classifyBuyingIntent(input.message, toolCallsLog, hasContactInfo);
    const { result: mergedIntent, isNewHighWaterMark } = mergeBuyingIntent(state, thisTurn);
    state.buyingIntent = mergedIntent.buyingIntent;
    state.leadScore = mergedIntent.leadScore;
    // conversationSummary: deterministic, verbatim capture of the message
    // that set the new high-water mark — no LLM summarization call.
    if (isNewHighWaterMark) state.conversationSummary = input.message;

    // Requirement capture — deliberately NOT gated on "contact info still
    // missing" (a real fix from review: the old design only ran its
    // extraction while name/contact was unknown, so anything a visitor
    // said about quantity/application AFTER giving their email was lost).
    if (isRequirementShaped(input.message)) state.requirement = input.message;

    if (items?.length) {
      const existing = state.interestedItems ?? [];
      for (const item of items) {
        if (!existing.some((e) => e.datasetId === item.datasetId && e.recordId === item.recordId)) {
          existing.push({ datasetId: item.datasetId, recordId: item.recordId, title: item.title, datasetVersion: item.datasetVersion });
        }
      }
      state.interestedItems = existing.slice(0, 10);
    }

    if (state.leadCreated) {
      // Enrichment path — only call the backend when something MEANINGFUL
      // changed since the last thing actually sent for this lead, not on
      // every turn.
      const bucketIncreased = !state.lastSentBuyingIntent
        || INTENT_RANK[state.buyingIntent!] > INTENT_RANK[state.lastSentBuyingIntent];
      const requirementChanged = !!state.requirement && state.requirement !== state.lastSentRequirement;
      const hasNewItem = (state.interestedItems?.length ?? 0) > (state.lastSentItemCount ?? 0);
      if ((bucketIncreased || requirementChanged || hasNewItem) && state.leadId) {
        const result = await backendClient.updateLeadFromWidget(input.tenantId, state.leadId, {
          leadScore: state.leadScore,
          buyingIntent: state.buyingIntent,
          interestedItems: state.interestedItems,
          requirement: state.requirement,
          conversationSummary: state.conversationSummary,
        });
        if (result.success) {
          state.lastSentBuyingIntent = state.buyingIntent;
          state.lastSentRequirement  = state.requirement;
          state.lastSentItemCount    = state.interestedItems?.length;
        }
      }
      await setLeadCaptureState(input.tenantId, input.sessionId, state);
      return;
    }

    await setLeadCaptureState(input.tenantId, input.sessionId, state);

    if (state.firstName && (state.email || state.phone)) {
      const result = await backendClient.createLeadFromWidget({
        tenantId:  input.tenantId,
        sessionId: input.sessionId,
        visitorId: input.visitorId,
        sourceUrl: input.pageUrl,
        firstName: state.firstName,
        lastName:  state.lastName,
        email:     state.email,
        phone:     state.phone,
        company:   state.company,
        service:   state.service,
        leadScore: state.leadScore,
        buyingIntent: state.buyingIntent,
        interestedItems: state.interestedItems,
        requirement: state.requirement,
        conversationSummary: state.conversationSummary,
      });
      if (result.success && result.leadId) {
        state.leadCreated = true;
        state.leadId = result.leadId;
        state.lastSentBuyingIntent = state.buyingIntent;
        state.lastSentRequirement  = state.requirement;
        state.lastSentItemCount    = state.interestedItems?.length;
        await setLeadCaptureState(input.tenantId, input.sessionId, state);
      }
    }
  } catch (err) {
    logger.warn('finalizeWidgetLeadCapture failed', { sessionId: input.sessionId, error: (err as Error).message });
  }
}
