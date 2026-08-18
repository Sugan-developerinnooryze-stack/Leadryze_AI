import { randomUUID } from 'crypto';
import { llm, DEFAULT_API_CONNECT_OPTIONS, type APIConnectOptions } from '@livekit/agents';
import { runLeadAgent } from '../agents/lead.agent';
import { isBookingOnlyMessage } from '../agents/base.agent';
import { logger } from '../utils/logger';

/** Minimal shape of what's needed from voice.AgentSession for the pre-tool
 * acknowledgement filler below — deliberately narrow (not importing the
 * whole AgentSession type) so this file doesn't need a hard dependency on
 * the session's own full surface, just the one method it actually calls. */
export interface SpeakableSession {
  say(text: string): unknown;
}

/** Filled in by worker.ts right after constructing the real AgentSession —
 * LeadAgentLLM is constructed BEFORE the session exists (the session's own
 * constructor needs the already-built `agent`, which needs this llm
 * instance), so a mutable ref is how the acknowledgement filler below gets
 * a working reference to `session.say()` once it's actually available. */
export interface SessionRef { current: SpeakableSession | null; }

/**
 * Adapts the EXISTING, unmodified runLeadAgent() (the same function
 * /api/chat and the push-to-talk voice route both already call) into
 * LiveKit's own LLM interface, so AgentSession's built-in STT->LLM->TTS
 * orchestration (turn detection, interruption/barge-in, recording) can drive
 * it without re-implementing any of the tool-calling/lead-capture/booking
 * logic natively inside the LiveKit framework. One utterance in, one full
 * text reply out — no token-by-token streaming (runLeadAgent() itself
 * doesn't stream), a deliberate, disclosed latency tradeoff in exchange for
 * reusing 100% of the already-hardened business logic.
 */
export class LeadAgentLLM extends llm.LLM {
  constructor(
    private readonly tenantId: string,
    private readonly sessionId: string,
    private readonly visitorId: string,
    private readonly sessionRef?: SessionRef,
  ) {
    super();
  }

  label(): string {
    return 'LeadAgentLLM';
  }

  override get provider(): string {
    return 'leadryze';
  }

  override get model(): string {
    return 'lead-agent';
  }

  chat({
    chatCtx,
    toolCtx,
    connOptions,
  }: {
    chatCtx: llm.ChatContext;
    toolCtx?: llm.ToolContextLike;
    connOptions?: APIConnectOptions;
  }): llm.LLMStream {
    return new LeadAgentLLMStream(this, this.tenantId, this.sessionId, this.visitorId, this.sessionRef, {
      chatCtx,
      toolCtx,
      connOptions,
    });
  }
}

class LeadAgentLLMStream extends llm.LLMStream {
  constructor(
    llmInstance: LeadAgentLLM,
    private readonly tenantId: string,
    private readonly sessionId: string,
    private readonly visitorId: string,
    private readonly sessionRef: SessionRef | undefined,
    params: {
      chatCtx: llm.ChatContext;
      toolCtx?: llm.ToolContextLike;
      connOptions?: APIConnectOptions;
    },
  ) {
    super(llmInstance, {
      chatCtx: params.chatCtx,
      toolCtx: params.toolCtx,
      connOptions: params.connOptions ?? DEFAULT_API_CONNECT_OPTIONS,
    });
  }

  protected async run(): Promise<void> {
    // START/END logging around the WHOLE run() body — added specifically to
    // empirically prove (or disprove) that one spoken turn now produces
    // exactly one run() invocation, after disabling LiveKit's default
    // preemptive generation (worker.ts's session.start() call) — that
    // feature was confirmed, by tracing @livekit/agents' own source, to
    // invoke this LLM interface up to 4 times per utterance (once per
    // interim STT transcript, plus a possible final call). A real call's
    // logs should show exactly one START/END pair per spoken turn now; more
    // than one means the fan-out wasn't actually eliminated.
    const requestId = randomUUID();
    const turnStart = Date.now();
    logger.info('LeadAgentLLM: turn START', { tenantId: this.tenantId, sessionId: this.sessionId, requestId });
    try {
      await this.runTurn(requestId);
    } finally {
      logger.info('LeadAgentLLM: turn END', { tenantId: this.tenantId, sessionId: this.sessionId, requestId, ms: Date.now() - turnStart });
    }
  }

  private async runTurn(requestId: string): Promise<void> {
    // Find the most recent user message in the chat context — that's the
    // just-transcribed utterance AgentSession wants a reply to.
    const items = this.chatCtx.items;
    let messageText = '';
    for (let i = items.length - 1; i >= 0; i--) {
      const item = items[i];
      if (item.type === 'message' && item.role === 'user') {
        messageText = (item as llm.ChatMessage).textContent || '';
        break;
      }
    }

    if (!messageText.trim()) {
      this.queue.put({
        id: randomUUID(),
        delta: { role: 'assistant', content: "Sorry, I didn't catch that — could you say it again?" },
      });
      return;
    }

    // Real fix for a confirmed bug: this call previously had zero reference
    // to this.abortController at all, so an "interrupted" turn (the
    // framework's own abort() on this stream) never actually stopped the
    // backend call — it kept running in the background, still competing for
    // the Redis turn lock, well after the framework had moved on. Racing
    // against the abort signal here means a superseded turn's reply is never
    // emitted (no stale TTS output for a question the visitor already moved
    // past), and runLeadAgent() itself checks the SAME signal at its own two
    // internal checkpoints (before real work starts, and right before the
    // single most expensive stage) so an abandoned turn frees the lock
    // early instead of running its full, otherwise-uncancellable duration.
    const signal = this.abortController.signal;
    if (signal.aborted) return; // already superseded before this even started

    let aborted = false;
    const abortedPromise = new Promise<'aborted'>((resolve) => {
      signal.addEventListener('abort', () => resolve('aborted'), { once: true });
    });

    // Deterministic, bounded "let me check that" filler — reuses the SAME
    // keyword-hint detector base.agent.ts's own tool-set narrowing already
    // uses, not a new heuristic. Only for genuinely booking/availability-
    // shaped messages, where a real tool call is the likely next step and
    // can legitimately take a few seconds — never counted as "the response"
    // itself. Fire-and-forget: never awaited, never blocks the real call.
    if (isBookingOnlyMessage(messageText)) {
      try { this.sessionRef?.current?.say('Sure, let me check that for you.'); } catch { /* best-effort only */ }
    }

    // Real, bounded latency fix: when the underlying LLM call is one of the
    // two shapes that are unconditionally safe to stream (no tools bound —
    // see runner.ts's invokeWithToolsStream() comment for why the other,
    // tool-bound calls are never streamed), each text delta is forwarded to
    // TTS as it arrives instead of only once the whole reply is complete —
    // this is what actually lets Cartesia's own TTS.stream() start speaking
    // the first sentence early rather than waiting for the full response.
    // Falls back to emitting the complete response as one chunk (today's
    // exact behavior) whenever streaming didn't happen for this particular
    // turn (a tool call was involved, the call was aborted, etc.).
    let streamed = false;
    const onChunk = (delta: string) => {
      if (!delta) return;
      streamed = true;
      this.queue.put({ id: randomUUID(), delta: { role: 'assistant', content: delta } });
    };

    try {
      const winner = await Promise.race([
        runLeadAgent({
          tenantId: this.tenantId,
          sessionId: this.sessionId,
          message: messageText,
          visitorId: this.visitorId,
          channel: 'continuous_voice',
          abortSignal: signal,
          onChunk,
        }),
        abortedPromise,
      ]);

      if (winner === 'aborted' || signal.aborted) {
        aborted = true;
        logger.info('LeadAgentLLM: turn abandoned (superseded by a later turn) — discarding reply', {
          tenantId: this.tenantId, sessionId: this.sessionId, requestId,
        });
        return;
      }

      // An empty response means the turn-lock's own repeated-fallback
      // suppression kicked in (base.agent.ts's shouldSpeakTurnLockFallback)
      // — a burst of colliding turns should stay silent after the first
      // spoken "still working", not emit an empty utterance to TTS.
      if (!winner.response) return;

      // Already fully emitted, incrementally, via onChunk above — emitting
      // the complete string again here would duplicate every word.
      if (streamed) return;

      this.queue.put({
        id: randomUUID(),
        delta: { role: 'assistant', content: winner.response },
      });
    } catch (err) {
      if (signal.aborted || aborted) {
        // A failure from an already-abandoned turn isn't worth surfacing —
        // the visitor has already moved on to a different question.
        return;
      }
      logger.error('LeadAgentLLM: runLeadAgent failed', {
        tenantId: this.tenantId,
        sessionId: this.sessionId,
        requestId,
        error: (err as Error).message,
      });
      this.queue.put({
        id: randomUUID(),
        delta: {
          role: 'assistant',
          content: "Sorry, I'm having trouble right now — let me connect you with our team.",
        },
      });
    }
  }
}
