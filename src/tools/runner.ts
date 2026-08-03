import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ToolMessage } from '@langchain/core/messages/tool';
import { mergeUsageMetadata, type UsageMetadata } from '@langchain/core/messages';
import { llm, LLMMessage, toLangChainMessages, extractUsage } from '../core/model-abstraction/llm.provider';
import { AgentTool, ToolContext, ToolSurface } from './tool.types';
import { toBindableTools } from './registry';
import { logger } from '../utils/logger';

export interface ToolCallLog {
  name: string;
  ok: boolean;
  ms: number;
  /** Whatever the tool's own execute() returned as `data` — generic and
   * forward-compatible so any tool's result can be inspected by the caller
   * (e.g. response-confidence.ts reads search_website_knowledge's topScore),
   * not a bespoke field for one specific tool. */
  data?: Record<string, unknown>;
}

export interface ToolLoopResult {
  content: string;
  provider: string;
  model: string;
  toolCalls: ToolCallLog[];
  /** Summed across every LLM call this loop made (each iteration + the
   * final unbound call) — a multi-iteration tool loop makes several real
   * provider calls, so only counting the last one would undercount cost. */
  usage?: UsageMetadata;
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
    ),
  ]);
}

function contentToString(content: unknown): string {
  return typeof content === 'string' ? content : JSON.stringify(content);
}

// Groq's tool-calling occasionally returns a 200 with no tool_calls array,
// but with its own unexecuted pseudo function-call syntax sitting in
// `content` instead (observed live: "<function=book_meeting>{...}</function>")
// — a real, user-facing bug if ever returned as-is (raw, broken syntax shown
// to a website visitor). This never indicates a genuine answer, so it's
// always treated as a failed turn, never returned directly.
const MALFORMED_TOOL_SYNTAX_RE = /<function[=\s]/i;

function looksMalformed(content: string): boolean {
  return MALFORMED_TOOL_SYNTAX_RE.test(content);
}

/** Drives a bounded tool-calling loop: invoke the tool-bound model, execute
 * whatever tool_calls it returns, feed the results back as ToolMessages, and
 * repeat until it answers in plain prose or a maxIterations/budgetMs limit is
 * hit — at which point one final unbound call is made so the caller always
 * gets a real answer, never a hang. Every tool lookup is re-scoped to
 * `surface` at execute time (not just at bind time) as a second, independent
 * layer of protection on top of the caller only ever passing a
 * surface-filtered tool list in the first place. */
export async function runToolLoop(opts: {
  messages: LLMMessage[];
  tools: AgentTool[];
  ctx: ToolContext;
  surface: ToolSurface;
  maxIterations?: number;
  budgetMs?: number;
  /** A tenant's chosen provider/model for tool-calling (see
   * Tenant.aiConfig.toolModelPreset / TOOL_MODEL_PRESETS) — undefined means
   * "use the global primary/fallback pair", today's unchanged default. Never
   * applied to the forceFallback retry below (that always goes to the real
   * global fallback, on purpose). */
  toolModelOverride?: { provider: string; model: string };
}): Promise<ToolLoopResult> {
  // Kept comfortably under the backend's own axios proxy timeout to /api/chat
  // (raised to 100s for the tool-calling path — see ai.routes.ts) — 35s of
  // iteration budget + worst-case 40s for the final unbound call
  // (primary+fallback each capped at 20s by invokeWithTools) leaves real
  // margin rather than racing that outer boundary.
  const { messages, tools, ctx, surface, maxIterations = 3, budgetMs = 35_000, toolModelOverride } = opts;
  const overrideOpts = toolModelOverride
    ? { overrideProvider: toolModelOverride.provider, overrideModel: toolModelOverride.model }
    : {};

  const toolMap = new Map(tools.filter((t) => t.surfaces.includes(surface)).map((t) => [t.name, t]));
  const bindable = toBindableTools(Array.from(toolMap.values()));
  const deadline = Date.now() + budgetMs;

  let lcMessages: BaseMessage[] = toLangChainMessages(messages);
  const toolCalls: ToolCallLog[] = [];
  let usage: UsageMetadata | undefined;

  for (let iteration = 0; iteration < maxIterations && Date.now() < deadline; iteration++) {
    let aiMsg: AIMessage, provider: string, model: string, asText: string;

    // invokeWithTools() already tries primary→fallback internally on a
    // THROWN error (e.g. Groq rejecting the tool-call attempt outright with
    // an HTTP 400, confirmed live — not every bad tool-call attempt comes
    // back as malformed 200 content). If BOTH already failed inside that one
    // call (e.g. the fallback provider is also down), there is no point
    // attempting our own retry-against-fallback below — the fallback was
    // just tried and just failed — so this goes straight to the final
    // safety net instead of throwing past it uncaught.
    try {
      ({ message: aiMsg, provider, model } = await llm.invokeWithTools(lcMessages, bindable, overrideOpts));
      usage = mergeUsageMetadata(usage, extractUsage(aiMsg));
      asText = contentToString(aiMsg.content);
    } catch (err) {
      logger.warn('Tool-bound LLM call failed entirely (primary and fallback both unavailable) — falling back to a plain unbound call', { error: (err as Error).message });
      break;
    }

    // Retry once, explicitly against the fallback model (still tool-bound)
    // rather than immediately giving up on tools — this covers the OTHER
    // failure shape, where the primary call above genuinely succeeded (no
    // exception) but returned malformed pseudo-syntax as ordinary content.
    // Retrying the identical primary model with the identical input rarely
    // changes that outcome, but a genuinely different model is a real second
    // chance. If the retry produces real tool_calls, the code below handles
    // them exactly like any other iteration; only a retry that's ALSO
    // malformed (or itself throws) falls through to the final unbound call.
    if (!aiMsg.tool_calls?.length && looksMalformed(asText)) {
      logger.warn('Tool-bound LLM call returned malformed function-call syntax — retrying once against the fallback model', { provider, model });
      try {
        ({ message: aiMsg, provider, model } = await llm.invokeWithTools(lcMessages, bindable, { forceFallback: true }));
        usage = mergeUsageMetadata(usage, extractUsage(aiMsg));
        asText = contentToString(aiMsg.content);
      } catch (retryErr) {
        logger.warn('Fallback-model retry itself failed — falling back to a plain unbound call', { error: (retryErr as Error).message });
        break; // same safety net as a malformed retry — never let this exception escape uncaught
      }
    }

    if (!aiMsg.tool_calls?.length) {
      if (looksMalformed(asText)) {
        logger.warn('Retry also returned malformed function-call syntax — falling back to a plain unbound call', { provider, model });
        break; // fall through to the final unbound call below, never return this verbatim
      }
      return { content: asText, provider, model, toolCalls, usage };
    }

    lcMessages = [...lcMessages, aiMsg];

    for (const call of aiMsg.tool_calls) {
      const started = Date.now();
      const tool = toolMap.get(call.name);
      let payload: { ok: boolean; summary: string; data?: Record<string, unknown> };
      let ok = false;

      if (!tool) {
        payload = { ok: false, summary: `Tool "${call.name}" is not available.` };
      } else {
        const parsed = tool.schema.safeParse(call.args);
        if (!parsed.success) {
          payload = { ok: false, summary: `Invalid arguments for "${call.name}".` };
        } else {
          try {
            const result = await withTimeout(tool.execute(parsed.data, ctx), 8000, call.name);
            payload = result;
            ok = result.ok;
          } catch (err) {
            logger.warn('Tool execution failed', { tool: call.name, error: (err as Error).message });
            payload = { ok: false, summary: `"${call.name}" failed and could not complete.` };
          }
        }
      }

      toolCalls.push({ name: call.name, ok, ms: Date.now() - started, data: payload.data });
      lcMessages = [
        ...lcMessages,
        new ToolMessage({ content: JSON.stringify(payload), tool_call_id: call.id ?? call.name }),
      ];
    }
  }

  // Exhausted iterations/budget, OR a malformed mid-loop response was just
  // discarded above — one final, unbound call (no tools attached at all, so
  // there's nothing left for the model to attempt malformed syntax against)
  // over everything gathered so far, so the visitor still gets a real prose
  // answer. If even THIS somehow comes back malformed, a safe static message
  // is returned rather than ever risking raw broken syntax reaching the visitor.
  // "connect you with" deliberately matches one of shouldEscalate()'s own
  // trigger phrases (base.agent.ts) — reached whenever every real fallback
  // has genuinely been exhausted, so this ends in a real human handoff, not
  // a silent dead end with no escalation flag and no lead-capture invitation.
  const HANDOFF_MESSAGE = "I'm having trouble processing that right now — let me connect you with our team so they can help directly. Could I get your name and best way to reach you?";

  try {
    const { message: finalMsg, provider, model } = await llm.invokeWithTools(lcMessages, [], overrideOpts);
    usage = mergeUsageMetadata(usage, extractUsage(finalMsg));
    const finalText = contentToString(finalMsg.content);
    return {
      content: looksMalformed(finalText) ? HANDOFF_MESSAGE : finalText,
      provider, model, toolCalls, usage,
    };
  } catch (finalErr) {
    // Both the primary AND fallback models are unavailable for this final
    // call too (e.g. a genuine provider outage) — the visitor still gets a
    // safe, escalating reply rather than an unhandled exception reaching
    // base.agent.ts's own generic, non-escalating "trouble connecting" catch.
    logger.error('Final unbound call also failed — returning the hardcoded handoff message', { error: (finalErr as Error).message });
    return { content: HANDOFF_MESSAGE, provider: 'none', model: 'none', toolCalls, usage };
  }
}
