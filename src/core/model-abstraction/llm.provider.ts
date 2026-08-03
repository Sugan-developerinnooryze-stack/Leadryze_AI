import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGroq } from '@langchain/groq';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type { ZodSchema } from 'zod';
import { config } from '../../config';
import { logger } from '../../utils/logger';

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMResponse {
  content: string;
  provider: string;
  model: string;
  /** Real token counts, when the provider reports them (all 4 configured
   * providers do via LangChain's own usage_metadata) — used for per-tenant
   * token/cost tracking. Absent rather than zeroed when unavailable, so
   * callers can distinguish "not reported" from "reported as zero". */
  usage?: UsageMetadata;
}

/** Pulls LangChain's own usage_metadata off a returned AIMessage — already a
 * real, typed field (input_tokens/output_tokens/total_tokens), just never
 * read anywhere in this codebase before now. */
export function extractUsage(message: AIMessage): UsageMetadata | undefined {
  return message.usage_metadata;
}

export interface LLMGenerateOptions {
  maxTokens?: number;
  temperature?: number;
}

function createModel(provider: string, model: string): BaseChatModel {
  switch (provider) {
    case 'anthropic':
      return new ChatAnthropic({
        model,
        apiKey: config.anthropic.apiKey,
        maxTokens: config.llm.maxTokens,
        temperature: 0.3,
      });
    case 'groq':
      return new ChatGroq({
        model,
        apiKey: config.groq.apiKey,
        temperature: 0.3,
      });
    case 'gemini':
      return new ChatGoogleGenerativeAI({
        model,
        apiKey: config.google.apiKey,
        temperature: 0.3,
      }) as unknown as BaseChatModel;
    case 'openai':
    default:
      return new ChatOpenAI({
        model,
        apiKey: config.openai.apiKey,
        maxTokens: config.llm.maxTokens,
        temperature: 0.3,
      });
  }
}

export function toLangChainMessages(messages: LLMMessage[]): BaseMessage[] {
  return messages.map((m) => {
    if (m.role === 'system') return new SystemMessage(m.content);
    if (m.role === 'assistant') return new AIMessage(m.content);
    return new HumanMessage(m.content);
  });
}

class ModelAbstractionLayer {
  private primary: BaseChatModel;
  private fallback: BaseChatModel;
  // Per-tenant tool-model overrides (see TOOL_MODEL_PRESETS in config/index.ts)
  // are constructed lazily and cached here — most requests use the global
  // primary/fallback pair above and never touch this map at all.
  private overrideModels = new Map<string, BaseChatModel>();

  private getOverrideModel(provider: string, model: string): BaseChatModel {
    const key = `${provider}:${model}`;
    let m = this.overrideModels.get(key);
    if (!m) {
      m = createModel(provider, model);
      this.overrideModels.set(key, m);
    }
    return m;
  }

  constructor() {
    this.primary = createModel(config.llm.provider, config.llm.model);
    this.fallback = createModel(config.llm.fallbackProvider, config.llm.fallbackModel);
    logger.info('Model Abstraction Layer initialized', {
      primary: `${config.llm.provider}/${config.llm.model}`,
      fallback: `${config.llm.fallbackProvider}/${config.llm.fallbackModel}`,
    });
  }

  async generate(messages: LLMMessage[], _options: LLMGenerateOptions = {}): Promise<LLMResponse> {
    const lcMessages = toLangChainMessages(messages);

    try {
      const result = await this.primary.invoke(lcMessages);
      return {
        content: typeof result.content === 'string' ? result.content : JSON.stringify(result.content),
        provider: config.llm.provider,
        model: config.llm.model,
        usage: extractUsage(result),
      };
    } catch (primaryErr) {
      logger.warn('Primary LLM failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      try {
        const result = await this.fallback.invoke(lcMessages);
        return {
          content: typeof result.content === 'string' ? result.content : JSON.stringify(result.content),
          provider: config.llm.fallbackProvider,
          model: config.llm.fallbackModel,
          usage: extractUsage(result),
        };
      } catch (fallbackErr) {
        logger.error('All LLM providers failed', { error: (fallbackErr as Error).message });
        throw new Error('All LLM providers are unavailable. Please try again later.');
      }
    }
  }

  getModel(): BaseChatModel {
    return this.primary;
  }

  /** Structured-output extraction, with the same primary/fallback shape as
   * generate() above — not invented from nothing: base.agent.ts's own
   * extractChatIntent() already calls `(llm.getModel() as any)
   * .withStructuredOutput(schema)` directly today for intent classification,
   * just against the primary model only, bypassing the fallback pair
   * entirely. This generalizes that already-proven technique into the
   * shared abstraction layer rather than introducing a new one. Callers
   * needing provider-specific recovery (e.g. extractChatIntent's own
   * Groq-strict-mode `failed_generation` JSON salvage) still layer that on
   * top themselves — this method only handles the generic
   * primary-then-fallback retry, not every provider's own quirks. */
  async generateStructured<T>(messages: LLMMessage[], schema: ZodSchema<T>): Promise<T> {
    try {
      const model = (this.primary as any).withStructuredOutput(schema);
      return await model.invoke(messages as any);
    } catch (primaryErr) {
      logger.warn('Primary LLM structured-output failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      const model = (this.fallback as any).withStructuredOutput(schema);
      return await model.invoke(messages as any);
    }
  }

  /** Real LLM tool/function-calling, same primary→fallback shape as generate()
   * above (including its {message, provider, model} return shape, so callers
   * that already log provider/model need no changes). Takes and returns raw
   * LangChain messages (not LLMMessage[]) because a tool-calling loop needs
   * to carry AIMessage.tool_calls and ToolMessage results across turns, which
   * the plain {role,content} shape can't express. Providers without
   * bindTools() (none currently wired, but a future one might lack it) fall
   * back to a normal, tool-free invoke — the caller always gets an AIMessage
   * back, just possibly with no tool_calls.
   *
   * Each provider call is individually bounded (PER_CALL_TIMEOUT_MS) — a hung
   * call (no error, no retry logged, just silence) can otherwise consume an
   * entire outer request timeout with no chance for the tool loop's own
   * graceful-degradation path to run. A timed-out primary now falls back
   * exactly like a *failed* primary; a timed-out fallback throws, same as
   * before (the caller — runToolLoop — still degrades to a safe static
   * message rather than ever crashing or leaking raw provider output).
   *
   * KNOWN, DISCLOSED LIMITATION (not fixed by this method, found via live-fire
   * testing): a SECOND tool call within the same conversation (e.g.
   * check_meeting_availability followed by book_meeting) is unreliable in
   * practice — reproduced repeatedly across both Groq (malformed
   * "<function=...>" pseudo-syntax instead of real tool_calls, or an outright
   * per-call timeout) and Gemini (consistent per-call timeouts on this same
   * second-call shape when tried as primary instead). Swapping provider order
   * for tool-bound calls was tried and did not clearly help — both providers
   * showed the same failure pattern — so the simpler, single primary→fallback
   * order used everywhere else in this file is kept rather than adding
   * complexity that isn't proven to help. A single tool call per conversation
   * (e.g. search_website_knowledge, or check_meeting_availability on its own)
   * has been reliable in every live-fire test run. */
  async invokeWithTools(
    lcMessages: BaseMessage[],
    tools: DynamicStructuredTool[],
    opts: { forceFallback?: boolean; overrideProvider?: string; overrideModel?: string } = {}
  ): Promise<{ message: AIMessage; provider: string; model: string }> {
    const bind = (model: BaseChatModel) =>
      typeof (model as any).bindTools === 'function' && tools.length
        ? (model as any).bindTools(tools)
        : model;

    const PER_CALL_TIMEOUT_MS = 20000;
    const withCallTimeout = <T>(p: Promise<T>): Promise<T> =>
      Promise.race([
        p,
        new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`LLM call timed out after ${PER_CALL_TIMEOUT_MS}ms`)), PER_CALL_TIMEOUT_MS)),
      ]);

    // Used by runner.ts's retry strategy: retrying the identical primary
    // model against the identical input rarely changes a malformed-syntax
    // outcome — going straight to the fallback model is a genuine second
    // chance, not just another roll of the same dice. Deliberately checked
    // BEFORE the tenant override below — a retry after a bad response should
    // go to the real, proven global fallback, never back through a
    // per-tenant override that may itself be what just failed.
    if (opts.forceFallback) {
      const message = (await withCallTimeout(bind(this.fallback).invoke(lcMessages))) as AIMessage;
      return { message, provider: config.llm.fallbackProvider, model: config.llm.fallbackModel };
    }

    // Per-tenant tool-model override (Tenant.aiConfig.toolModelPreset) — a
    // tenant opted into a specific provider/model for RAG/catalog/booking
    // tool calls. On failure this still falls back to the real global
    // fallback pair, so a tenant's override never removes the safety net.
    if (opts.overrideProvider && opts.overrideModel) {
      const overrideModel = this.getOverrideModel(opts.overrideProvider, opts.overrideModel);
      try {
        const message = (await withCallTimeout(bind(overrideModel).invoke(lcMessages))) as AIMessage;
        return { message, provider: opts.overrideProvider, model: opts.overrideModel };
      } catch (overrideErr) {
        logger.warn('Tenant tool-model override failed, falling back to the global fallback pair', {
          overrideProvider: opts.overrideProvider, overrideModel: opts.overrideModel,
          error: (overrideErr as Error).message,
        });
        const message = (await withCallTimeout(bind(this.fallback).invoke(lcMessages))) as AIMessage;
        return { message, provider: config.llm.fallbackProvider, model: config.llm.fallbackModel };
      }
    }

    try {
      const message = (await withCallTimeout(bind(this.primary).invoke(lcMessages))) as AIMessage;
      return { message, provider: config.llm.provider, model: config.llm.model };
    } catch (primaryErr) {
      logger.warn('Primary LLM tool-call invoke failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      const message = (await withCallTimeout(bind(this.fallback).invoke(lcMessages))) as AIMessage;
      return { message, provider: config.llm.fallbackProvider, model: config.llm.fallbackModel };
    }
  }
}

export const llm = new ModelAbstractionLayer();
