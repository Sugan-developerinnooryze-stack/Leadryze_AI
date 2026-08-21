import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGroq } from '@langchain/groq';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, SystemMessage, AIMessage, AIMessageChunk, BaseMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type { ZodSchema } from 'zod';
import { config } from '../../config';
import { logger } from '../../utils/logger';
import { toJsonSchema } from '@langchain/core/utils/json_schema';

/** Gemini tool-schema sanitizer — real, live-confirmed root cause (traced
 * directly, not assumed): `@langchain/google-genai`'s own conversion path
 * for LangChain tools (utils/common.js's convertToGenerativeAITools() ->
 * schemaToGenerativeAIParameters()) hands Gemini's API the RAW output of
 * toJsonSchema() with zero nullable-union sanitization. A Zod `.nullish()`
 * field (used throughout this codebase's tool schemas specifically to
 * handle Groq's own quirk of emitting explicit `null` for an unset
 * optional arg — see search-products.tool.ts's own comment) compiles to
 * `{"type": ["string", "null"]}` — a JSON array — which Gemini's stricter,
 * protobuf-backed schema format rejects outright with a real 400
 * ("Unknown name 'type' ... Proto field is not repeating, cannot start
 * list"), confirmed live against this exact shape. Recursively walks a
 * JSON-schema tree and collapses any nullable-union `type` array into
 * Gemini's expected form: the single real type plus a sibling
 * `nullable: true` flag. */
function isNullSchemaNode(node: unknown): boolean {
  return !!node && typeof node === 'object' && (node as Record<string, unknown>).type === 'null'
    && Object.keys(node as Record<string, unknown>).length === 1;
}

function sanitizeSchemaForGemini(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(sanitizeSchemaForGemini);
  if (node === null || typeof node !== 'object') return node;

  const obj = node as Record<string, unknown>;

  // Second real, live-confirmed nullable shape (distinct from the
  // `"type": ["x","null"]` array case above) — a Zod `.nullable()` on a
  // non-primitive field (an object/array, e.g. QueryPlanSchema's
  // `filters`/`sort`/`aggregation`) produces `"anyOf": [<realSchema>,
  // {"type":"null"}]` instead, which Gemini's schema format also can't
  // accept as-is (its own `anyOf` support is limited/undocumented for this
  // shape — treated defensively the same way: flatten to the real schema
  // plus a sibling `nullable: true`, never leave the raw anyOf/null pair
  // in place). Only collapses the clean 2-branch "real schema OR null"
  // case; a genuine multi-type anyOf falls through unmodified rather than
  // guessing.
  if (Array.isArray(obj.anyOf) && obj.anyOf.length === 2) {
    const [a, b] = obj.anyOf as unknown[];
    const real = isNullSchemaNode(a) ? b : isNullSchemaNode(b) ? a : undefined;
    const hasNullBranch = isNullSchemaNode(a) || isNullSchemaNode(b);
    if (real !== undefined && hasNullBranch) {
      const { anyOf: _anyOf, ...rest } = obj;
      const sanitizedReal = sanitizeSchemaForGemini(real) as Record<string, unknown>;
      const sanitizedRest = sanitizeSchemaForGemini(rest) as Record<string, unknown>;
      const merged = { ...sanitizedReal, ...sanitizedRest };
      merged.nullable = true;
      return merged;
    }
  }

  const sanitized: Record<string, unknown> = {};
  let nullable = false;
  for (const [key, value] of Object.entries(obj)) {
    if (key === 'type' && Array.isArray(value)) {
      const nonNullTypes = value.filter((t) => t !== 'null');
      if (value.includes('null')) nullable = true;
      // A real union of >1 non-null types has no clean single-type Gemini
      // equivalent — falls back to the first one rather than dropping the
      // field entirely (a slightly loose type beats an invalid request).
      sanitized[key] = nonNullTypes[0] ?? 'string';
    } else {
      sanitized[key] = sanitizeSchemaForGemini(value);
    }
  }
  if (nullable) sanitized.nullable = true;
  return sanitized;
}

/** Builds Gemini-safe tool definitions from this codebase's normal
 * DynamicStructuredTool[] — deliberately bypasses @langchain/google-genai's
 * own (buggy, for this codebase's schemas) Zod->Gemini conversion entirely
 * by pre-converting to the OpenAI tool-call shape ({type:'function',
 * function:{name, description, parameters}}) with an already-sanitized
 * JSON schema. Google's own conversion path for THIS shape
 * (convertOpenAIToolToGenAI, utils/tools.js) only strips
 * additionalProperties — it does no further schema rewriting, so a
 * pre-sanitized schema passes through untouched. Only ever used for the
 * `gemini` provider — Groq/OpenAI/Anthropic never see this, and don't need
 * it (none of them reject a nullable-union type array). */
function toGeminiSafeTools(tools: DynamicStructuredTool[]): Array<{ type: 'function'; function: { name: string; description: string; parameters: unknown } }> {
  return tools.map((tool) => ({
    type: 'function' as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: sanitizeSchemaForGemini(toJsonSchema(tool.schema as any)),
    },
  }));
}

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
    // OpenRouter's API is genuinely OpenAI-compatible (same request/response
    // shape, including tools) — reusing ChatOpenAI with a custom baseURL
    // rather than a separate client, per this being real OpenAI-compatible
    // API surface, not a new agent architecture. See .env's own comment on
    // why openai/gpt-oss-20b:free was chosen and its one known limitation
    // (unreliable withStructuredOutput() — tool-calling itself is solid).
    case 'openrouter':
      return new ChatOpenAI({
        model,
        apiKey: config.openrouter.apiKey,
        configuration: { baseURL: 'https://openrouter.ai/api/v1' },
        maxTokens: config.llm.maxTokens,
        temperature: 0.3,
      });
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

  /** Boot-time liveness check for the two models every single call actually
   * depends on (global primary + global fallback) — deliberately NOT every
   * tenant's toolModelPreset override, since pinging N per-tenant
   * combinations on every restart doesn't scale and isn't necessary: a dead
   * override is instead caught gracefully at request time by
   * runToolLoop()'s own short fallback chain (falls through to the global
   * pair checked here). This is the check that would have caught the dead
   * `gemini-2.0-flash-lite` fallback model at deploy time instead of mid
   * conversation with a real visitor. */
  async checkHealth(): Promise<{ primaryOk: boolean; fallbackOk: boolean }> {
    // Generous relative to the per-turn PER_CALL_TIMEOUT_MS values above —
    // this is a one-time, fire-and-forget boot check (see server.ts), not a
    // latency-sensitive per-turn call, and a cold LangChain client's first
    // real request can genuinely take longer than a warm one (confirmed
    // live: the same model via the raw provider SDK responded in ~2s, but
    // this wrapper's very first invoke() took longer) — a tight timeout
    // here risks a false "model is dead" alarm on a model that's actually
    // fine.
    const HEALTH_CHECK_TIMEOUT_MS = 20000;
    const ping = async (model: BaseChatModel, label: string, providerModel: string): Promise<boolean> => {
      try {
        await Promise.race([
          model.invoke([new HumanMessage('ping')]),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`health check timed out after ${HEALTH_CHECK_TIMEOUT_MS}ms`)), HEALTH_CHECK_TIMEOUT_MS)
          ),
        ]);
        return true;
      } catch (err) {
        // Wording note: this is a diagnostic ping only — its result is never
        // stored or checked before a real call, so this does NOT block or
        // skip the model for actual requests. A real turn independently
        // retries this exact model and may still succeed even after this
        // log fires (confirmed live: a prior version of this message
        // ("every call that reaches this model will fail until this is
        // fixed") was read as a hard gate and caused real confusion
        // diagnosing an unrelated live issue — it never was one).
        logger.error(
          `[LLM HEALTH CHECK] ${label} model (${providerModel}) failed a live ping at boot — diagnostic only, does not block real requests to this model`,
          { error: (err as Error).message }
        );
        return false;
      }
    };
    const [primaryOk, fallbackOk] = await Promise.all([
      ping(this.primary, 'primary', `${config.llm.provider}/${config.llm.model}`),
      ping(this.fallback, 'fallback', `${config.llm.fallbackProvider}/${config.llm.fallbackModel}`),
    ]);
    if (primaryOk && fallbackOk) {
      logger.info('[LLM HEALTH CHECK] primary and fallback models both responded to a live ping', {
        primary: `${config.llm.provider}/${config.llm.model}`,
        fallback: `${config.llm.fallbackProvider}/${config.llm.fallbackModel}`,
      });
    }
    return { primaryOk, fallbackOk };
  }

  /** Streaming twin of generate() above — same primary→fallback shape, same
   * LLMResponse return, but calls onChunk with each incremental text delta
   * as it arrives. No tools involved at all (this is the plain, no-tool-loop
   * path — internal_staff has zero bound tools in this codebase today), so
   * there's no tool-call ambiguity to worry about, same as
   * invokeWithToolsStream()'s own reasoning. */
  async generateStream(messages: LLMMessage[], onChunk: (delta: string) => void): Promise<LLMResponse> {
    const lcMessages = toLangChainMessages(messages);

    const runStream = async (model: BaseChatModel, providerName: string, modelName: string) => {
      const stream = await model.stream(lcMessages);
      let accumulated: AIMessageChunk | undefined;
      for await (const chunk of stream) {
        accumulated = accumulated ? accumulated.concat(chunk) : chunk;
        const delta = typeof chunk.content === 'string' ? chunk.content : '';
        if (delta) onChunk(delta);
      }
      if (!accumulated) throw new Error('Empty stream response');
      return {
        content: typeof accumulated.content === 'string' ? accumulated.content : JSON.stringify(accumulated.content),
        provider: providerName,
        model: modelName,
        usage: extractUsage(accumulated as unknown as AIMessage),
      };
    };

    try {
      return await runStream(this.primary, config.llm.provider, config.llm.model);
    } catch (primaryErr) {
      logger.warn('Primary LLM streaming failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      try {
        return await runStream(this.fallback, config.llm.fallbackProvider, config.llm.fallbackModel);
      } catch (fallbackErr) {
        logger.error('All LLM providers failed (streaming)', { error: (fallbackErr as Error).message });
        throw new Error('All LLM providers are unavailable. Please try again later.');
      }
    }
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
  /** withStructuredOutput() accepts either a raw Zod schema OR an
   * already-converted JSON schema object — LangChain detects which was
   * passed. For the `gemini` provider specifically, pass the pre-sanitized
   * JSON schema instead of the raw Zod schema: withStructuredOutput()'s
   * own internal Zod->Gemini conversion is the SAME buggy path already
   * fixed for tool-binding (confirmed live: the identical "Proto field is
   * not repeating, cannot start list" 400 on a genuinely different call
   * site — LLM.generateStructured(), used by the dataset query-router's
   * classifier — not just invokeWithTools()). Groq/OpenAI/Anthropic keep
   * getting the raw Zod schema unchanged; they don't have this problem. */
  private structuredSchemaFor(model: BaseChatModel, schema: ZodSchema<unknown>): unknown {
    return model instanceof ChatGoogleGenerativeAI ? sanitizeSchemaForGemini(toJsonSchema(schema as any)) : schema;
  }

  async generateStructured<T>(messages: LLMMessage[], schema: ZodSchema<T>): Promise<T> {
    try {
      const model = (this.primary as any).withStructuredOutput(this.structuredSchemaFor(this.primary, schema));
      return await model.invoke(messages as any);
    } catch (primaryErr) {
      logger.warn('Primary LLM structured-output failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      const model = (this.fallback as any).withStructuredOutput(this.structuredSchemaFor(this.fallback, schema));
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
    opts: {
      forceFallback?: boolean;
      overrideProvider?: string;
      overrideModel?: string;
      /** Per-call budget in ms — defaults to 20000. Voice callers pass a
       * much tighter value (see runner.ts's fastDegrade path) since a
       * hands-free conversation can't tolerate the same wait a text chat
       * can. */
      timeoutMs?: number;
      /** When true, a primary/override failure is NOT chased by an internal
       * fallback attempt — it throws immediately instead. Lets a caller
       * (runToolLoop's fastDegrade path) bound the TOTAL number of real LLM
       * attempts across a whole turn, rather than each invokeWithTools()
       * call silently spending up to 2 attempts on its own. */
      singleAttempt?: boolean;
    } = {}
  ): Promise<{ message: AIMessage; provider: string; model: string }> {
    const bind = (model: BaseChatModel) =>
      typeof (model as any).bindTools === 'function' && tools.length
        ? (model as any).bindTools(model instanceof ChatGoogleGenerativeAI ? toGeminiSafeTools(tools) : tools)
        : model;

    const PER_CALL_TIMEOUT_MS = opts.timeoutMs ?? 20000;
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
        if (opts.singleAttempt) throw overrideErr;
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
      if (opts.singleAttempt) throw primaryErr;
      logger.warn('Primary LLM tool-call invoke failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      const message = (await withCallTimeout(bind(this.fallback).invoke(lcMessages))) as AIMessage;
      return { message, provider: config.llm.fallbackProvider, model: config.llm.fallbackModel };
    }
  }

  /** Streaming twin of invokeWithTools() above — same primary→fallback shape,
   * same {message, provider, model} return, but calls onChunk with each
   * incremental text delta AS IT ARRIVES rather than only once the whole
   * response is complete.
   *
   * DELIBERATELY ONLY SAFE (and only ever called in this codebase) with
   * `tools=[]` — a genuinely tool-bound streaming call would need to decide,
   * mid-stream, whether the response is going to be a tool_call or plain
   * prose, and speculatively-streamed-then-discarded tool-call text is a
   * real correctness risk (a visitor hearing/reading a stray sentence that
   * gets abandoned once tool_call_chunks appear) — not attempted here. With
   * no tools bound, the response is unambiguously prose from the first
   * chunk, so this is unconditionally safe to stream. See runner.ts's own
   * "final unbound call" (tools=[] there too) for the one caller that uses
   * this today. */
  async invokeWithToolsStream(
    lcMessages: BaseMessage[],
    tools: DynamicStructuredTool[],
    opts: { overrideProvider?: string; overrideModel?: string; timeoutMs?: number; singleAttempt?: boolean; forceFallback?: boolean } = {},
    onChunk?: (delta: string) => void,
  ): Promise<{ message: AIMessage; provider: string; model: string }> {
    const bind = (model: BaseChatModel) =>
      typeof (model as any).bindTools === 'function' && tools.length
        ? (model as any).bindTools(model instanceof ChatGoogleGenerativeAI ? toGeminiSafeTools(tools) : tools)
        : model;

    const PER_CALL_TIMEOUT_MS = opts.timeoutMs ?? 20000;

    const runStream = async (model: BaseChatModel, providerName: string, modelName: string) => {
      const stream = await bind(model).stream(lcMessages);
      let accumulated: AIMessageChunk | undefined;
      const start = Date.now();
      for await (const chunk of stream) {
        if (Date.now() - start > PER_CALL_TIMEOUT_MS) {
          throw new Error(`LLM stream timed out after ${PER_CALL_TIMEOUT_MS}ms`);
        }
        accumulated = accumulated ? accumulated.concat(chunk) : chunk;
        const delta = typeof chunk.content === 'string' ? chunk.content : '';
        if (delta && onChunk) onChunk(delta);
      }
      if (!accumulated) throw new Error('Empty stream response');
      return { message: accumulated as unknown as AIMessage, provider: providerName, model: modelName };
    };

    // Mirrors invokeWithTools()'s own forceFallback handling — runner.ts's
    // final-call retry (on a real, confirmed Groq quirk: rejecting its own
    // tool-call-shaped output when tools=[]) needs the SAME capability on
    // the streamed variant, not just the plain one, so a continuous-voice
    // caller's onChunk still gets real incremental output on a successful
    // retry instead of silently falling through with no forceFallback effect.
    if (opts.forceFallback) {
      return await runStream(this.fallback, config.llm.fallbackProvider, config.llm.fallbackModel);
    }

    if (opts.overrideProvider && opts.overrideModel) {
      const overrideModel = this.getOverrideModel(opts.overrideProvider, opts.overrideModel);
      try {
        return await runStream(overrideModel, opts.overrideProvider, opts.overrideModel);
      } catch (overrideErr) {
        if (opts.singleAttempt) throw overrideErr;
        logger.warn('Tenant tool-model override streaming failed, falling back to the global fallback pair', {
          overrideProvider: opts.overrideProvider, overrideModel: opts.overrideModel,
          error: (overrideErr as Error).message,
        });
        return await runStream(this.fallback, config.llm.fallbackProvider, config.llm.fallbackModel);
      }
    }

    try {
      return await runStream(this.primary, config.llm.provider, config.llm.model);
    } catch (primaryErr) {
      if (opts.singleAttempt) throw primaryErr;
      logger.warn('Primary LLM streaming invoke failed, switching to fallback', {
        error: (primaryErr as Error).message,
      });
      return await runStream(this.fallback, config.llm.fallbackProvider, config.llm.fallbackModel);
    }
  }
}

export const llm = new ModelAbstractionLayer();
