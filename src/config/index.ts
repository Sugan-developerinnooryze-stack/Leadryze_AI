import dotenv from 'dotenv';
dotenv.config();

export const config = {
  app: {
    env: process.env.NODE_ENV || 'development',
    port: parseInt(process.env.PORT || '5001', 10),
    internalApiKey: process.env.INTERNAL_API_KEY || 'internal-key',
  },
  llm: {
    provider: (process.env.LLM_PROVIDER || 'anthropic') as 'openai' | 'anthropic' | 'gemini' | 'groq' | 'local',
    model: process.env.LLM_MODEL || 'claude-haiku-4-5-20251001',
    fallbackProvider: (process.env.LLM_FALLBACK_PROVIDER || 'openai') as 'openai' | 'anthropic' | 'gemini' | 'groq' | 'local',
    fallbackModel: process.env.LLM_FALLBACK_MODEL || 'gpt-4o-mini',
    maxTokens: parseInt(process.env.MAX_TOKENS_PER_REQUEST || '4096', 10),
  },
  openai: {
    apiKey: process.env.OPENAI_API_KEY || '',
    orgId: process.env.OPENAI_ORG_ID || '',
  },
  anthropic: {
    apiKey: process.env.ANTHROPIC_API_KEY || '',
  },
  google: {
    apiKey: process.env.GOOGLE_API_KEY || '',
  },
  groq: {
    apiKey: process.env.GROQ_API_KEY || '',
  },
  embeddings: {
    provider: process.env.EMBEDDING_PROVIDER || 'voyage',
    model: process.env.EMBEDDING_MODEL || 'voyage-3',
  },
  voyage: {
    apiKey: process.env.VOYAGE_API_KEY || '',
  },
  qdrant: {
    url: process.env.QDRANT_URL || 'http://localhost:6333',
    apiKey: process.env.QDRANT_API_KEY || '',
    collection: process.env.QDRANT_COLLECTION || 'leadryze_knowledge',
  },
  redis: {
    url: process.env.REDIS_URL || '',
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    password: process.env.REDIS_PASSWORD || '',
    tls: process.env.REDIS_TLS === 'true',
  },
  guardrails: {
    enableModeration: process.env.ENABLE_CONTENT_MODERATION !== 'false',
    enablePiiFilter: process.env.ENABLE_PII_FILTER !== 'false',
    maxRequestsPerTenantPerHour: parseInt(
      process.env.MAX_REQUESTS_PER_TENANT_PER_HOUR || '500',
      10
    ),
    maxTemplateAnalysesPerTenantPerDay: parseInt(
      process.env.MAX_TEMPLATE_ANALYSES_PER_TENANT_PER_DAY || '20',
      10
    ),
  },
  // PDF/Image Template Analyzer — a one-shot structured-extraction call, not
  // part of the conversational/RAG agent flow, so it calls @google/genai
  // directly rather than going through llm.provider.ts's LangChain
  // abstraction (which only supports plain-text messages, no document/image
  // content parts). Gemini natively accepts both PDF and image input with a
  // free tier — chosen after the Anthropic account this originally used hit
  // a billing/credit limit.
  templateAnalyzer: {
    // 'gemini-flash-latest' is Google's maintained alias for their current
    // flash model — verified live against this key: pinned versions like
    // 'gemini-2.5-flash' can get cut off from new API keys/projects (as
    // happened here) even while still listed in the model catalog; the
    // alias avoids re-hitting that wall as models rotate over time.
    model: process.env.TEMPLATE_ANALYZER_MODEL || 'gemini-flash-latest',
    // gemini-flash-latest's real ceiling is 65536 (checked via the live
    // models list for this key) — 8000 was hit exactly on a real dense
    // invoice (candidatesTokenCount 7984/8000, finishReason MAX_TOKENS).
    // 32000 leaves generous headroom while still bounding worst-case cost.
    maxTokens: parseInt(process.env.TEMPLATE_ANALYZER_MAX_TOKENS || '32000', 10),
    maxPdfPages: parseInt(process.env.TEMPLATE_ANALYZER_MAX_PDF_PAGES || '10', 10),
  },
  backend: {
    url: process.env.BACKEND_URL || 'http://localhost:5000',
    internalServiceKey: process.env.INTERNAL_SERVICE_KEY || 'leadryze-service-key-change-in-prod',
  },
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    dir: process.env.LOG_DIR || './logs',
  },
};

/** Per-tenant tool-model selection (Tenant.aiConfig.toolModelPreset) —
 * governs only the RAG/catalog/booking tool-calling path (invokeWithTools()/
 * runToolLoop()), never the plain fast-path/conversational/lead-extraction
 * paths, which keep using config.llm's own primary/fallback pair above
 * unchanged. Each preset resolves to a provider+model pair already used
 * somewhere in this codebase — no new model names invented. The 'openai'
 * preset is labeled "GPT-4o mini" in the UI (not "GPT-4.1", which isn't a
 * model string that exists anywhere in this codebase) since gpt-4o-mini is
 * the only OpenAI model literal already present. */
export const TOOL_MODEL_PRESETS: Record<'groq' | 'anthropic' | 'openai' | 'google', { provider: string; model: string }> = {
  groq:      { provider: 'groq',      model: config.llm.provider === 'groq'      ? config.llm.model : (config.llm.fallbackProvider === 'groq'      ? config.llm.fallbackModel : 'openai/gpt-oss-20b') },
  google:    { provider: 'gemini',    model: config.llm.provider === 'gemini'    ? config.llm.model : (config.llm.fallbackProvider === 'gemini'    ? config.llm.fallbackModel : 'gemini-flash-latest') },
  anthropic: { provider: 'anthropic', model: config.llm.provider === 'anthropic' ? config.llm.model : (config.llm.fallbackProvider === 'anthropic' ? config.llm.fallbackModel : 'claude-haiku-4-5-20251001') },
  openai:    { provider: 'openai',    model: config.llm.provider === 'openai'    ? config.llm.model : (config.llm.fallbackProvider === 'openai'    ? config.llm.fallbackModel : 'gpt-4o-mini') },
};

/** Curated Cartesia voice presets for CONTINUOUS voice (real, verified IDs —
 * fetched live from Cartesia's own GET /voices API using the configured
 * CARTESIA_API_KEY, not guessed). Kept small and clean-professional rather
 * than emotion-variant/persona-named voices Cartesia's own catalog also has
 * many of (e.g. "Carson - Angry Friendly Support") — a tenant picks Male or
 * Female, not a mood. Push-to-talk's own Groq/Orpheus voice stays separate
 * (its voice names remain unverified pending that model's terms being
 * accepted) — this preset map is Cartesia-only, matching
 * Tenant.widget.voice.voicePreset.provider's own 'cartesia'-only type. */
export const CARTESIA_VOICE_PRESETS: Record<'female' | 'male', {
  provider: 'cartesia'; voiceId: string; displayName: string; gender: 'male' | 'female'; language: string;
}> = {
  female: {
    provider: 'cartesia',
    voiceId: '8a1b8af0-c4f6-423f-a268-5507fd4aefdf', // "Denise - Professional Woman"
    displayName: 'Denise (Professional Woman)',
    gender: 'female',
    language: 'en',
  },
  male: {
    provider: 'cartesia',
    voiceId: '5cf0e4d9-ca2b-4fd5-81fa-89db3b645539', // "Derrick - Professional Man"
    displayName: 'Derrick (Professional Man)',
    gender: 'male',
    language: 'en',
  },
};

if (config.app.env === 'production') {
  const aiInsecureChecks: Array<[string, string, string]> = [
    ['INTERNAL_API_KEY',     config.app.internalApiKey,         'internal-key'],
    ['INTERNAL_SERVICE_KEY', config.backend.internalServiceKey, 'leadryze-service-key-change-in-prod'],
  ];
  for (const [name, value, badDefault] of aiInsecureChecks) {
    if (!value || value === badDefault || value.length < 16) {
      throw new Error(
        `[AI CONFIG] ${name} is insecure or missing. Set a strong unique value in production .env before starting.`
      );
    }
  }
}
