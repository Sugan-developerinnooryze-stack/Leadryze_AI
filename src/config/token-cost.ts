/**
 * Static, approximate per-1K-token USD pricing for the providers/models this
 * service is actually configured to use. Directional for the Super Admin
 * usage dashboard, not billing-grade — provider list prices change over time
 * and this isn't wired to any invoicing system.
 */

interface TokenCostRate {
  promptPer1K: number;
  completionPer1K: number;
}

const RATES: Record<string, TokenCostRate> = {
  'openai:gpt-4o-mini': { promptPer1K: 0.00015, completionPer1K: 0.0006 },
  'openai:gpt-4o': { promptPer1K: 0.0025, completionPer1K: 0.01 },
  'anthropic:claude-haiku-4-5-20251001': { promptPer1K: 0.001, completionPer1K: 0.005 },
  'anthropic:claude-sonnet-4-5': { promptPer1K: 0.003, completionPer1K: 0.015 },
  'groq:llama-3.1-8b-instant': { promptPer1K: 0.00005, completionPer1K: 0.00008 },
  'gemini:gemini-2.0-flash-lite': { promptPer1K: 0.000075, completionPer1K: 0.0003 },
  'gemini:gemini-2.0-flash': { promptPer1K: 0.0001, completionPer1K: 0.0004 },
};

const DEFAULT_RATE: TokenCostRate = { promptPer1K: 0.0005, completionPer1K: 0.0015 };

export function estimateCostUsd(
  provider: string,
  model: string,
  promptTokens: number,
  completionTokens: number
): number {
  const rate = RATES[`${provider}:${model}`] || DEFAULT_RATE;
  return (promptTokens / 1000) * rate.promptPer1K + (completionTokens / 1000) * rate.completionPer1K;
}
