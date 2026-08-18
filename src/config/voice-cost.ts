/**
 * Static, approximate USD pricing for the voice providers this service is
 * actually configured to use — same "directional, not billing-grade"
 * disclaimer as token-cost.ts's own LLM rates. STT is billed per audio
 * second, TTS per output character (both providers' real billing units).
 */

const STT_RATE_PER_SECOND: Record<string, number> = {
  groq: 0.0000111, // ~$0.04/hour on Groq's whisper-large-v3-turbo
};

const TTS_RATE_PER_CHAR: Record<string, number> = {
  groq: 0.00005, // ~$50 / 1M characters on Groq's playai-tts
};

export function estimateSttCostUsd(provider: string, seconds: number): number {
  return seconds * (STT_RATE_PER_SECOND[provider] ?? STT_RATE_PER_SECOND.groq);
}

export function estimateTtsCostUsd(provider: string, characters: number): number {
  return characters * (TTS_RATE_PER_CHAR[provider] ?? TTS_RATE_PER_CHAR.groq);
}

/**
 * Continuous-voice (LiveKit) cost estimate — three separate vendor meters,
 * same "directional, not billing-grade" disclaimer as above. NOT verified
 * against your actual account/contract pricing (list-price ballpark figures
 * only) — adjust these constants once you have real invoiced rates from
 * LiveKit Cloud, Deepgram, and Cartesia.
 */
const LIVEKIT_RATE_PER_MINUTE = 0.003; // ballpark LiveKit Cloud per-participant-minute rate
const DEEPGRAM_STT_RATE_PER_SECOND = 0.0000433; // ballpark Nova-2 streaming rate (~$0.0026/min)
const CARTESIA_TTS_RATE_PER_CHAR = 0.00003; // ballpark Cartesia per-character rate

export function estimateContinuousVoiceCostUsd(params: {
  minutes: number; deepgramSttSeconds: number; cartesiaTtsCharacters: number;
}): number {
  return (
    params.minutes * LIVEKIT_RATE_PER_MINUTE +
    params.deepgramSttSeconds * DEEPGRAM_STT_RATE_PER_SECOND +
    params.cartesiaTtsCharacters * CARTESIA_TTS_RATE_PER_CHAR
  );
}
