import { SpeechToTextProvider, TextToSpeechProvider } from './types';
import { groqSttProvider, groqTtsProvider } from './groq.provider';

/** Mirrors TOOL_MODEL_PRESETS' own shape (ai/src/config/index.ts) — v1 has
 * exactly one entry each; adding a second provider (ElevenLabs for TTS,
 * Deepgram for STT, etc.) later is a new provider file + one more map entry,
 * not a rewrite of anything that calls these. */
export const VOICE_STT_PRESETS: Record<'groq', SpeechToTextProvider> = {
  groq: groqSttProvider,
};

export const VOICE_TTS_PRESETS: Record<'groq', TextToSpeechProvider> = {
  groq: groqTtsProvider,
};

export function resolveSttProvider(preset: string | undefined): SpeechToTextProvider {
  return VOICE_STT_PRESETS[(preset as 'groq') ?? 'groq'] ?? VOICE_STT_PRESETS.groq;
}

export function resolveTtsProvider(preset: string | undefined): TextToSpeechProvider {
  return VOICE_TTS_PRESETS[(preset as 'groq') ?? 'groq'] ?? VOICE_TTS_PRESETS.groq;
}
