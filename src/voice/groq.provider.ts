import Groq, { toFile } from 'groq-sdk';
import { config } from '../config';
import { logger } from '../utils/logger';
import { normalizeLanguageCode } from '../utils/normalize-language';
import { SpeechToTextProvider, TextToSpeechProvider } from './types';

// Explicit timeout + zero SDK-level retries, same reasoning as
// core/guardrails/moderation.ts's OpenAI client: a slow/erroring provider
// call should fail fast to the caller's own error handling, not silently
// burn several extra seconds on the SDK's own default retry/backoff first.
const client = new Groq({ apiKey: config.groq.apiKey, timeout: 15000, maxRetries: 0 });

// Groq's own PlayAI TTS model ('playai-tts') was decommissioned after this
// integration was first written against the SDK's static types — confirmed
// live via a direct call to the account's actual model list
// (client.models.list()), not assumed from the SDK's type definitions alone
// (which still list the old literal and don't reflect runtime deprecations).
// 'canopylabs/orpheus-v1-english' is the current active replacement
// (input_modalities:['text'], output_modalities:['speech']).
const TTS_MODEL = 'canopylabs/orpheus-v1-english';
// NOT independently verified against a real API call — this account hasn't
// accepted this model's terms yet (see the live 400 this integration hit:
// "model_terms_required", console.groq.com/playground?model=canopylabs%2Forpheus-v1-english),
// so the exact valid voice-name values couldn't be confirmed experimentally.
// 'tara' is Orpheus's commonly-documented default voice — treat as a
// best-effort placeholder to re-confirm once terms are accepted, not a
// verified value.
const DEFAULT_VOICE = 'tara';

// Groq's API validates the uploaded file's type from the FILENAME's
// extension, not just the multipart part's declared MIME type — confirmed
// live: a WAV buffer sent with type:'audio/wav' but filename 'audio' (no
// extension) was rejected with "file must be one of the following types:
// [flac mp3 mp4 mpeg mpga m4a ogg opus wav webm]" even though the MIME type
// itself was valid and in that exact list. A real filename with a matching
// extension is required.
const EXT_BY_MIME: Record<string, string> = {
  'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/wav': 'wav', 'audio/x-wav': 'wav',
  'audio/mp4': 'mp4', 'audio/mpeg': 'mp3', 'audio/mp3': 'mp3', 'audio/m4a': 'm4a',
  'audio/flac': 'flac', 'audio/opus': 'opus',
};

function filenameFor(mimeType: string): string {
  const base = mimeType.split(';')[0].trim().toLowerCase();
  return `audio.${EXT_BY_MIME[base] ?? 'webm'}`;
}

export const groqSttProvider: SpeechToTextProvider = {
  async transcribe(audio, mimeType, language) {
    const file = await toFile(audio, filenameFor(mimeType), { type: mimeType });
    // Defensive normalization at the point of use (not just at the source
    // field) — a bad/unrecognized value (e.g. the literal "English", the
    // real root cause of a live 400 "unsupported language" crash) falls
    // back to auto-detect (omitting the field) rather than forwarding a
    // value Whisper is guaranteed to reject.
    const normalized = normalizeLanguageCode(language);
    const result = await client.audio.transcriptions.create({
      model: 'whisper-large-v3-turbo',
      file,
      ...(normalized ? { language: normalized } : {}),
    });
    return result.text;
  },
};

export const groqTtsProvider: TextToSpeechProvider = {
  async speak(text, voice) {
    const response = await client.audio.speech.create({
      input: text,
      model: TTS_MODEL,
      voice: voice || DEFAULT_VOICE,
      response_format: 'mp3',
    });
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  },
};

// Both wrapped once more here (rather than left to voice.routes.ts) so every
// call site gets identical "log then rethrow" behavior — the route handler
// decides how to respond to a caller, this module only decides how to log.
export async function transcribeWithLogging(audio: Buffer, mimeType: string, language?: string): Promise<string> {
  try {
    return await groqSttProvider.transcribe(audio, mimeType, language);
  } catch (err) {
    logger.error('Groq STT transcription failed', { error: (err as Error).message });
    throw err;
  }
}

export async function speakWithLogging(text: string, voice?: string): Promise<Buffer> {
  try {
    return await groqTtsProvider.speak(text, voice);
  } catch (err) {
    logger.error('Groq TTS synthesis failed', { error: (err as Error).message });
    throw err;
  }
}
