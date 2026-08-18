import { SUPPORTED_LANGUAGE_CODES } from '../config/languages';

/** Maps common display names / regional variants to the ISO 639-1 code
 * SUPPORTED_LANGUAGES already uses — the root cause of the live "unsupported
 * language: English" crash was a free-text tenant field ("English") flowing
 * completely unvalidated into Groq's Whisper API, which only accepts real
 * ISO codes. Anything unrecognized returns undefined (auto-detect) rather
 * than forwarding a value guaranteed to error. */
const DISPLAY_NAME_ALIASES: Record<string, string> = {
  english: 'en', hindi: 'hi', tamil: 'ta', telugu: 'te', kannada: 'kn',
  malayalam: 'ml', bengali: 'bn', marathi: 'mr', gujarati: 'gu', punjabi: 'pa',
  urdu: 'ur', indonesian: 'id', malay: 'id', bahasa: 'id',
  chinese: 'zh', mandarin: 'zh', japanese: 'ja', korean: 'ko',
  spanish: 'es', french: 'fr', german: 'de', portuguese: 'pt', italian: 'it',
  dutch: 'nl', russian: 'ru', arabic: 'ar', turkish: 'tr',
};

export function normalizeLanguageCode(input?: string | null): string | undefined {
  if (!input) return undefined;
  const trimmed = input.trim().toLowerCase();
  if (!trimmed) return undefined;

  // Already a bare ISO code (e.g. "en", "hi") — accept as-is if known.
  if (SUPPORTED_LANGUAGE_CODES.has(trimmed)) return trimmed;

  // Regional variants ("en-us", "en_US", "pt-BR") — take the base subtag.
  const base = trimmed.split(/[-_]/)[0];
  if (SUPPORTED_LANGUAGE_CODES.has(base)) return base;

  // Display names ("English", "Hindi", ...).
  const aliased = DISPLAY_NAME_ALIASES[trimmed] || DISPLAY_NAME_ALIASES[base];
  if (aliased) return aliased;

  return undefined;
}
