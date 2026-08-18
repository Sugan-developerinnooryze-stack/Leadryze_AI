/** Plain async-function interfaces, deliberately not tied to Express req/res
 * or to any one provider's SDK shape — this is what lets a future streaming
 * variant (or a second provider like ElevenLabs/Deepgram) slot in later
 * without reshaping the call sites in voice.routes.ts. */
export interface SpeechToTextProvider {
  transcribe(audio: Buffer, mimeType: string, language?: string): Promise<string>;
}

export interface TextToSpeechProvider {
  speak(text: string, voice?: string): Promise<Buffer>;
}
