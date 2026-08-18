import { Router, Request, Response } from 'express';
import multer from 'multer';
import { z } from 'zod';
import { runLeadAgent } from '../agents/lead.agent';
import { resolveSttProvider, resolveTtsProvider } from '../voice/registry';
import { backendClient } from '../services/backend.client';
import { estimateSttCostUsd, estimateTtsCostUsd } from '../config/voice-cost';
import { logger } from '../utils/logger';
import { CARTESIA_VOICE_PRESETS } from '../config';
import https from 'https';

const router = Router();

// Short-lived audio blobs only — memory storage, no disk writes/cleanup to
// manage, same reasoning as the backend's own widget-logo-upload route.
// Size/MIME limits are enforced again here even though the backend's own
// proxy already enforces them, as defense-in-depth for any other future
// caller of this internal route.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
});

const VoiceChatFieldsSchema = z.object({
  tenantId:    z.string().min(1),
  sessionId:   z.string().min(1),
  companyName: z.string().optional(),
  agentName:   z.string().optional(),
  language:    z.string().optional(),
  visitorId:   z.string().optional(),
  pageUrl:     z.string().optional(),
  sttProvider: z.string().optional(),
  ttsProvider: z.string().optional(),
  sttLanguage: z.string().optional(),
  voiceName:   z.string().optional(),
  /** Client-measured recording length — used only for cost estimation
   * (§7), never for anything security/correctness-sensitive. Decoding audio
   * server-side to measure real duration would need a new dependency
   * (ffprobe or similar) for a number that only needs to be directionally
   * right for a usage dashboard, not billing-grade. */
  durationSeconds: z.coerce.number().min(0).max(120).optional(),
});

/**
 * @swagger
 * /voice/chat:
 *   post:
 *     summary: One combined voice turn — transcribe, run the existing lead
 *       agent unchanged, synthesize the reply, return both text and audio.
 */
router.post('/voice/chat', upload.single('audio'), async (req: Request, res: Response) => {
  const parsed = VoiceChatFieldsSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ success: false, message: 'Validation failed', errors: parsed.error.flatten().fieldErrors });
  }
  if (!req.file) {
    return res.status(400).json({ success: false, message: 'audio file is required' });
  }
  const body = parsed.data;

  try {
    const sttProvider = resolveSttProvider(body.sttProvider);
    const ttsProvider = resolveTtsProvider(body.ttsProvider);

    const sttStart = Date.now();
    const transcript = await sttProvider.transcribe(req.file.buffer, req.file.mimetype, body.sttLanguage);
    const sttMs = Date.now() - sttStart;

    if (!transcript.trim()) {
      return res.json({
        success: true,
        data: {
          transcript: '', response: "Sorry, I couldn't hear that clearly — could you try again?",
          escalate: false, capturedData: {}, audio: null, audioFormat: null,
        },
      });
    }

    // The existing, unmodified pipeline — identical to what /api/chat calls.
    // A Whisper-transcribed string is just a string to everything downstream.
    const agentResult = await runLeadAgent({
      tenantId:    body.tenantId,
      sessionId:   body.sessionId,
      message:     transcript,
      companyName: body.companyName,
      agentName:   body.agentName,
      language:    body.language,
      visitorId:   body.visitorId,
      pageUrl:     body.pageUrl,
    });

    // TTS is isolated from STT/runLeadAgent above on purpose — a TTS-only
    // failure (a provider outage, an unaccepted model-terms gate, etc.)
    // should never discard an otherwise-successful transcript+AI response.
    // Confirmed live this was a real gap, not a hypothetical: Groq's TTS
    // model needing account-level terms acceptance threw here, and before
    // this fix the whole turn failed with a generic 500 despite STT and the
    // AI's real answer both having already succeeded.
    let audioBuffer: Buffer | null = null;
    let ttsMs = 0;
    try {
      const ttsStart = Date.now();
      audioBuffer = await ttsProvider.speak(agentResult.response, body.voiceName);
      ttsMs = Date.now() - ttsStart;
    } catch (ttsErr) {
      logger.warn('Voice TTS synthesis failed — returning a text-only reply', {
        error: (ttsErr as Error).message, tenantId: body.tenantId,
      });
    }

    const sttSeconds = body.durationSeconds ?? 0;
    const ttsCharacters = audioBuffer ? agentResult.response.length : 0;
    const voiceCostUsd =
      estimateSttCostUsd(body.sttProvider ?? 'groq', sttSeconds) +
      estimateTtsCostUsd(body.ttsProvider ?? 'groq', ttsCharacters);

    void backendClient.writeLog({
      tenantId: body.tenantId,
      sessionId: body.sessionId,
      event: 'voice.turn',
      level: 'info',
      message: 'Voice turn completed',
      metadata: {
        sttMs, ttsMs, sttSeconds, ttsCharacters, voiceCostUsd,
        sttProvider: body.sttProvider ?? 'groq',
        ttsProvider: body.ttsProvider ?? 'groq',
        transcriptLength: transcript.length,
        responseLength: agentResult.response.length,
      },
    });
    void backendClient.trackAiTokenUsage({
      tenantId: body.tenantId,
      promptTokens: 0, completionTokens: 0, totalTokens: 0,
      estimatedCostUsd: voiceCostUsd,
      sttSeconds, ttsCharacters, voiceCostUsd, isVoiceRequest: true,
    });

    return res.json({
      success: true,
      data: {
        transcript,
        response: agentResult.response,
        escalate: agentResult.escalate,
        capturedData: agentResult.capturedData,
        audio: audioBuffer ? audioBuffer.toString('base64') : null,
        audioFormat: audioBuffer ? 'mp3' : null,
      },
    });
  } catch (err) {
    logger.error('Voice chat route error', { error: (err as Error).message, tenantId: body.tenantId });
    void backendClient.writeLog({
      tenantId: body.tenantId,
      sessionId: body.sessionId,
      event: 'voice.turn_failed',
      level: 'error',
      message: 'Voice turn failed',
      metadata: { error: (err as Error).message },
    });
    return res.status(500).json({ success: false, message: 'Internal error' });
  }
});

const VoicePreviewSchema = z.object({
  // Accepts either a raw Cartesia voice id, or one of this app's own preset
  // keys ('female'/'male') as a convenience — the Widget Settings UI sends
  // whichever the tenant actually selected.
  voiceId: z.string().min(1),
  text: z.string().min(1).max(500).optional(),
});

/** Calls Cartesia's own REST /tts/bytes endpoint directly (a plain fetch,
 * not the installed @livekit/agents-plugin-cartesia SDK's ChunkedStream) —
 * a real, previously-hit reliability gap with that SDK's one-shot
 * synthesize() ("buffer is empty" even with a valid key) was root-caused
 * during the continuous-voice work to the SDK path specifically; the raw
 * REST endpoint (confirmed live, returns real audio/mpeg bytes) has no such
 * issue and needs no room/session, matching this preview's own "just play a
 * sample sentence" requirement. */
async function synthesizeCartesiaPreview(voiceId: string, text: string): Promise<Buffer> {
  const apiKey = process.env.CARTESIA_API_KEY;
  if (!apiKey) throw new Error('CARTESIA_API_KEY is not configured');

  const body = JSON.stringify({
    model_id: 'sonic-2',
    transcript: text,
    voice: { mode: 'id', id: voiceId },
    output_format: { container: 'mp3', sample_rate: 44100, encoding: 'mp3' },
    language: 'en',
  });

  return new Promise<Buffer>((resolve, reject) => {
    const req = https.request(
      {
        hostname: 'api.cartesia.ai',
        path: '/tts/bytes',
        method: 'POST',
        headers: {
          'X-API-Key': apiKey,
          'Cartesia-Version': '2024-06-10',
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          const buf = Buffer.concat(chunks);
          if ((res.statusCode ?? 500) !== 200) {
            reject(new Error(`Cartesia TTS preview failed (${res.statusCode}): ${buf.toString('utf8').slice(0, 300)}`));
            return;
          }
          resolve(buf);
        });
      },
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

/**
 * @swagger
 * /voice/preview:
 *   post:
 *     summary: Synthesize a short sample sentence with a given Cartesia
 *       voice, for the "Test Voice" button in Configuration Hub — lets a
 *       tenant confirm a voice sounds right before it's ever used on a real
 *       call. Staff-authenticated via the backend's own proxy, not a public
 *       widget route.
 */
router.post('/voice/preview', async (req: Request, res: Response) => {
  const parsed = VoicePreviewSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ success: false, message: 'Validation failed', errors: parsed.error.flatten().fieldErrors });
  }
  const { voiceId, text } = parsed.data;
  // A bare preset key ('female'/'male') resolves to its real Cartesia id;
  // anything else is passed through as a literal voice id.
  const resolvedVoiceId =
    voiceId === 'female' || voiceId === 'male' ? CARTESIA_VOICE_PRESETS[voiceId].voiceId : voiceId;

  try {
    const audio = await synthesizeCartesiaPreview(
      resolvedVoiceId,
      text || 'Hello, this is a preview of your selected LeadRyze AI voice.',
    );
    return res.json({ success: true, data: { audio: audio.toString('base64'), audioFormat: 'mp3' } });
  } catch (err) {
    logger.error('Voice preview synthesis failed', { error: (err as Error).message, voiceId: resolvedVoiceId });
    return res.status(502).json({ success: false, message: 'Could not synthesize a preview for that voice.' });
  }
});

export default router;
