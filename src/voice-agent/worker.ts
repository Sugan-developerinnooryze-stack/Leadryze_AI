import { cli, defineAgent, voice, WorkerOptions, type JobContext } from '@livekit/agents';
import * as deepgram from '@livekit/agents-plugin-deepgram';
import * as cartesia from '@livekit/agents-plugin-cartesia';
import * as silero from '@livekit/agents-plugin-silero';
import * as os from 'os';
import { LeadAgentLLM, type SessionRef } from './lead-agent-llm';
import { logger } from '../utils/logger';
import { backendClient, type TenantContext } from '../services/backend.client';
import { estimateContinuousVoiceCostUsd } from '../config/voice-cost';
import { connectRateLimiterRedis, getRedisClient } from '../core/guardrails/rate-limiter';
import { normalizeLanguageCode } from '../utils/normalize-language';

const HEARTBEAT_KEY = 'ai:voiceagent:heartbeat';
const HEARTBEAT_INTERVAL_MS = 15_000;
const HEARTBEAT_TTL_SECONDS = 45; // 3x the interval — a crashed/killed worker's key self-expires, no shutdown cleanup needed

/** Writes a self-expiring liveness key so backend's /system/health can report
 * whether this separate, easy-to-forget-to-start process is actually running
 * — closing what was previously a completely silent failure mode (a widget
 * joins a LiveKit room, but no agent ever joins to converse, with zero error
 * anywhere). Reuses the same Redis client/connection helper rate-limiter.ts
 * already exports and server.ts already calls at bootstrap — no new Redis
 * client, no new connection logic. */
async function startHeartbeat(): Promise<void> {
  await connectRateLimiterRedis();
  const startedAt = new Date().toISOString();

  const write = async () => {
    const client = getRedisClient();
    if (!client) return; // Redis unavailable — heartbeat simply won't appear; no crash, no throw.
    const payload = JSON.stringify({ pid: process.pid, hostname: os.hostname(), startedAt, updatedAt: new Date().toISOString() });
    try {
      await client.setex(HEARTBEAT_KEY, HEARTBEAT_TTL_SECONDS, payload);
    } catch (err) {
      logger.warn('Voice agent: failed to write heartbeat', { error: (err as Error).message });
    }
  };

  await write();
  logger.info('Voice agent: heartbeat reporting started', { intervalMs: HEARTBEAT_INTERVAL_MS, ttlSeconds: HEARTBEAT_TTL_SECONDS });
  setInterval(write, HEARTBEAT_INTERVAL_MS).unref();
}

/** Room names are minted by backend's getVoiceToken() as
 * `voice-<24-hex-char tenantId>-<sessionId>` — see
 * backend/src/modules/public-widget/public-widget.controller.ts. */
const ROOM_NAME_RE = /^voice-([a-f0-9]{24})-(.+)$/;

/** Takes the first sentence of a crawled website summary, capped to a
 * length that still sounds natural spoken aloud (a full crawled paragraph
 * read verbatim by TTS is exactly the kind of robotic-sounding open this
 * exists to avoid). */
function firstSpokenClause(text: string, maxLen = 140): string {
  const firstSentence = text.split(/(?<=[.!?])\s+/)[0]?.trim() ?? text.trim();
  return firstSentence.length > maxLen ? `${firstSentence.slice(0, maxLen).trim()}…` : firstSentence;
}

/** Builds the call's opening line dynamically from THIS tenant's own crawled
 * website content, rather than reading a fixed string an admin typed once
 * (which is what widget.greeting is — the same text chat's first bubble
 * uses, and doesn't automatically stay in sync with whatever site the
 * widget actually gets embedded on). LeadRyze is multi-tenant — every
 * tenant's widget sits on a different real business site — so the crawled
 * websiteProfile (summary/services, built by the RAG crawler) is the
 * closest thing to "what this business actually does" available without an
 * LLM call on this unprompted first turn (session.say() deliberately never
 * touches the LLM — see its own call site's comment). Falls back to
 * widget.greeting, then to a generic template, only when nothing crawled is
 * available at all. */
function buildDynamicGreeting(tenantCtx: TenantContext | null): string {
  const companyName = tenantCtx?.tenant.branding?.companyName ?? tenantCtx?.tenant.name ?? 'this business';
  const profile = tenantCtx?.websiteProfile;

  // Priority order matches how commonly each field is actually populated in
  // practice — a real crawl frequently finds no explicit "services" list or
  // summary paragraph on a site (confirmed live: this exact tenant's crawl
  // has staff/hours but empty summary/services). Deliberately does NOT fall
  // back to naming staff here — this whole session's very first fix was
  // removing a hallucinated staff name from visitor-facing text, and the
  // explicit design rule that followed (never expose staff/team identity
  // when the visitor didn't choose it themselves, gated on
  // bookingRequireTeam) applies just as much to a greeting as to a booking
  // confirmation. `hours` is the next safe, still-useful signal instead.
  let about = '';
  if (profile?.summary) {
    about = ` ${firstSpokenClause(profile.summary)}`;
  } else if (profile?.services?.length) {
    about = ` We offer ${profile.services.slice(0, 3).join(', ')}.`;
  } else if (profile?.hours) {
    about = ` We're open ${profile.hours}.`;
  }

  if (about) {
    return `Hi, welcome to ${companyName}!${about} What are you looking for today?`;
  }

  return tenantCtx?.widget?.greeting
    || `Hi, I'm ${tenantCtx?.tenant.aiConfig?.agentName ?? 'your assistant'} from ${companyName}. How can I help you today?`;
}

export default defineAgent({
  prewarm: async (proc) => {
    // Loaded once per worker process and reused across every job — VAD model
    // loading is comparatively expensive, matching the framework's own
    // documented prewarm convention for exactly this kind of per-process
    // asset (silero.VAD's own JSDoc example does exactly this).
    proc.userData.vad = await silero.VAD.load();
  },

  entry: async (ctx: JobContext) => {
    // Read the room name off the job dispatch itself (ctx.job.room.name) —
    // available immediately, unlike ctx.room.name, which is only populated
    // AFTER ctx.connect() actually joins the room. Reading it too early was a
    // real bug found during live-fire testing: entry() ran, but the room
    // hadn't been joined yet, so ctx.room.name was still empty.
    const roomName = ctx.job.room?.name ?? '';
    const match = ROOM_NAME_RE.exec(roomName);
    if (!match) {
      logger.error('Voice agent: room name does not match expected voice-<tenantId>-<sessionId> shape', {
        roomName,
      });
      return;
    }
    const [, tenantId, sessionId] = match;

    await ctx.connect();

    // The remote (visitor) participant's identity is "visitor-<visitorId>",
    // minted by getVoiceToken() — used as the same visitorId runLeadAgent()
    // needs to gate the public-widget lead-capture flow.
    //
    // A real, benign race found during production-readiness testing: if the
    // visitor disconnects (closes the tab, network drops) before this ever
    // resolves — e.g. a token was minted but the browser navigated away
    // before the room was actually joined — waitForParticipant() rejects
    // with "Room disconnected while waiting for participant". Previously
    // uncaught here, it propagated as an "error in entry function" the
    // framework's own outer supervisor logs and recovers from (confirmed:
    // the worker itself stays healthy, the job's process is eventually
    // reaped) — but it's a normal, expected outcome, not a real fault, so
    // it's handled explicitly here instead of relying on that outer catch.
    let participant;
    try {
      participant = await ctx.waitForParticipant();
    } catch (err) {
      logger.info('Voice agent: visitor disconnected before joining — no session started', {
        tenantId, sessionId, error: (err as Error).message,
      });
      return;
    }
    const visitorId = participant.identity.replace(/^visitor-/, '');

    const vad = ctx.proc.userData.vad as silero.VAD;

    // Fetched here (moved up from its previous post-session.start() spot,
    // where it was only used for the greeting/maxSessionMinutes) so the
    // tenant's own saved voice persona/language config — voicePreset,
    // voiceName, sttLanguage — can actually reach the STT/TTS construction
    // below. Previously these fields were saved via Widget Settings but
    // NEVER consumed by continuous voice at all — a real, separate gap from
    // the language-crash bug, confirmed by direct read of this exact
    // construction before this fix.
    const tenantCtx = await backendClient.getTenantContext(tenantId);
    const voiceCfg = tenantCtx?.widget?.voice;
    const normalizedSttLanguage = normalizeLanguageCode(voiceCfg?.sttLanguage);
    // Cartesia voice: prefer the structured voicePreset (set via the
    // Gender/Voice picker in Configuration Hub), then the free-text
    // voiceName override, then the plugin's own default.
    const cartesiaVoiceId = voiceCfg?.voicePreset?.voiceId || voiceCfg?.voiceName || undefined;
    const cartesiaLanguage = normalizeLanguageCode(voiceCfg?.voicePreset?.language) || normalizedSttLanguage || 'en';

    // Filled in right after the real AgentSession is constructed below —
    // LeadAgentLLM needs to exist before voice.Agent (which needs it before
    // AgentSession, which needs the agent) but its own pre-tool-call
    // acknowledgement filler needs session.say(). A mutable ref is the
    // simplest way to bridge that construction-order constraint.
    const sessionRef: SessionRef = { current: null };

    const agent = new voice.Agent({
      instructions:
        'You are a helpful voice assistant for this business. Keep replies concise and conversational — this is a spoken conversation, not a written one.',
      stt: new deepgram.STT({
        apiKey: process.env.DEEPGRAM_API_KEY,
        // Only auto-detect when the tenant hasn't set a language — passing a
        // bad/unnormalized value here was the live, deployed root cause of
        // "unsupported language: English" crashing push-to-talk's own STT
        // call; normalizeLanguageCode() defensively catches the identical
        // class of bad input here too before it ever reaches Deepgram.
        ...(normalizedSttLanguage ? { language: normalizedSttLanguage, detectLanguage: false } : { detectLanguage: true }),
      }),
      vad,
      llm: new LeadAgentLLM(tenantId, sessionId, visitorId, sessionRef),
      tts: new cartesia.TTS({
        apiKey: process.env.CARTESIA_API_KEY,
        ...(cartesiaVoiceId ? { voice: cartesiaVoiceId } : {}),
        language: cartesiaLanguage,
      }),
    });

    // turnHandling.preemptiveGeneration explicitly disabled — the framework's
    // own default (enabled, up to 3 retries) fires a real LeadAgentLLM.chat()
    // call — and therefore a real runLeadAgent() pipeline execution,
    // including a real backend getTenantContext() call — on every INTERIM
    // (non-final) STT transcript as it changes, not just the final one, plus
    // a possible extra call at confirmed turn-end. That's up to 4 real
    // pipeline executions for one spoken utterance, confirmed live by
    // tracing @livekit/agents' own agent_activity.js. Disabling it means
    // generateReply() fires exactly once, at confirmed end-of-turn — trading
    // away a latency optimization (start generating before the visitor
    // finishes talking) for the correctness guarantee that one spoken turn
    // is one backend operation. Revisit as a deliberate Phase B latency
    // optimization once the base (single-call) pipeline's real numbers are in.
    const session = new voice.AgentSession({
      turnHandling: { preemptiveGeneration: { enabled: false } },
    });
    sessionRef.current = session;

    const sessionStartedAt = Date.now();
    let usageReported = false;
    // Assigned after session.start() below, once the tenant's own
    // maxSessionMinutes config is known — declared here so the Close handler
    // (registered before that point) can still close over and clear it.
    let maxSessionTimer: NodeJS.Timeout | undefined;

    // Transient STT/TTS/LLM faults the framework may internally retry/recover
    // from — distinct from 'close', which only fires when the session
    // actually ends. Without this, such faults were previously invisible: no
    // log, no metric, nothing surfaced if e.g. Deepgram had a brief hiccup
    // mid-call. Routed through backendClient.writeLog (a durable sink) rather
    // than only worker stdout, since this job's own process — and its
    // console output — is ephemeral and vanishes with it.
    session.on(voice.AgentSessionEventTypes.Error, (ev) => {
      const inner = ev.error as { type?: string; label?: string; error?: Error; recoverable?: boolean } | undefined;
      logger.warn('Voice agent: session error event', {
        tenantId, sessionId, visitorId,
        errorType: inner?.type, label: inner?.label, message: inner?.error?.message, recoverable: inner?.recoverable,
      });
      void backendClient.writeLog({
        tenantId, sessionId,
        event: 'voice_agent.session_error',
        level: 'warn',
        message: `Continuous-voice session error: ${inner?.type ?? 'unknown'} (${inner?.label ?? 'unlabeled'})`,
        metadata: {
          errorType: inner?.type, label: inner?.label, message: inner?.error?.message, recoverable: inner?.recoverable, visitorId,
        },
      });
    });

    // ── §0 real latency instrumentation — measure, don't assert ──
    // The framework's own MetricsCollected event reports real per-stage
    // timing (Deepgram STT durationMs, Cartesia TTS ttfbMs/durationMs, and
    // the LLM adapter's own ttftMs/durationMs) as each stage actually
    // completes — reused rather than hand-threading timestamps through
    // LeadAgentLLM. Honestly labeled: LeadAgentLLMStream returns one full
    // text reply, not a token stream (see its own code comment), so
    // llmTtftMs from a non-streaming adapter naturally equals its total
    // durationMs — this is disclosed here, not presented as a true
    // first-token measurement. Logged per stage as it arrives (not joined
    // into one cross-stage "turn" record, which would need fragile
    // requestId/speechId correlation across three independent event
    // streams) via backendClient.writeLog, the same durable sink every
    // other voice-agent log already uses.
    session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev) => {
      const m = ev.metrics;
      if (m.type === 'stt_metrics') {
        void backendClient.writeLog({
          tenantId, sessionId, event: 'voice_agent.stage_timing', level: 'info',
          message: 'Continuous-voice STT stage timing',
          metadata: { stage: 'stt', durationMs: m.durationMs, audioDurationMs: m.audioDurationMs, visitorId },
        });
      } else if (m.type === 'llm_metrics') {
        void backendClient.writeLog({
          tenantId, sessionId, event: 'voice_agent.stage_timing', level: 'info',
          message: 'Continuous-voice LLM stage timing',
          metadata: {
            stage: 'llm',
            // Honestly labeled per the note above — not a real "first
            // token" measurement, since runLeadAgent() returns one full
            // response, not a stream.
            llmResponseReadyMs: m.ttftMs,
            durationMs: m.durationMs,
            visitorId,
          },
        });
      } else if (m.type === 'tts_metrics') {
        void backendClient.writeLog({
          tenantId, sessionId, event: 'voice_agent.stage_timing', level: 'info',
          message: 'Continuous-voice TTS stage timing',
          metadata: { stage: 'tts', firstAudioMs: m.ttfbMs, durationMs: m.durationMs, characters: m.charactersCount, visitorId },
        });
      }
    });

    // Real usage capture — session.usage.modelUsage carries provider-reported
    // numbers (Deepgram's actual audioDurationMs, Cartesia's actual
    // charactersCount), not client-side estimates. Reported once, on close,
    // guarded by usageReported since 'close' could in principle fire more
    // than once in edge cases (e.g. an error during shutdown) and this must
    // never double-count a session's usage.
    session.on(voice.AgentSessionEventTypes.Close, () => {
      if (maxSessionTimer) clearTimeout(maxSessionTimer);
      if (usageReported) return;
      usageReported = true;

      const minutes = (Date.now() - sessionStartedAt) / 60000;
      let deepgramSttSeconds = 0;
      let cartesiaTtsCharacters = 0;
      for (const u of session.usage.modelUsage) {
        if (u.type === 'stt_usage' && u.audioDurationMs) deepgramSttSeconds += u.audioDurationMs / 1000;
        if (u.type === 'tts_usage' && u.charactersCount) cartesiaTtsCharacters += u.charactersCount;
      }
      const estimatedCostUsd = estimateContinuousVoiceCostUsd({ minutes, deepgramSttSeconds, cartesiaTtsCharacters });

      void backendClient.reportContinuousVoiceUsage({
        tenantId, minutes, deepgramSttSeconds, cartesiaTtsCharacters, estimatedCostUsd,
      });
      logger.info('Voice agent: session closed, usage reported', {
        tenantId, sessionId, visitorId, minutes: minutes.toFixed(2), deepgramSttSeconds, cartesiaTtsCharacters, estimatedCostUsd,
      });
    });

    // record:false — no conversation-recording/storage feature is planned or
    // built for v1 (see the plan's own scope); this also sidesteps needing
    // AgentSession's local ffmpeg-based muxing pipeline, which was found to
    // crash on this Windows dev machine's bundled ffmpeg binary during
    // live-fire testing (a native-binary issue, unrelated to STT/TTS
    // themselves — Deepgram/Cartesia handle their own audio codecs directly
    // over their own APIs, no local transcoding needed for the live
    // conversation path).
    await session.start({ agent, room: ctx.room, record: false });

    // Diagnostic log for the ACTUAL voice used this call — closes the loop
    // on a real, confirmed-hard-to-diagnose bug (a tenant selected "Male"
    // in Configuration Hub but a call spoke in a female voice; every stage
    // of the save/persist/consume pipeline was independently confirmed
    // correct, leaving "was the worker actually running the code that reads
    // this" as the real, previously unanswerable question). A future report
    // can now be checked against a real log line instead of guessed at.
    logger.info('Voice agent: session started', {
      tenantId, sessionId, visitorId, room: ctx.room.name,
      cartesiaVoiceId: cartesiaVoiceId ?? '(plugin default)',
      voicePresetDisplayName: voiceCfg?.voicePreset?.displayName ?? null,
      voicePresetGender: voiceCfg?.voicePreset?.gender ?? null,
      cartesiaLanguage,
    });

    // ── AI speaks first — deterministic, not an LLM turn ──
    // session.say() synthesizes and speaks a fixed string directly via TTS,
    // bypassing LeadAgentLLM/runLeadAgent()/the tool loop entirely — unlike
    // session.generateReply(), which routes through that same LLM-driven
    // machinery every normal turn uses and could let the model decide to
    // call create_lead/book_meeting on this unprompted first turn. Matches
    // this codebase's own established "don't trust the LLM for something
    // that must be deterministic" pattern (fast-path, the
    // booking-confirmation-shortcut). Built dynamically per-tenant from
    // whatever site's actually crawled — see buildDynamicGreeting()'s own
    // comment. tenantCtx was already fetched above (needed earlier now, for
    // STT/TTS construction).
    const greeting = buildDynamicGreeting(tenantCtx);
    try {
      // allowInterruptions: false — a real, confirmed gap this closes: say()
      // is interruptible by default, and a visitor who starts talking the
      // instant the call visually connects (very natural — nothing in the
      // UI tells them setup is still finishing) can barge in and cut the
      // greeting off before it's ever heard, making their own first
      // utterance become the de-facto first turn instead. The greeting is a
      // single short deterministic line — there's nothing useful to
      // interrupt it FOR — so it always plays out in full; anything the
      // visitor says while it's playing is simply picked up as their real
      // first turn right after, not lost.
      await session.say(greeting, { allowInterruptions: false }).waitForPlayout();
    } catch (err) {
      // A visitor who disconnects mid-greeting (or a TTS hiccup) shouldn't
      // crash the whole job — the session's own Error/Close handlers above
      // already cover the real failure/cleanup paths.
      logger.warn('Voice agent: greeting playout did not complete', { tenantId, sessionId, error: (err as Error).message });
    }

    // ── Hard per-call duration cap — independent of the tenant's monthly
    // aggregate voice-minutes quota (checkTenantVoiceMinutesQuota, already
    // enforced per-turn inside runBaseAgent() for every channel). Protects
    // against one runaway/stuck call consuming a tenant's whole monthly
    // budget alone. Uses the same session.say() mechanism as the greeting —
    // a wrap-up notice is exactly the kind of fixed, must-not-trigger-a-tool
    // line that pattern exists for. ──
    const maxSessionMinutes = tenantCtx?.widget?.voice?.maxSessionMinutes;
    if (maxSessionMinutes && maxSessionMinutes > 0) {
      maxSessionTimer = setTimeout(() => {
        void (async () => {
          logger.info('Voice agent: max session duration reached, wrapping up', { tenantId, sessionId, maxSessionMinutes });
          try {
            await session.say(
              "We've reached the time limit for this call — thank you for chatting with us! Feel free to continue by typing, or start a new call.",
            ).waitForPlayout();
          } catch (err) {
            logger.warn('Voice agent: failed to speak max-session wrap-up', { error: (err as Error).message });
          }
          await session.close();
          // session.close() only tears down the internal STT/LLM/TTS
          // pipeline — confirmed directly in the installed package's own
          // source, it never touches the room connection at all. Without
          // this, the agent's own participant stays in the room
          // indefinitely (silent — no further replies), and the visitor's
          // browser never learns the call ended. Disconnecting the room
          // here makes the agent's own participant leave, which the
          // widget's client-side ParticipantDisconnected handler treats as
          // "the call is over" and hangs up on its own — the same way a
          // real phone call ends when the other party hangs up.
          try {
            await ctx.room.disconnect();
          } catch (err) {
            logger.warn('Voice agent: failed to disconnect room after max-session close', { error: (err as Error).message });
          }
        })();
      }, maxSessionMinutes * 60_000);
      maxSessionTimer.unref();
    }
  },
});

// ai/ compiles to CommonJS (see tsconfig.json), not ESM — import.meta isn't
// valid here, unlike the framework's own ESM-flavored examples. require.main
// is the CommonJS-native equivalent of "was this file run directly," and
// __filename is what dynamically-imported-per-job worker processes need to
// re-load this same compiled file.
if (require.main === module) {
  void startHeartbeat();
  // numIdleProcesses: the framework's own default is 0 in dev mode (only
  // >0 in production) — meaning every single call cold-starts a brand new
  // child process from scratch (Node boot + re-loading every import: the
  // full LangChain/OpenAI/Anthropic SDK surface, config, etc.), a real,
  // confirmed multi-second contributor to the reported 15-20s delay before
  // the AI ever speaks. Keeping 1 process pre-warmed and idle means a call
  // can be dispatched to an already-booted process immediately instead of
  // paying that cold-start tax on every single call.
  cli.runApp(new WorkerOptions({ agent: __filename, numIdleProcesses: 1 }));
}
