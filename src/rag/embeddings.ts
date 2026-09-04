import axios from 'axios';
import { OpenAIEmbeddings } from '@langchain/openai';
import { config } from '../config';
import { logger } from '../utils/logger';

export const VECTOR_SIZE = config.embeddings.provider === 'voyage' ? 1024 : 1536;

// Real, confirmed live failure this closes: a burst of embedding calls
// (dataset import batches, the query-router's semantic fallback, and real
// visitor traffic all hit the same Voyage account) can trip Voyage's
// per-minute rate limit for a single request even while the account is
// otherwise healthy — confirmed by the SAME call succeeding a few seconds
// later with no code change. Previously this had zero retry: one 429 during
// a chat turn killed search_dataset's tool call outright, which then made
// response-confidence.ts correctly-but-unnecessarily downgrade an otherwise
// answerable question to the low-confidence handoff message. Retrying only
// on 429/5xx (never on a real 4xx like bad input or an invalid key, which
// retrying would never fix) with a short, fixed backoff is enough — the
// underlying bursts clear in well under a second in practice.
const EMBED_MAX_ATTEMPTS = 3;
const EMBED_RETRY_DELAYS_MS = [500, 1500];

function isRetryableEmbedError(err: unknown): boolean {
  const status = (err as { response?: { status?: number } })?.response?.status;
  return status === 429 || (typeof status === 'number' && status >= 500);
}

async function voyageEmbed(texts: string[], inputType: 'document' | 'query'): Promise<number[][]> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= EMBED_MAX_ATTEMPTS; attempt++) {
    try {
      const response = await axios.post(
        'https://api.voyageai.com/v1/embeddings',
        { model: config.embeddings.model, input: texts, input_type: inputType },
        {
          headers: {
            Authorization: `Bearer ${config.voyage.apiKey}`,
            'Content-Type': 'application/json',
          },
          timeout: 30000,
        }
      );
      return (response.data.data as { embedding: number[] }[]).map((d) => d.embedding);
    } catch (err) {
      lastErr = err;
      if (attempt === EMBED_MAX_ATTEMPTS || !isRetryableEmbedError(err)) throw err;
      logger.warn('Voyage embeddings call failed — retrying', {
        attempt, status: (err as { response?: { status?: number } })?.response?.status,
      });
      await new Promise((resolve) => setTimeout(resolve, EMBED_RETRY_DELAYS_MS[attempt - 1]));
    }
  }
  throw lastErr;
}

let openaiModel: OpenAIEmbeddings | null = null;
function getOpenAIModel(): OpenAIEmbeddings {
  if (!openaiModel) {
    openaiModel = new OpenAIEmbeddings({
      model: config.embeddings.model,
      apiKey: config.openai.apiKey,
      batchSize: 512,
    });
    logger.info('OpenAI embeddings initialized', { model: config.embeddings.model });
  }
  return openaiModel;
}

export async function embedText(text: string): Promise<number[]> {
  if (config.embeddings.provider === 'voyage') {
    const [embedding] = await voyageEmbed([text], 'document');
    return embedding;
  }
  const [embedding] = await getOpenAIModel().embedDocuments([text]);
  return embedding;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (config.embeddings.provider === 'voyage') {
    return voyageEmbed(texts, 'document');
  }
  return getOpenAIModel().embedDocuments(texts);
}

export async function embedQuery(query: string): Promise<number[]> {
  if (config.embeddings.provider === 'voyage') {
    const [embedding] = await voyageEmbed([query], 'query');
    return embedding;
  }
  return getOpenAIModel().embedQuery(query);
}
