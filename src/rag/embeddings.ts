import axios from 'axios';
import { OpenAIEmbeddings } from '@langchain/openai';
import { config } from '../config';
import { logger } from '../utils/logger';

export const VECTOR_SIZE = config.embeddings.provider === 'voyage' ? 1024 : 1536;

async function voyageEmbed(texts: string[], inputType: 'document' | 'query'): Promise<number[][]> {
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
