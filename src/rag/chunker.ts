import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

export interface Chunk {
  content: string;
  metadata: Record<string, unknown>;
}

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
  separators: ['\n\n', '\n', '. ', ' ', ''],
});

export async function chunkText(
  text: string,
  metadata: Record<string, unknown> = {}
): Promise<Chunk[]> {
  const docs = await splitter.createDocuments([text], [metadata]);
  return docs.map((doc: { pageContent: string; metadata: Record<string, unknown> }) => ({
    content: doc.pageContent,
    metadata: doc.metadata,
  }));
}
