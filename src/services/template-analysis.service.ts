import { GoogleGenAI, Type, Schema, Part } from '@google/genai';
import { jsonrepair } from 'jsonrepair';
import { config } from '../config';
import { logger } from '../utils/logger';
import {
  buildTemplateAnalysisPrompt, buildStructuredAnalysisPrompt,
  CatalogGroup, PdfStructure,
} from '../prompts/template-analysis.prompt';

export interface TemplateAnalysisInput {
  docType: string;
  // Vision path: both present, no structuredContent.
  mimetype?: string;
  fileBase64?: string;
  // Real-PDF path: backend already extracted ground-truth text positions —
  // no image/vision call needed at all for this case.
  structuredContent?: PdfStructure;
  variableCatalog: CatalogGroup[];
}

export interface TemplateAnalysisResult {
  elements: unknown[];
  warnings: string[];
}

// A one-shot structured-extraction call, not part of the conversational/RAG
// agent flow this service otherwise runs — calls @google/genai directly
// rather than going through llm.provider.ts's LangChain abstraction, which
// only supports plain-text messages (no document/image content parts).
// Gemini was chosen over Claude here because it natively accepts both PDF
// and image input (inlineData) the same way Claude does, with a free tier —
// this feature was originally built against Anthropic, but switched to
// Gemini after hitting a billing/credit issue on the Anthropic account.
let client: GoogleGenAI | null = null;
function getClient(): GoogleGenAI {
  if (!client) client = new GoogleGenAI({ apiKey: config.google.apiKey });
  return client;
}

// The backend's repairAndValidateElements pipeline is the real strictness
// gate against the exact designElementSchema bounds (numeric clamping,
// enum coercion, etc.) — this schema doesn't need to mirror every bound.
// But it DOES need to list every field name the model might want to use:
// verified live that Gemini's structured-output mode only ever emits
// properties actually declared in the schema — an earlier, narrower version
// of this schema (just id/type/x/y/w/h) came back with well-positioned but
// completely EMPTY elements every time (no content, no table columns, no
// totals rows, nothing), because there was nowhere in the schema for the
// model to put that data. This full field list fixes that; the earlier
// narrower version's job (never emitting bare `{}` stubs) is preserved by
// still requiring the core positional fields.
//
// All numeric fields are Type.INTEGER, not NUMBER — verified live that
// NUMBER lets the model spiral into a degenerate floating-point
// serialization bug (a real response returned "h":
// 40.50632911392405461159333903823...  followed by literally thousands of
// trailing zero digits for a single field, burning the entire token budget
// on garbage precision). Every numeric field in this schema is a whole
// pixel/point value in practice anyway (the repair pipeline rounds/clamps
// regardless), so INTEGER removes any reason to emit a fractional value.
const RESPONSE_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    elements: {
      type: Type.ARRAY,
      description: 'Every layout element found in the document, in reading order (top to bottom).',
      items: {
        type: Type.OBJECT,
        properties: {
          id:   { type: Type.STRING },
          type: { type: Type.STRING },
          x:    { type: Type.INTEGER },
          y:    { type: Type.INTEGER },
          w:    { type: Type.INTEGER },
          h:    { type: Type.INTEGER },
          // typography (text / richtext / table / totals)
          content:    { type: Type.STRING },
          fontSize:   { type: Type.INTEGER },
          fontFamily: { type: Type.STRING },
          fontWeight: { type: Type.STRING },
          fontStyle:  { type: Type.STRING },
          color:      { type: Type.STRING },
          textAlign:  { type: Type.STRING },
          padding:    { type: Type.INTEGER },
          // image
          src:       { type: Type.STRING },
          objectFit: { type: Type.STRING },
          // box / divider borders
          backgroundColor: { type: Type.STRING },
          borderColor:     { type: Type.STRING },
          borderWidth:     { type: Type.INTEGER },
          borderRadius:    { type: Type.INTEGER },
          // table
          dataset: { type: Type.STRING },
          columns: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                key: { type: Type.STRING }, label: { type: Type.STRING },
                width: { type: Type.INTEGER }, align: { type: Type.STRING },
              },
              required: ['key', 'label'],
            },
          },
          headerBg:    { type: Type.STRING },
          headerColor: { type: Type.STRING },
          altRowBg:    { type: Type.STRING },
          showBorders: { type: Type.BOOLEAN },
          // totals
          totalsRows: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: { key: { type: Type.STRING }, label: { type: Type.STRING } },
              required: ['key'],
            },
          },
          totalsEmphasizeLast: { type: Type.BOOLEAN },
          // gridtable
          gridRows: { type: Type.INTEGER },
          gridCols: { type: Type.INTEGER },
          gridCells: {
            type: Type.ARRAY,
            items: { type: Type.ARRAY, items: { type: Type.STRING } },
          },
          gridHeaderRow: { type: Type.BOOLEAN },
        },
        required: ['id', 'type', 'x', 'y', 'w', 'h'],
      },
    },
  },
  required: ['elements'],
};

/** Cheap page-count guardrail for PDFs, before the paid Gemini call. Text-only extraction — no layout/vision here. Only relevant to the vision path; the structured path never sends the document to Gemini as an image, so a page-count-driven cost blowup there isn't a concern the same way. */
async function assertPdfPageLimit(fileBase64: string): Promise<void> {
  try {
    const pdfParseModule = await import('pdf-parse');
    const buffer = Buffer.from(fileBase64, 'base64');
    const parsed = await pdfParseModule.default(buffer);
    if (parsed.numpages > config.templateAnalyzer.maxPdfPages) {
      throw new Error(
        `This PDF has ${parsed.numpages} pages — the analyzer supports up to ${config.templateAnalyzer.maxPdfPages}. Try a shorter document.`
      );
    }
  } catch (err: any) {
    if (err.message?.includes('the analyzer supports up to')) throw err;
    // pdf-parse failing to read metadata shouldn't block analysis outright —
    // Gemini will still attempt the document; this check is a cost
    // guardrail, not a correctness gate.
    logger.warn('PDF page-count pre-check failed, proceeding anyway', { error: err.message });
  }
}

/** Shared call + parse/repair logic for both the vision and structured-content paths. */
async function callGeminiAndParse(parts: Part[]): Promise<TemplateAnalysisResult> {
  const response = await getClient().models.generateContent({
    model: config.templateAnalyzer.model,
    contents: [{ role: 'user', parts }],
    config: {
      maxOutputTokens: config.templateAnalyzer.maxTokens,
      responseMimeType: 'application/json',
      responseSchema: RESPONSE_SCHEMA,
      // Verified live (not assumed): with the default automatic thinking
      // budget, this model spends most of its output budget "thinking" and
      // can silently return an incomplete elements[] array (still valid
      // JSON, just missing most of the layout) — a 25-element test document
      // came back with only 1 element. Disabling thinking entirely put the
      // full token budget toward the actual JSON and returned all 25. This
      // task needs a direct structured answer, not exposed reasoning.
      thinkingConfig: { thinkingBudget: 0 },
      // Default temperature is 1 — this is a deterministic structured-
      // extraction task (echo/label real data, not creative generation), and
      // a low temperature reduces exactly the kind of rare stochastic
      // degenerate output that caused the number-serialization bug above.
      temperature: 0.2,
    },
  });

  const finishReason = response.candidates?.[0]?.finishReason;
  const text = response.text;
  if (!text) {
    logger.error('Template analysis: empty response from model', { finishReason, usage: response.usageMetadata });
    throw new Error('The analyzer did not return a structured result — please try again.');
  }

  const warnings: string[] = [];
  let parsed: { elements?: unknown[] };
  try {
    parsed = JSON.parse(text);
  } catch (err: any) {
    // Verified live: a real dense document can still hit maxOutputTokens
    // mid-generation (finishReason 'MAX_TOKENS'), leaving an unterminated
    // JSON string. Rather than fail the whole analysis over a cut-off tail,
    // repair the truncated structure and keep whatever elements completed
    // before the cutoff — the backend's own repair pipeline still discards
    // any resulting incomplete/empty elements, so this only ever adds
    // usable content, never lets junk through.
    try {
      parsed = JSON.parse(jsonrepair(text));
      if (finishReason === 'MAX_TOKENS') {
        warnings.push('The document was complex enough that the analysis was cut off partway through — some elements near the end may be missing.');
      }
      logger.warn('Template analysis: JSON.parse failed, salvaged via jsonrepair', { finishReason, textLength: text.length });
    } catch (repairErr: any) {
      // Log the raw output server-side (not shown to the user) so a future
      // failure here is actually debuggable instead of a dead-end guess.
      logger.error('Template analysis: model output failed JSON.parse and jsonrepair', {
        finishReason, usage: response.usageMetadata, error: err.message, repairError: repairErr.message,
        textPreview: text.slice(0, 500), textLength: text.length,
      });
      throw new Error('The analyzer returned malformed output — please try again.');
    }
  }

  if (!Array.isArray(parsed.elements)) {
    throw new Error('The analyzer returned no layout elements.');
  }

  return { elements: parsed.elements, warnings };
}

export async function analyzeTemplateDocument(input: TemplateAnalysisInput): Promise<TemplateAnalysisResult> {
  if (input.structuredContent) {
    // Real digital PDF — the backend already extracted ground-truth text
    // positions directly from the file's own data. Text-only call: no
    // image/vision input needed (cheaper too, no image tokens).
    const prompt = buildStructuredAnalysisPrompt(input.docType, input.variableCatalog, input.structuredContent);
    return callGeminiAndParse([{ text: prompt }]);
  }

  if (!input.mimetype || !input.fileBase64) {
    throw new Error('No document content provided for analysis.');
  }

  // Vision fallback — scanned/photographed documents (no extractable text
  // layer) and plain images always take this path; a real PDF only falls
  // back here if extraction found no meaningful text.
  if (input.mimetype === 'application/pdf') {
    await assertPdfPageLimit(input.fileBase64);
  }
  const prompt = buildTemplateAnalysisPrompt(input.docType, input.variableCatalog);
  return callGeminiAndParse([
    { inlineData: { data: input.fileBase64, mimeType: input.mimetype as any } },
    { text: prompt },
  ]);
}
