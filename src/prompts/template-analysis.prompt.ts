export interface CatalogGroup {
  label: string;
  items: { key: string; label: string; elemType: string }[];
}

export interface PdfTextRun {
  text:     string;
  x:        number;
  y:        number;
  fontSize: number;
  fontName: string;
}

export interface PdfStructure {
  textRuns:   PdfTextRun[];
  imageCount: number;
}

// Shared between both prompt variants — the target element schema and the
// per-docType {{token}} catalog don't change based on how the document's
// content was obtained (vision vs. real extracted text).
function buildSharedSchemaAndCatalog(docType: string, variableCatalog: CatalogGroup[]): string {
  const tokenList = variableCatalog
    .map((g) => `${g.label}:\n${g.items.map((i) => `  {{${i.key}}} — ${i.label} (${i.elemType})`).join('\n')}`)
    .join('\n\n');

  return `## Canvas
The canvas is an A4 page, 794×1123px at 96dpi, origin top-left. Position every element with absolute x/y (px from top-left of the page) and w/h (px). If the source document is genuinely multi-page or a services/parts table would run long, that's fine — a 'table'/'richtext'/'gridtable' element's height is just a rough placeholder anyway; real content flows past it at render time. Keep single-page documents within roughly 0-794 horizontally.

**Critical layout rule**: this system only supports a single-column, vertically-stacked layout AROUND 'table' elements (the dynamic services/parts line-items table) — it has no concept of something sitting BESIDE one (same vertical range, different horizontal position). Every other element's [y, y+h) range must NOT overlap any 'table' element's [y, y+h) range at all. If the source document visually places something next to the line-items table (a common pattern: a totals summary in the upper-right corner beside where the table begins), you must instead position that element so it's entirely ABOVE the table's y (if it belongs before/beside the table's start) or entirely BELOW the table's y+h (if it belongs after/beside the table's later rows) — pick whichever keeps it closest to its real visual context, but never let the ranges overlap. 'richtext' and 'gridtable' elements do NOT have this restriction — they render as fixed-height boxes exactly where you place them, so it's fine for another element to sit beside one of those.

## Element types you can emit (exact shape required)
Every element needs: id (short unique string, e.g. "el1"), type, x, y, w, h.

- **text** — static or token-bound label. Fields: content (string — put a literal {{token}} here if this text should show live data, e.g. "{{company.name}}"), fontSize (6-120), fontFamily (one of: Arial, Helvetica, Georgia, "Times New Roman", "Courier New", Verdana, Tahoma, "Trebuchet MS"), fontWeight ("normal"|"bold"), fontStyle ("normal"|"italic"), color (hex like "#111827"), textAlign ("left"|"center"|"right"), padding (0-100).
- **richtext** — a larger block of formatted text (terms/notes/multi-line paragraphs). Same typography fields as text. Set content to "{{doc.notes}}" or "{{doc.terms}}" if this block looks like a notes/terms section, otherwise the literal text. **At most ONE richtext element may bind to {{doc.notes}}, and at most ONE to {{doc.terms}}, in the entire document** — a document typically has zero or one genuine notes/terms block; a signature line, a closing "Thank you" message, or any other distinct text section is NEVER the same block and must be its own separate 'text' element with literal content, never bound to the same token.
- **image** — a logo/signature/stamp/QR/photo. Fields: src (a {{token}} like "{{company.logo}}" if it matches a recognized image token below, otherwise omit), objectFit ("contain"|"cover"|"fill").
- **table** — a dynamic line-items table (services or parts, one row per line item — NOT a manually laid-out grid). Fields: dataset ("services"|"parts"), columns (array of up to 12 {key, label, width?, align?} — key MUST be one of: index, name, description, count, amount, lineTotal, partNumber — partNumber only valid when dataset is "parts"), headerBg, headerColor, altRowBg (hex or ""), showBorders (boolean).
- **totals** — a subtotal/discount/tax/total summary block. Fields: totalsRows (array of up to 8 {key, label?} — key MUST be one of: servicesSubtotal, partsSubtotal, subtotal, discount, gst, total, paid, balance), totalsEmphasizeLast (boolean).
- **divider** — a plain horizontal line. Fields: borderColor, borderWidth (0-30).
- **box** — a background/border rectangle (used for bordered sections like "Bill To"). Fields: backgroundColor, borderColor, borderWidth, borderRadius.
- **gridtable** — a manually-authored grid (Word/Excel-style — fixed rows/cols, NOT bound to real line-item data). Use this ONLY for things that are clearly a hand-drawn grid/matrix in the source, not a services/parts table. Fields: gridRows (1-30), gridCols (1-12), gridCells (array of gridRows arrays, each with EXACTLY gridCols string entries — plain text or simple HTML like <b>bold</b>, can include a {{token}}), gridHeaderRow (boolean).

## Available {{token}} bindings for this docType (use these, don't invent new ones)
${tokenList}

If a recognized field in the document doesn't match any token above (e.g. a label with no real data behind it), just use literal static text instead of inventing a token.`;
}

/**
 * Vision-based prompt — the document is attached directly as an image/PDF
 * and the model has to visually determine both WHERE things are and WHAT
 * they mean. This is the fallback path for scanned/photographed documents
 * with no extractable text layer (see buildStructuredAnalysisPrompt for the
 * more accurate path used when real text-position data is available).
 */
export function buildTemplateAnalysisPrompt(docType: string, variableCatalog: CatalogGroup[]): string {
  return `You are analyzing an existing ${docType} document (PDF or image) so it can be recreated as an editable drag-and-drop template in a document design tool. Look at the uploaded document and respond with JSON matching the required schema: an "elements" array of layout elements that reproduces its structure as closely as possible.

${buildSharedSchemaAndCatalog(docType, variableCatalog)}

## Instructions
1. Look at the whole document first — identify the header/logo area, customer/billing section, the line-items table, totals, notes/terms, footer, bank details, signature — before emitting elements.
2. Map every recognized field to its matching {{token}} above. Only fall back to literal text when nothing matches.
3. The real line-items table (however it's drawn — bordered, borderless, striped) should become exactly one 'table' element with dataset set correctly, not a 'gridtable'.
4. Keep position/size numbers realistic and proportioned relative to the 794×1123 canvas.
5. Enforce the critical layout rule above: nothing else may share a vertical range with a 'table' element, even if it visually sits beside one in the source (a common case: a totals summary in a corner next to the line-items table) — move it clear, above or below. 'richtext'/'gridtable' elements have no such restriction.
6. Return every element in a single "elements" array, in reading order (top to bottom).`;
}

/**
 * Structured-content prompt — used when the source is a real digital PDF
 * (not a scan): the backend already extracted every text run's EXACT
 * position/font-size directly from the PDF's own embedded data via
 * pdfjs-dist, so the model is doing pure semantic labeling here, not
 * guessing positions. This is the fix for the accuracy ceiling of pure
 * vision analysis — real PDFs already contain ground-truth layout data,
 * reading it directly beats asking an AI to visually reconstruct it.
 */
export function buildStructuredAnalysisPrompt(docType: string, variableCatalog: CatalogGroup[], structure: PdfStructure): string {
  // Reading order (top-to-bottom, then left-to-right) makes row/column
  // grouping far easier for the model than the PDF's own internal order,
  // which often interleaves unrelated regions.
  const sortedRuns = [...structure.textRuns].sort((a, b) => (a.y - b.y) || (a.x - b.x));
  const runsList = sortedRuns.map((r) => `[x=${r.x}, y=${r.y}, fontSize=${r.fontSize}] "${r.text}"`).join('\n');

  return `You are analyzing an existing ${docType} document (a real digital PDF) so it can be recreated as an editable drag-and-drop template in a document design tool.

The exact text content of this document has already been extracted directly from the PDF's own internal data — every position and font size below is GROUND TRUTH, not a guess. **Use these x/y/fontSize values directly for each element you emit — do not re-estimate or adjust them** — with exactly one exception: the critical layout rule below (no element may share a vertical range with the 'table' element) overrides the raw extracted y when the two conflict, since the real document may have things positioned beside the line-items table in a way this system can't reproduce directly. Your job is otherwise purely semantic: figure out what each piece of text represents, group related runs into the right element type, and map recognized fields to the correct {{token}}.

## Extracted text runs (reading order, top to bottom)
${runsList}

## Images
This document also contains ${structure.imageCount} image(s) (logo, signature, stamp, QR, etc.) whose exact position wasn't extracted — place 'image' elements using standard document conventions (a logo is normally top-left or top-right near the company name; a signature/stamp is normally near the bottom) and bind each to the closest matching {{token}} below if it fits, otherwise omit src.

${buildSharedSchemaAndCatalog(docType, variableCatalog)}

## Instructions
1. Group text runs that are visually one unit (e.g. a multi-line address, several stacked lines of a paragraph) into a single element's content, using line breaks as needed — don't emit one 'text' element per single word or line unless they're genuinely visually separate.
2. Group a repeating grid of runs (a header row of column names followed by aligned rows of values) into exactly ONE 'table' element with matching columns, not one element per cell.
3. Every extracted run should be accounted for by some element — don't silently drop content, but also don't fabricate content that isn't in the list above.
4. Map every recognized field to its matching {{token}} above; only fall back to literal text when nothing matches.
5. Enforce the critical layout rule above: if a run's real extracted y would place it inside the 'table' element's vertical range (e.g. a totals summary extracted at a y that falls within the services table's rows, because in the source it actually sits beside the table, not below it), move that element's y so it's entirely above or entirely below the table instead — do not leave it at its raw extracted position in that case. 'richtext'/'gridtable' elements have no such restriction; leave their extracted y as-is even if another element sits beside them.
6. Return every element in a single "elements" array, in the same top-to-bottom reading order as the extracted text runs.`;
}
