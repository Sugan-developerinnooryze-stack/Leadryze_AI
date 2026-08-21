/** A single item card built from a real search_dataset tool result — never
 * from the LLM's own text, so a "Request Quote" click always traces back to
 * a genuine DatasetRecord. Generic naming (not "Product") since Business
 * Knowledge carries services/machines/courses/properties, not only
 * products; the widget UI can still label these "product cards" visually
 * where that reads naturally. See search-dataset.tool.ts's
 * resolveDisplayTitle() for how `title` is guaranteed non-empty even when a
 * dataset has no name/title column at all. */
export interface DatasetItemCard {
  datasetId: string;
  datasetName: string;
  /** Which Dataset version this card's data was actually read from — a
   * re-upload can change price/specs under the same datasetId+recordId, so
   * downstream Lead references need to say WHICH version the visitor saw. */
  datasetVersion: number;
  recordId: string;
  title: string;
  /** Exact raw display string from the source cell (e.g. "₹45,000–₹60,000"
   * or "On Request") — never reformatted/simplified into a single number. */
  price?: string;
  /** Validated http(s):// only — see search-dataset.tool.ts's
   * isSafeImageUrl(). */
  imageUrl?: string;
  keySpecs?: string[];
}
