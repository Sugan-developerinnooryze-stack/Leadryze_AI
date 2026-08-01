// Groq's small model occasionally emits the literal string "null" (or
// "undefined"/"n/a"/"none") for a field it doesn't actually know, instead of
// omitting the key or using real JSON null — confirmed live via testing.
// Zod's z.string().optional()/.nullable() happily accepts that as a real,
// truthy string, which would otherwise pass an "is this field present" check
// with garbage data (e.g. a Lead saved with firstName: "null"). Shared by
// every place that reads an LLM-extracted string field before using it.
export function cleanArg(value: string | null | undefined): string | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  return /^(null|undefined|n\/a|none)$/i.test(trimmed) ? undefined : trimmed;
}
