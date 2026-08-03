import { z } from 'zod';

/** Structured lead-field extraction for the AI website widget's own
 * conversation flow. Uses `.nullable()` throughout, matching this codebase's
 * own existing convention for LLM-extracted fields (see base.agent.ts's own
 * ChatIntentSchema.entities) rather than `.optional()` — an LLM asked to
 * fill a structured schema needs an explicit "I don't know this" value, not
 * an omittable key. */
export const LeadFieldExtractionSchema = z.object({
  firstName: z.string().nullable(),
  lastName:  z.string().nullable(),
  email:     z.string().nullable(),
  phone:     z.string().nullable(),
  company:   z.string().nullable(),
  /** What they're contacting about / interested in — maps onto
   * Lead.interestedServices on the backend. */
  service:   z.string().nullable(),
  confidence: z.number().min(0).max(1),
});

export type LeadFieldExtraction = z.infer<typeof LeadFieldExtractionSchema>;
