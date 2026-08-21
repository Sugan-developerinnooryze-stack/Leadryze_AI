import { z } from 'zod';
import { AgentTool } from './tool.types';
import { retrieveContext } from '../rag/pipeline';
import { screenFieldForInjection } from '../utils/redact-injection';

const schema = z.object({
  query: z.string().min(1).describe('A focused rephrasing of what the visitor wants to know, used to search the tenant\'s own website content'),
});

export const searchWebsiteKnowledgeTool: AgentTool<z.infer<typeof schema>> = {
  name: 'search_website_knowledge',
  description:
    "Search this business's own website content (products, services, pricing, FAQs, policies) for an answer. " +
    'Call this when the visitor asks something specific that the context already provided doesn\'t answer, ' +
    'or when they rephrase/narrow a question you couldn\'t answer confidently before.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const results = await retrieveContext(args.query, ctx.tenantId, 5);
    if (!results.length) {
      return { ok: true, summary: 'No matching website content found for this query.' };
    }
    // Hardening Gap 7 — crawled page content is externally-controlled text,
    // screened per-passage before it reaches the model.
    const block = results
      .map((r, i) => `[${i + 1} — ${r.source}] ${screenFieldForInjection(r.source, r.content)}`)
      .join('\n\n');
    return {
      ok: true,
      summary: `Found ${results.length} relevant passage(s) from the website.`,
      data: { passages: block, topScore: results[0].score },
    };
  },
};
