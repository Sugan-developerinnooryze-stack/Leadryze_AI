import { z } from 'zod';
import { AgentTool } from './tool.types';
import { backendClient } from '../services/backend.client';
import { cleanArg } from '../utils/clean-arg';
import { screenFieldForInjection } from '../utils/redact-injection';

const schema = z.object({
  recordId: z.string().min(1).describe(
    'The exact "id" of one of the records just found via search_dataset — never invent one yourself.',
  ),
  datasetName: z.string().nullish().describe(
    'The dataset name this record came from, ONLY if the tenant has more than one dataset — omit when there is only one.',
  ),
});

export const getDatasetRecordTool: AgentTool<z.infer<typeof schema>> = {
  name: 'get_dataset_record',
  description:
    'Get the full stored data for one specific record from a business dataset, by the "id" search_dataset returned. ' +
    'Always call search_dataset first to find the right id — never guess one.',
  schema,
  surfaces: ['public_widget'],
  async execute(args, ctx) {
    const datasets = await backendClient.listDatasetsForChatbot(ctx.tenantId);
    if (!datasets.length) {
      return { ok: false, summary: 'No business datasets are configured for this tenant.' };
    }
    const nameHint = cleanArg(args.datasetName)?.toLowerCase();
    const target =
      (nameHint ? datasets.find((d) => d.name.toLowerCase().includes(nameHint)) : undefined) ??
      datasets[0];

    const record = await backendClient.getDatasetRecord(ctx.tenantId, target.datasetId, args.recordId);
    if (!record) {
      return { ok: false, summary: `No record found with id "${args.recordId}" in ${target.name} — call search_dataset again to find the correct id.` };
    }
    // Relabel sanitized normalizedName keys back to the real header text
    // (hardening Gap 5), and screen each value for injection content
    // (hardening Gap 7) before it reaches the model.
    const columns = await backendClient.getDatasetSchema(ctx.tenantId, target.datasetId);
    const labelMap = backendClient.buildDatasetLabelMap(columns);
    const relabeled: Record<string, unknown> = {};
    const fieldParts: string[] = [];
    for (const [k, v] of Object.entries(record.data)) {
      if (v === undefined || v === null || v === '') continue;
      const label = labelMap.get(k) ?? k;
      const safeValue = screenFieldForInjection(label, String(v), { recordId: record.recordId });
      relabeled[label] = safeValue;
      fieldParts.push(`${label}: ${safeValue}`);
    }
    return {
      ok: true,
      summary: `According to ${record.datasetName}: ${fieldParts.join('; ')}`,
      data: { datasetName: record.datasetName, ...relabeled },
    };
  },
};
