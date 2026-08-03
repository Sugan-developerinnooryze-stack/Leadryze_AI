import { backendClient, TenantContext, InlineRecord, WebsiteProfileSummary } from './backend.client';
import { logger } from '../utils/logger';

export interface ResolvedTenantConfig {
  companyName: string;
  agentName: string;
  language: string;
  systemPrompt?: string;
  fallbackToHuman: boolean;
  crmContext: string;
  customerContext: string;
  inlineRecordsBlock: string;  // Formatted actual records for small modules — always injected
  hasConnectors: boolean;
  connectorTypes: string[];
  crmModules: Record<string, Array<{ module: string; count: number }>>;
  qnaPairs: Array<{ question: string; answer: string; category: string }>;
  templates: Array<{ name: string; type: string; category: string; subject?: string; body: string; variables: string[] }>;
  /** Built once per crawl (ai/src/rag/website-profile-extractor.ts) — null
   * for a tenant that's never crawled their site. */
  websiteProfile: WebsiteProfileSummary | null;
  /** Monthly LLM token budget for the public widget only (never applies to
   * the internal, staff-authenticated assistant — see isPublicVisitor in
   * base.agent.ts). Always a real number — a plan-tier default when the
   * tenant has no explicit aiConfig.monthlyTokenLimit override. */
  monthlyTokenLimit: number;
  /** Which already-integrated LLM provider/model powers RAG/catalog/booking
   * tool-calling for this tenant — undefined means "use the global primary/
   * fallback pair", today's unchanged default. */
  toolModelPreset?: 'groq' | 'anthropic' | 'openai' | 'google';
  /** True when this tenant has at least one active Team marked
   * showInWidget:true — gates whether the booking prompt ever mentions
   * departments/doctors at all. False for every tenant that hasn't opted
   * into the department/doctor wizard, so today's tenant-wide booking flow
   * stays completely unaffected. */
  hasWidgetDepartments: boolean;
}

/** Plan-tier default monthly token budgets — deliberately generous
 * starting points, not a hard business decision; a Super Admin can override
 * per-tenant via aiConfig.monthlyTokenLimit for a custom deal. */
const DEFAULT_MONTHLY_TOKEN_LIMITS: Record<string, number> = {
  starter: 300_000,
  professional: 1_500_000,
  enterprise: 8_000_000,
};

/**
 * Build a formatted CRM overview block for injection into the system prompt.
 * Shows the AI what data sources the tenant has and how many records per module.
 */
function formatCRMContext(ctx: TenantContext): string {
  const { connectors, crmModules, recentCustomers } = ctx;

  if (!connectors.length) return '';

  const lines: string[] = ['=== TENANT CRM DATA OVERVIEW ==='];

  // Active connectors
  const activeConnectors = connectors.filter((c) => c.isActive);
  if (activeConnectors.length) {
    lines.push(`\nACTIVE DATA CONNECTORS: ${activeConnectors.map((c) => `${c.name} (${c.type})`).join(', ')}`);
  }

  // CRM modules summary
  if (Object.keys(crmModules).length) {
    lines.push('\nCRM MODULES & RECORD COUNTS:');
    for (const [channel, modules] of Object.entries(crmModules)) {
      lines.push(`  ${channel.toUpperCase()}:`);
      for (const { module, count } of modules.slice(0, 10)) {
        lines.push(`    - ${module}: ${count} records`);
      }
    }
  }

  // Customer pipeline snapshot
  if (recentCustomers.length) {
    const byStatus: Record<string, number> = {};
    for (const c of recentCustomers) {
      byStatus[c.status] = (byStatus[c.status] || 0) + 1;
    }
    lines.push('\nRECENT LEAD PIPELINE (last 20):');
    for (const [status, count] of Object.entries(byStatus)) {
      lines.push(`    - ${status}: ${count}`);
    }

    // List names for personalisation
    const named = recentCustomers.filter((c) => c.name).slice(0, 5);
    if (named.length) {
      lines.push(`\nRECENT CUSTOMERS: ${named.map((c) => c.name).join(', ')}`);
    }
  }

  lines.push('\nUse the above data to answer questions about the company\'s leads, accounts, and CRM records accurately. Do NOT fabricate numbers not listed above.');
  lines.push('=================================');

  return lines.join('\n');
}

/**
 * Format recent customers as a short context block for the LLM.
 */
function formatCustomerContext(ctx: TenantContext): string {
  const { recentCustomers } = ctx;
  if (!recentCustomers.length) return '';

  const lines = ['=== RECENT LEADS/CUSTOMERS ==='];
  for (const c of recentCustomers.slice(0, 10)) {
    const parts = [`${c.name} (${c.status})`];
    if (c.channel) parts.push(`via ${c.channel}`);
    if (c.daysAgo !== null && c.daysAgo !== undefined) parts.push(`${c.daysAgo}d ago`);
    if (c.intent) parts.push(`interest: ${c.intent}`);
    lines.push(`  - ${parts.join(' | ')}`);
  }
  lines.push('==============================');
  return lines.join('\n');
}

/**
 * Format inline records (small modules fetched eagerly) into a compact data block.
 * This is injected into EVERY message so the LLM always has the actual data.
 */
// Priority fields shown first — keeps token count low while preserving key values
const PRIORITY_FIELDS = [
  'name', 'price', 'total', 'amount', 'listprice', 'rate', 'cost', 'quantity',
  'discount', 'tax', 'status', 'stage', 'email', 'phone', 'company',
  'subject', 'closingdate', 'duedate', 'description',
];

function formatInlineRecords(inlineRecords: Record<string, InlineRecord[]>): string {
  const allRecords = Object.values(inlineRecords).flat();
  if (!allRecords.length) return '';

  // Group by module
  const byModule: Record<string, InlineRecord[]> = {};
  for (const r of allRecords) {
    const key = `${r.channel}:${r.module}`;
    if (!byModule[key]) byModule[key] = [];
    byModule[key].push(r);
  }

  const sections: string[] = ['=== CRM RECORDS ==='];
  for (const [key, records] of Object.entries(byModule)) {
    const [, mod] = key.split(':');
    sections.push(`[${mod}]`);
    for (const r of records) {
      // Sort fields: priority fields first, then others
      const allFields = Object.entries(r.data).filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== '');
      const priority = allFields.filter(([k]) => PRIORITY_FIELDS.includes(k.toLowerCase()));
      const rest     = allFields.filter(([k]) => !PRIORITY_FIELDS.includes(k.toLowerCase()));
      const shown    = [...priority, ...rest].slice(0, 4);
      const fieldStr = shown.map(([k, v]) => `${k}=${v}`).join(', ');
      sections.push(`• ${r.displayName}${fieldStr ? ': ' + fieldStr : ''}`);
    }
  }
  sections.push('Use the above data to answer questions about products, prices, invoices, deals, and accounts.');
  return sections.join('\n');
}

/**
 * Exported for use in the LangGraph CRM query node.
 * Returns a concise CRM overview string from a full TenantContext.
 */
export function formatCRMContextForQuery(ctx: TenantContext): string {
  return formatCRMContext(ctx);
}

/**
 * Resolve all tenant configuration for the AI agent.
 * Falls back to request-body values if the backend is unavailable.
 */
export async function resolveTenantConfig(
  tenantId: string,
  fallback: {
    companyName?: string;
    agentName?: string;
    language?: string;
    customInstructions?: string;
  } = {}
): Promise<ResolvedTenantConfig> {
  const ctx = await backendClient.getTenantContext(tenantId);

  if (!ctx) {
    logger.warn('Using fallback tenant config — backend unavailable', { tenantId });
    return {
      companyName: fallback.companyName || 'our company',
      agentName: fallback.agentName || 'Aria',
      language: fallback.language || 'English',
      systemPrompt: fallback.customInstructions,
      fallbackToHuman: true,
      crmContext: '',
      customerContext: '',
      inlineRecordsBlock: '',
      hasConnectors: false,
      connectorTypes: [],
      crmModules: {},
      qnaPairs: [],
      templates: [],
      websiteProfile: null,
      monthlyTokenLimit: DEFAULT_MONTHLY_TOKEN_LIMITS.starter,
      hasWidgetDepartments: false,
    };
  }

  return {
    companyName: ctx.tenant.branding?.companyName || ctx.tenant.name || fallback.companyName || 'our company',
    agentName: ctx.tenant.aiConfig?.agentName || fallback.agentName || 'Aria',
    language: ctx.tenant.aiConfig?.language || ctx.tenant.settings?.language || fallback.language || 'English',
    systemPrompt: ctx.tenant.aiConfig?.systemPrompt || fallback.customInstructions,
    fallbackToHuman: ctx.tenant.aiConfig?.fallbackToHuman ?? true,
    crmContext: formatCRMContext(ctx),
    customerContext: formatCustomerContext(ctx),
    inlineRecordsBlock: formatInlineRecords(ctx.inlineRecords || {}),
    hasConnectors: ctx.connectors.length > 0,
    connectorTypes: ctx.connectors.map((c) => c.type),
    crmModules: ctx.crmModules,
    qnaPairs: ctx.qnaPairs || [],
    templates: ctx.templates || [],
    websiteProfile: ctx.websiteProfile ?? null,
    monthlyTokenLimit: ctx.tenant.aiConfig?.monthlyTokenLimit
      ?? DEFAULT_MONTHLY_TOKEN_LIMITS[ctx.tenant.plan]
      ?? DEFAULT_MONTHLY_TOKEN_LIMITS.starter,
    toolModelPreset: ctx.tenant.aiConfig?.toolModelPreset,
    hasWidgetDepartments: ctx.hasWidgetDepartments ?? false,
  };
}
