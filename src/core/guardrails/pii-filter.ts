export interface PIIFilterResult {
  filtered: string;
  detected: string[];
}

const PII_PATTERNS: Array<{ name: string; pattern: RegExp; replacement: string }> = [
  { name: 'credit_card', pattern: /\b(?:\d[ -]?){13,16}\b/g, replacement: '[CARD-REDACTED]' },
  { name: 'ssn', pattern: /\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b/g, replacement: '[SSN-REDACTED]' },
  {
    name: 'email',
    pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    replacement: '[EMAIL-REDACTED]',
  },
  {
    name: 'phone_sg',
    pattern: /\b(?:\+65\s?)?[689]\d{3}\s?\d{4}\b/g,
    replacement: '[PHONE-REDACTED]',
  },
  {
    name: 'phone_my',
    pattern: /\b(?:\+?60[-\s]?)?(?:1[0-9][-\s]?\d{7,8}|[3-9]\d[-\s]?\d{6,7})\b/g,
    replacement: '[PHONE-REDACTED]',
  },
  {
    name: 'phone_in',
    pattern: /\b(?:\+?91[-\s]?)?[6-9]\d{9}\b/g,
    replacement: '[PHONE-REDACTED]',
  },
  {
    name: 'phone_intl',
    pattern: /\+[1-9]\d{6,14}\b/g,
    replacement: '[PHONE-REDACTED]',
  },
  {
    name: 'nric',
    pattern: /\b[STFGM]\d{7}[A-Z]\b/gi,
    replacement: '[NRIC-REDACTED]',
  },
  {
    name: 'passport',
    pattern: /\b[A-Z]{1,2}\d{6,9}\b/g,
    replacement: '[PASSPORT-REDACTED]',
  },
];

export function filterPII(text: string): PIIFilterResult {
  const detected: string[] = [];
  let filtered = text;

  for (const { name, pattern, replacement } of PII_PATTERNS) {
    if (pattern.test(filtered)) {
      detected.push(name);
      filtered = filtered.replace(pattern, replacement);
    }
    pattern.lastIndex = 0;
  }

  return { filtered, detected };
}

import { config } from '../../config';

export function maskPIIForLLM(text: string): string {
  if (!config.guardrails.enablePiiFilter) return text;
  return filterPII(text).filtered;
}
