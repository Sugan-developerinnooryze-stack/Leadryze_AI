// Cheap, regex-based first-pass extraction of email/phone/name from a raw
// visitor message — used both by base.agent.ts's own lead-capture path and
// the deterministic booking-confirmation shortcut. Lives in its own module
// (rather than staying inline in base.agent.ts) so both can import it
// without a circular dependency between the two.
export function extractCapturedData(userMessage: string): Record<string, string> {
  const data: Record<string, string> = {};

  const emailMatch = userMessage.match(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/);
  if (emailMatch) data.email = emailMatch[0];

  const phoneMatch = userMessage.match(
    /\b(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b/
  );
  if (phoneMatch) data.phone = phoneMatch[0].trim();

  const namePatterns = [
    /(?:my name is|my name's|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i,
    /(?:name:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i,
  ];
  for (const p of namePatterns) {
    const m = userMessage.match(p);
    if (m) { data.name = m[1]; break; }
  }

  return data;
}
