// Shared time-of-day parsing helpers — extracted so both the deterministic
// booking shortcut (booking-confirmation-shortcut.ts) and the availability
// tool (check-availability.tool.ts) use the exact same logic, rather than
// two copies that could drift apart.

/** Extracts the trailing "h:mm AM/PM" token from a slot label formatted by
 * availability.service.ts's own formatLabel() (e.g. "Mon, Aug 3, 1:00 PM"),
 * normalised to 24h "HH:MM" for comparison. */
export function labelTimeTo24h(label: string): string | null {
  const m = label.match(/(\d{1,2}):(\d{2})\s*([AP]M)/i);
  if (!m) return null;
  let hour = parseInt(m[1], 10);
  const minute = m[2];
  const ampm = m[3].toUpperCase();
  if (ampm === 'PM' && hour !== 12) hour += 12;
  if (ampm === 'AM' && hour === 12) hour = 0;
  return `${String(hour).padStart(2, '0')}:${minute}`;
}

/** Extracts a time mention from the visitor's own message (e.g. "1pm",
 * "1:00 pm", "13:00"), normalised the same way as labelTimeTo24h(). */
export function messageTimeTo24h(message: string): string | null {
  const ampmMatch = message.match(/\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b/i);
  if (ampmMatch) {
    let hour = parseInt(ampmMatch[1], 10);
    const minute = ampmMatch[2] ?? '00';
    const isPm = ampmMatch[3].toLowerCase() === 'p';
    if (isPm && hour !== 12) hour += 12;
    if (!isPm && hour === 12) hour = 0;
    return `${String(hour).padStart(2, '0')}:${minute}`;
  }
  const militaryMatch = message.match(/\b([01]?\d|2[0-3]):([0-5]\d)\b/);
  if (militaryMatch) {
    return `${militaryMatch[1].padStart(2, '0')}:${militaryMatch[2]}`;
  }
  return null;
}

/** Minutes-since-midnight for a "HH:MM" string — used to sort/rank slots by
 * closeness to a requested time. */
export function timeToMinutes(hhmm: string): number {
  const [h, m] = hhmm.split(':').map(Number);
  return h * 60 + m;
}
