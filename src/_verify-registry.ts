import { getToolsForSurface } from './tools/registry';

const full = getToolsForSurface('public_widget');
const bookingOnly = getToolsForSurface('public_widget', { bookingOnly: true });
const notBookingOnly = getToolsForSurface('public_widget', { bookingOnly: false });

console.log('full:', full.map(t => t.name));
console.log('bookingOnly=true:', bookingOnly.map(t => t.name));
console.log('bookingOnly=false:', notBookingOnly.map(t => t.name));
