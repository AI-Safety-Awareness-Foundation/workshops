import type { ConversationSettings } from '../types';
import { DEFAULT_SETTINGS } from '../types';

// Merge overrides from /config.json into DEFAULT_SETTINGS. The file is
// gitignored and deployed separately by upload.sh; any subset of
// ConversationSettings keys may be present. Must run before the app renders
// (see main.tsx) so new conversations pick up the overridden defaults.
export async function loadConfigOverrides(): Promise<void> {
  try {
    const response = await fetch('/config.json', { cache: 'no-store' });
    if (!response.ok) {
      console.warn('No config.json found - using built-in default settings.');
      return;
    }
    const overrides: Partial<ConversationSettings> = await response.json();
    Object.assign(DEFAULT_SETTINGS, overrides);
  } catch {
    // Missing or unparseable (e.g. dev server returning index.html as the
    // SPA fallback) - keep built-in defaults.
    console.warn('No config.json found - using built-in default settings.');
  }
}
