import { defineConfig } from 'vitest/config';

// Relative base so the built site works from any static host or sub-path.
export default defineConfig({
  base: './',
  test: { include: ['tests/**/*.test.ts'], environment: 'node' },
});
