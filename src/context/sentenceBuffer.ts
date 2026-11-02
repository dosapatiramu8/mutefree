import type { RecognitionToken } from '../core/types';

export interface SentenceBufferOptions {
  /** Silence after the last word that ends the sentence. */
  timeoutMs: number;
  /** Safety cap so a never-ending stream still gets spoken. */
  maxTokens: number;
}

/** Structured record of the current sentence: tokens, timing, confidence, source and candidates are all kept. */
export class SentenceBuffer {
  private tokens: RecognitionToken[] = [];
  constructor(public opts: SentenceBufferOptions) {}

  get size() { return this.tokens.length; }
  peek(): readonly RecognitionToken[] { return this.tokens; }
  add(t: RecognitionToken) { this.tokens.push(t); }
  clear() { this.tokens = []; }
  /** Drain and return the tokens. */
  take(): RecognitionToken[] { const t = this.tokens; this.tokens = []; return t; }

  /** True once the speaker has paused long enough, or the sentence is getting too long. */
  shouldFinalize(nowMs: number): boolean {
    if (!this.tokens.length) return false;
    const lastEnd = this.tokens[this.tokens.length - 1].endMs;
    return nowMs - lastEnd >= this.opts.timeoutMs || this.tokens.length >= this.opts.maxTokens;
  }
}

export type UnresolvedPolicy = 'skip' | 'hold';

/** What the screen shows. Unresolved tokens stay visible as "?", inferred ones are bracketed. */
export function displayText(tokens: readonly RecognitionToken[]): string {
  return tokens.map((t) => (t.source === 'context_inferred' ? `[${t.text}?]` : t.text)).join(' ');
}

/**
 * What the voice says: only words the system actually has (confident visual or context-inferred).
 * `skip` drops unresolved gaps; `hold` returns '' if any gap remains so nothing misleading is spoken.
 */
export function spokenText(tokens: readonly RecognitionToken[], policy: UnresolvedPolicy = 'skip'): string {
  const unresolved = tokens.some((t) => t.source === 'visual_uncertain' || t.source === 'unknown');
  if (unresolved && policy === 'hold') return '';
  const words = tokens.filter((t) => t.source === 'visual_confident' || t.source === 'context_inferred').map((t) => t.text);
  if (!words.length) return '';
  const s = words.join(' ');
  return s.charAt(0).toUpperCase() + s.slice(1) + '.';
}
