import type { ContextResolution, ContextResolver, ConversationContext, RecognitionToken } from '../core/types';
import { displayText, spokenText, SentenceBuffer, type UnresolvedPolicy } from './sentenceBuffer';
import { NullResolver, resolveSafely } from './resolver';

export interface FinalSentence {
  raw: RecognitionToken[];
  resolution: ContextResolution;
  display: string;
  spoken: string;
  startMs: number;
  endMs: number;
}

const MAX_CONTEXT_SENTENCES = 5;

/**
 * visual tokens -> sentence buffer -> (optional) context resolver -> final text -> speak.
 * Words are never spoken one by one; the sentence is spoken once, when it is finalized.
 * A failing resolver never loses the utterance: the raw visual tokens are used instead.
 */
export class SentencePipeline {
  readonly buffer: SentenceBuffer;
  readonly context: ConversationContext = { sentences: [] };
  resolver: ContextResolver = new NullResolver();
  policy: UnresolvedPolicy = 'skip';
  resolverTimeoutMs = 1500;
  onChange?: () => void;
  onFinal?: (f: FinalSentence) => void;
  /** Called with the text to speak, only when there is something to say. */
  speak?: (text: string, final: FinalSentence) => void;
  private busy = false;
  /** True while the context resolver is working on a finished sentence. */
  resolving = false;

  constructor(opts = { timeoutMs: 1500, maxTokens: 12 }) {
    this.buffer = new SentenceBuffer(opts);
  }

  addToken(t: RecognitionToken) { this.buffer.add(t); this.onChange?.(); }
  clearBuffer() { this.buffer.clear(); this.onChange?.(); }
  clearContext() { this.context.sentences.length = 0; }

  /** Call regularly (e.g. every frame). Finalizes and speaks the sentence when it is complete. */
  async tick(nowMs: number): Promise<FinalSentence | null> {
    if (this.busy || !this.buffer.shouldFinalize(nowMs)) return null;
    this.busy = true;
    try {
      const raw = this.buffer.take();
      this.resolving = true; this.onChange?.();
      const resolution = await resolveSafely(this.resolver, raw, this.context, this.resolverTimeoutMs);
      const spoken = spokenText(resolution.tokens, this.policy);
      const final: FinalSentence = {
        raw, resolution, display: displayText(resolution.tokens), spoken,
        startMs: raw[0].startMs, endMs: raw[raw.length - 1].endMs,
      };
      if (spoken) {
        this.context.sentences.push(spoken);
        if (this.context.sentences.length > MAX_CONTEXT_SENTENCES) this.context.sentences.shift();
        this.speak?.(spoken, final);
      }
      this.onFinal?.(final);
      return final;
    } finally {
      this.busy = false; this.resolving = false;
      this.onChange?.();
    }
  }
}
