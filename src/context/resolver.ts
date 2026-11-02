import type { ContextResolution, ContextResolver, ConversationContext, RecognitionToken } from '../core/types';

/** Leaves every token exactly as the visual model produced it (Smart completion OFF). */
export class NullResolver implements ContextResolver {
  async resolve(tokens: RecognitionToken[], _context: ConversationContext): Promise<ContextResolution> {
    return { tokens, replacements: [], confidence: 1, fallback: false, latencyMs: 0 };
  }
}

/**
 * Wraps any resolver so that a failure or a timeout can never lose the utterance:
 * the unmodified visual tokens come back flagged `fallback: true`.
 */
export async function resolveSafely(
  resolver: ContextResolver, tokens: RecognitionToken[], context: ConversationContext, timeoutMs: number,
): Promise<ContextResolution> {
  const t0 = performance.now();
  const fallback = (): ContextResolution => ({ tokens, replacements: [], confidence: 0, fallback: true, latencyMs: performance.now() - t0 });
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      resolver.resolve(tokens, context),
      new Promise<ContextResolution>((resolve) => { timer = setTimeout(() => resolve(fallback()), timeoutMs); }),
    ]);
  } catch {
    return fallback();
  } finally {
    clearTimeout(timer);
  }
}
