import { describe, expect, it, vi } from 'vitest';
import { SentencePipeline } from '../src/context/sentencePipeline';
import type { ContextResolver, RecognitionToken } from '../src/core/types';
import { tok } from './helpers';

const fill = (p: SentencePipeline, toks: RecognitionToken[]) => toks.forEach((t) => p.addToken(t));

describe('SentencePipeline', () => {
  it('speaks the sentence once, only after the pause', async () => {
    const p = new SentencePipeline({ timeoutMs: 1000, maxTokens: 12 });
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    fill(p, [tok('I', 'visual_confident', 1000), tok('want', 'visual_confident', 2000), tok('water', 'visual_confident', 3000)]);
    expect(await p.tick(3500)).toBeNull(); // still inside the pause window
    expect(spoken).toEqual([]);
    await p.tick(4100);
    expect(spoken).toEqual(['I want water.']);
    expect(await p.tick(9000)).toBeNull(); // nothing left to say twice
    expect(spoken).toHaveLength(1);
  });

  it('skips unresolved gaps in speech but keeps them visible', async () => {
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    fill(p, [tok('I', 'visual_confident', 100), tok('?', 'visual_uncertain', 200), tok('please', 'visual_confident', 300)]);
    const f = await p.tick(1000);
    expect(f!.display).toBe('I ? please');
    expect(spoken).toEqual(['I please.']);
  });

  it('says nothing when every token is unresolved', async () => {
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    fill(p, [tok('?', 'unknown', 100)]);
    await p.tick(1000);
    expect(spoken).toEqual([]);
  });

  it('marks resolver output as context_inferred and speaks it', async () => {
    const resolver: ContextResolver = {
      async resolve(tokens) {
        return {
          tokens: tokens.map((t) => (t.source === 'visual_uncertain' ? { ...t, text: 'water', source: 'context_inferred' as const, languageConfidence: 0.87 } : t)),
          replacements: [{ position: 2, from: '?', to: 'water' }], confidence: 0.87, fallback: false, latencyMs: 5,
        };
      },
    };
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    p.resolver = resolver;
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    fill(p, [tok('I', 'visual_confident', 100), tok('want', 'visual_confident', 200), tok('?', 'visual_uncertain', 300)]);
    const f = await p.tick(1000);
    expect(f!.display).toBe('I want [water?]');
    expect(spoken).toEqual(['I want water.']);
    expect(f!.resolution.replacements[0]).toMatchObject({ from: '?', to: 'water' });
  });

  it('falls back to the raw visual tokens when the resolver throws', async () => {
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    p.resolver = { resolve: () => Promise.reject(new Error('model crashed')) };
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    fill(p, [tok('yes', 'visual_confident', 100)]);
    const f = await p.tick(1000);
    expect(f!.resolution.fallback).toBe(true);
    expect(spoken).toEqual(['Yes.']);
  });

  it('falls back when the resolver is too slow', async () => {
    vi.useFakeTimers();
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    p.resolverTimeoutMs = 50;
    p.resolver = { resolve: () => new Promise(() => { /* never resolves */ }) };
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    fill(p, [tok('stop', 'visual_confident', 100)]);
    const run = p.tick(1000);
    await vi.advanceTimersByTimeAsync(60);
    expect((await run)!.resolution.fallback).toBe(true);
    expect(spoken).toEqual(['Stop.']);
    vi.useRealTimers();
  });

  it('keeps recent sentences as context and can clear them', async () => {
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    for (let i = 0; i < 7; i++) {
      fill(p, [tok('hi', 'visual_confident', 100 + i * 1000)]);
      await p.tick(100 + i * 1000 + 500);
    }
    expect(p.context.sentences).toHaveLength(5);
    p.clearContext();
    expect(p.context.sentences).toHaveLength(0);
  });
});
