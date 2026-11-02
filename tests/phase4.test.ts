import { describe, expect, it } from 'vitest';
import { CandidateScoringResolver } from '../src/context/candidateResolver';
import { NgramScorer } from '../src/context/ngramScorer';
import { CORPUS } from '../src/context/corpus';
import { WebLlmScorer, lettersToProbs, type EngineFactory } from '../src/context/webllmScorer';
import { SentencePipeline } from '../src/context/sentencePipeline';
import type { LanguageScorer, RecognitionToken, ScoreRequest } from '../src/core/types';
import { rng, tok } from './helpers';

const uncertain = (cands: [string, number][], endMs: number): RecognitionToken =>
  tok('?', 'visual_uncertain', endMs, { candidates: cands.map(([text, visualConfidence]) => ({ text, visualConfidence })) });

const ctx = { sentences: [] as string[] };

describe('NgramScorer (built-in phrase model)', () => {
  const s = new NgramScorer(CORPUS);
  it('prefers the candidate that fits the left context', async () => {
    const p = await s.score({ left: ['I', 'want'], right: [], candidates: ['water', 'waiter', 'later'], history: [] });
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[0]).toBeGreaterThan(p[1]);
    expect(p[0]).toBeGreaterThan(p[2]);
  });
  it('uses the right context too', async () => {
    const p = await s.score({ left: ['I', 'am'], right: ['please', 'turn', 'it', 'off'], candidates: ['hot', 'hut', 'hat'], history: [] });
    expect(p[0]).toBeGreaterThan(p[1]);
  });
  it('gives words it has never seen only the smoothing floor', async () => {
    const p = await s.score({ left: ['I', 'want'], right: [], candidates: ['water', 'zzyzx'], history: [] });
    expect(p[0]).toBeGreaterThan(0.9);
  });
});

describe('CandidateScoringResolver safety rules', () => {
  const resolver = new CandidateScoringResolver(new NgramScorer(CORPUS));

  it('fills a gap from the visual candidates and records the evidence', async () => {
    const tokens = [tok('I', 'visual_confident', 100), tok('want', 'visual_confident', 200), uncertain([['water', 0.4], ['waiter', 0.35], ['later', 0.25]], 300)];
    const r = await resolver.resolve(tokens, ctx);
    expect(r.tokens[2]).toMatchObject({ text: 'water', source: 'context_inferred' });
    expect(r.tokens[2].languageConfidence).toBeGreaterThan(0.5);
    expect(r.replacements[0]).toMatchObject({ position: 2, from: '?', to: 'water' });
    expect(r.replacements[0].combinedConfidence).toBeGreaterThanOrEqual(0.6);
    expect(r.fallback).toBe(false);
  });

  it('never modifies confident tokens, even if the language model would prefer another word', async () => {
    const tokens = [tok('I', 'visual_confident', 100), tok('want', 'visual_confident', 200), tok('later', 'visual_confident', 300, {
      candidates: [{ text: 'later', visualConfidence: 0.5 }, { text: 'water', visualConfidence: 0.5 }],
    })];
    const r = await resolver.resolve(tokens, ctx);
    expect(r.tokens.map((t) => t.text)).toEqual(['I', 'want', 'later']);
    expect(r.replacements).toEqual([]);
  });

  it('leaves unknown tokens as ?', async () => {
    const tokens = [tok('I', 'visual_confident', 100), tok('want', 'visual_confident', 200), tok('?', 'unknown', 300)];
    const r = await resolver.resolve(tokens, ctx);
    expect(r.tokens[2]).toMatchObject({ text: '?', source: 'unknown' });
  });

  it('does not guess from a single candidate', async () => {
    const tokens = [tok('I', 'visual_confident', 100), tok('want', 'visual_confident', 200), uncertain([['water', 1]], 300)];
    expect((await resolver.resolve(tokens, ctx)).tokens[2].text).toBe('?');
  });

  it('keeps the ? when the language evidence is flat', async () => {
    const flat: LanguageScorer = { name: 'flat', score: async (q) => q.candidates.map(() => 1 / q.candidates.length) };
    const tokens = [uncertain([['water', 0.4], ['waiter', 0.35], ['later', 0.25]], 300)];
    expect((await new CandidateScoringResolver(flat).resolve(tokens, ctx)).tokens[0].text).toBe('?');
  });

  it('cannot introduce a word the visual model did not offer', async () => {
    const greedy: LanguageScorer = { name: 'x', score: async (q) => q.candidates.map((_, i) => (i === 0 ? 0.97 : 0.03 / (q.candidates.length - 1))) };
    const tokens = [uncertain([['cat', 0.5], ['cap', 0.5]], 100)];
    const r = await new CandidateScoringResolver(greedy).resolve(tokens, ctx);
    expect(['cat', 'cap', '?']).toContain(r.tokens[0].text);
  });

  it('uses an earlier inference as context for a later one, left to right', async () => {
    const seen: ScoreRequest[] = [];
    const spy: LanguageScorer = { name: 'spy', score: async (q) => { seen.push(q); return q.candidates.map((_, i) => (i === 0 ? 0.9 : 0.1 / (q.candidates.length - 1))); } };
    const tokens = [uncertain([['want', 0.5], ['wait', 0.5]], 100), uncertain([['water', 0.5], ['later', 0.5]], 200)];
    await new CandidateScoringResolver(spy).resolve(tokens, ctx);
    expect(seen[0].left).toEqual([]);
    expect(seen[1].left).toEqual(['want']);   // first gap resolved, now context
    expect(seen[0].right).toEqual(['_']);     // the other gap is still unresolved
  });

  it('a scorer failure falls back to the raw visual tokens via the pipeline', async () => {
    const broken: LanguageScorer = { name: 'broken', score: () => Promise.reject(new Error('GPU lost')) };
    const p = new SentencePipeline({ timeoutMs: 100, maxTokens: 12 });
    p.resolver = new CandidateScoringResolver(broken);
    const spoken: string[] = [];
    p.speak = (t) => spoken.push(t);
    [tok('I', 'visual_confident', 100), uncertain([['a', 0.5], ['b', 0.5]], 200)].forEach((t) => p.addToken(t));
    const f = await p.tick(1000);
    expect(f!.resolution.fallback).toBe(true);
    expect(f!.display).toBe('I ?');
    expect(spoken).toEqual(['I.']);
  });
});

describe('WebLlmScorer', () => {
  it('turns answer-letter logprobs into probabilities', () => {
    const p = lettersToProbs([{ token: 'B', logprob: -0.1 }, { token: 'A', logprob: -2.5 }, { token: 'the', logprob: -3 }], 3);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[1]).toBeGreaterThan(0.85);
    expect(p[2]).toBeLessThan(0.001); // letter never proposed -> floor
  });

  const fake = (letterFor: (prompt: string) => string): EngineFactory => async () => ({
    chat: {
      completions: {
        create: async (req: unknown) => {
          const prompt = (req as { messages: { content: string }[] }).messages[1].content;
          return { choices: [{ logprobs: { content: [{ top_logprobs: [{ token: letterFor(prompt), logprob: -0.05 }, { token: 'C', logprob: -6 }] }] } }] };
        },
      },
    },
  });

  it('refuses to score before the model is loaded', async () => {
    await expect(new WebLlmScorer(fake(() => 'A')).score({ left: [], right: [], candidates: ['a', 'b'], history: [] })).rejects.toThrow(/not loaded/);
  });

  it('removes letter-position bias by asking in both orders', async () => {
    // A biased model that always answers "A": without debiasing it would always pick the first-listed candidate.
    const s = new WebLlmScorer(fake(() => 'A'));
    await s.load(() => {});
    const biased = await s.score({ left: ['I', 'want'], right: [], candidates: ['water', 'later'], history: [] });
    expect(Math.abs(biased[0] - biased[1])).toBeLessThan(0.05);   // bias cancels: no real preference

    // A model that really prefers "water" wherever it is listed.
    const real = new WebLlmScorer(fake((prompt) => (prompt.indexOf('A) water') >= 0 ? 'A' : 'B')));
    await real.load(() => {});
    const p = await real.score({ left: ['I', 'want'], right: [], candidates: ['water', 'later'], history: [] });
    expect(p[0]).toBeGreaterThan(0.9);
  });

  it('reports a clear error when loading fails', async () => {
    const s = new WebLlmScorer(async () => { throw new Error('WebGPU is not available in this browser'); });
    await expect(s.load(() => {})).rejects.toThrow(/WebGPU/);
    expect(s.state).toBe('error');
    expect(s.error).toMatch(/WebGPU/);
  });
});

describe('synthetic sanity check (NOT real evidence)', () => {
  // The phrase model is built from the same sentences, so this only proves the mechanism behaves,
  // not that it will help on real lip movements. Real numbers come from the Phase 8 evaluation.
  it('on near-tied visual candidates, picks the truth more often than the visual top-1 and rarely replaces wrongly', async () => {
    const { rnd } = rng(5);
    const vocabWords = ['water', 'help', 'doctor', 'medicine', 'toilet', 'family', 'tired', 'drink', 'again', 'sorry', 'later', 'waiter', 'hat', 'cat'];
    const resolver = new CandidateScoringResolver(new NgramScorer(CORPUS));
    let cases = 0, visualRight = 0, resolvedRight = 0, replaced = 0, wrongReplaced = 0;
    for (const line of CORPUS) {
      const w = line.split(' ');
      if (w.length < 2) continue;
      const pos = w.length - 1;
      const truth = w[pos].toLowerCase();
      const distract = vocabWords.filter((x) => x !== truth).sort(() => rnd() - 0.5).slice(0, 2);
      const cands = [truth, ...distract].sort(() => rnd() - 0.5);
      const scores = [0.38, 0.34, 0.28]; // visually near-tied
      const tokens: RecognitionToken[] = [
        ...w.slice(0, pos).map((x, i) => tok(x.toLowerCase(), 'visual_confident', (i + 1) * 100)),
        uncertain(cands.map((c, i) => [c, scores[i]] as [string, number]), (pos + 1) * 100),
      ];
      const r = await resolver.resolve(tokens, ctx);
      cases++;
      if (cands[0] === truth) visualRight++;
      if (r.tokens[pos].source === 'context_inferred') { replaced++; if (r.tokens[pos].text === truth) resolvedRight++; else wrongReplaced++; }
      else if (cands[0] === truth) resolvedRight += 0; // left as "?": not counted as right
    }
    expect(replaced / cases).toBeGreaterThan(0.5);
    expect(resolvedRight / cases).toBeGreaterThan(visualRight / cases);
    expect(wrongReplaced / Math.max(replaced, 1)).toBeLessThan(0.15);
  });
});
