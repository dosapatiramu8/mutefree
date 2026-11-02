import { describe, expect, it } from 'vitest';
import { parseSimulated } from '../src/context/simulate';
import { SentencePipeline } from '../src/context/sentencePipeline';
import { CandidateScoringResolver } from '../src/context/candidateResolver';
import { NgramScorer } from '../src/context/ngramScorer';
import { CORPUS } from '../src/context/corpus';

describe('typed simulator', () => {
  it('parses confident, uncertain and unknown tokens', () => {
    const t = parseSimulated('I want water/waiter/later ?', 5000);
    expect(t.map((x) => x.source)).toEqual(['visual_confident', 'visual_confident', 'visual_uncertain', 'unknown']);
    expect(t[2].candidates.map((c) => c.text)).toEqual(['water', 'waiter', 'later']);
    expect(t[2].candidates.reduce((a, c) => a + c.visualConfidence, 0)).toBeCloseTo(1, 6);
    expect(t[3].endMs).toBe(5000);
    expect(t[0].endMs).toBeLessThan(t[1].endMs);
  });
  it('runs through the real pipeline: fills the gap, marks it inferred, speaks only real words', async () => {
    const p = new SentencePipeline({ timeoutMs: 500, maxTokens: 12 });
    p.resolver = new CandidateScoringResolver(new NgramScorer(CORPUS));
    const spoken: string[] = [];
    p.speak = (s) => spoken.push(s);
    parseSimulated('I want water/waiter/later', 1000).forEach((t) => p.addToken(t));
    const f = await p.tick(2000);
    expect(f!.display).toBe('I want [water?]');
    expect(spoken).toEqual(['I want water.']);
  });
  it('leaves a gap alone when the typed candidates give no safe winner', async () => {
    const p = new SentencePipeline({ timeoutMs: 500, maxTokens: 12 });
    p.resolver = new CandidateScoringResolver(new NgramScorer(CORPUS));
    const spoken: string[] = [];
    p.speak = (s) => spoken.push(s);
    parseSimulated('zzz/yyy', 1000).forEach((t) => p.addToken(t));
    const f = await p.tick(2000);
    expect(f!.display).toBe('?');
    expect(spoken).toEqual([]);
  });
});
