import { describe, expect, it } from 'vitest';
import { DtwRecognizer, scoreCandidates } from '../src/recognition/dtwRecognizer';
import { makeWords } from './helpers';

describe('DtwRecognizer', () => {
  it('throws a clear error without calibration', async () => {
    const w = makeWords(4);
    await expect(new DtwRecognizer().recognize(w.utter(0))).rejects.toThrow(/No calibration/);
  });

  it('returns visual_confident tokens with provenance and ranked candidates', async () => {
    const w = makeWords(6);
    const ev = await w.train();
    const r = new DtwRecognizer();
    r.model = ev.model;
    r.threshold = 0.05;
    let ok = 0;
    for (let i = 0; i < 6; i++) {
      const res = await r.recognize(w.utter(i));
      if (res.token.source === 'visual_confident' && res.token.text === w.vocab[i]) ok++;
      expect(res.token.candidates.length).toBeLessThanOrEqual(3);
      const sum = res.token.candidates.reduce((a, c) => a + c.visualConfidence, 0);
      expect(sum).toBeCloseTo(1, 5);
      expect(res.token.candidates[0].visualConfidence).toBeGreaterThanOrEqual(res.token.candidates.at(-1)!.visualConfidence);
    }
    expect(ok).toBe(6);
  });

  it('marks the same movement visual_uncertain (text "?") with candidates when below the threshold', async () => {
    const w = makeWords(6);
    const ev = await w.train();
    const r = new DtwRecognizer();
    r.model = ev.model;
    r.threshold = 0.99; // nothing can be this sure
    const res = await r.recognize(w.utter(2));
    expect(res.token.source).toBe('visual_uncertain');
    expect(res.token.text).toBe('?');
    expect(res.token.candidates[0].text).toBe('w2');
  });

  it('marks movement unlike any calibrated word as unknown, with no candidates', async () => {
    const w = makeWords(6);
    const ev = await w.train();
    const r = new DtwRecognizer();
    r.model = ev.model;
    let unknown = 0;
    for (let i = 0; i < 10; i++) {
      const res = await r.recognize(w.junk());
      if (res.token.source === 'unknown' && res.token.candidates.length === 0) unknown++;
    }
    expect(unknown).toBe(10);
  });
});

describe('scoreCandidates', () => {
  it('is monotone in distance and normalized', () => {
    const c = scoreCandidates([{ word: 'a', distance: 0.4 }, { word: 'b', distance: 0.5 }, { word: 'c', distance: 0.9 }], 1);
    expect(c.map((x) => x.text)).toEqual(['a', 'b', 'c']);
    expect(c[0].visualConfidence).toBeGreaterThan(c[1].visualConfidence);
    expect(c.reduce((a, x) => a + x.visualConfidence, 0)).toBeCloseTo(1, 6);
  });
  it('handles empty and infinite distances', () => {
    expect(scoreCandidates([], 1)).toEqual([]);
    expect(scoreCandidates([{ word: 'a', distance: Infinity }], 1)).toEqual([]);
  });
});
