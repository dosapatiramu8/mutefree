import { FEATURE_DIM, STATIC_DIM } from '../src/features/mouth';
import { resample, trainAndEvaluate, type Sample } from '../src/model/classifier';
import type { RecognitionToken, TokenSource } from '../src/core/types';

/** Deterministic PRNG so tests never flake. */
export function rng(seed = 7) {
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  return { rnd, gauss };
}

/** Synthetic "words": smooth random trajectories per word, with time warping and noise. */
export function makeWords(n: number, seed = 7) {
  const { rnd, gauss } = rng(seed);
  const vocab = Array.from({ length: n }, (_, i) => `w${i}`);
  const tmpl = vocab.map(() => Array.from({ length: STATIC_DIM }, () => Array.from({ length: 4 }, () => gauss())));
  const utter = (w: number, noise = 0.3) => {
    const len = 14 + Math.floor(rnd() * 10);
    const frames: Float32Array[] = [], times: number[] = [];
    for (let f = 0; f < len; f++) {
      const u = (f / (len - 1)) * 3, k = Math.min(2, Math.floor(u)), a = u - k;
      const v = new Float32Array(FEATURE_DIM);
      for (let d = 0; d < STATIC_DIM; d++) v[d] = tmpl[w][d][k] * (1 - a) + tmpl[w][d][k + 1] * a + gauss() * noise;
      for (let d = 0; d < 83; d++) v[STATIC_DIM + d] = 1; // "moving" velocity so trimming keeps everything
      frames.push(v); times.push(f * 33);
    }
    return { frames, times, startMs: 0, endMs: len * 33 };
  };
  const junk = () => {
    const frames = Array.from({ length: 15 }, () => {
      const v = new Float32Array(FEATURE_DIM);
      for (let d = 0; d < STATIC_DIM; d++) v[d] = gauss() * 1.5;
      for (let d = 0; d < 83; d++) v[STATIC_DIM + d] = 1;
      return v;
    });
    return { frames, times: frames.map((_, i) => i * 33), startMs: 0, endMs: 500 };
  };
  const train = async () => {
    const samples: Sample[] = vocab.flatMap((word, w) =>
      Array.from({ length: 5 }, () => { const s = utter(w); return { word, seq: resample(s.frames, s.times), durationMs: 400 }; }));
    return trainAndEvaluate(samples, vocab, 'geometry+extras');
  };
  return { vocab, utter, junk, train };
}

export function tok(text: string, source: TokenSource, endMs: number, extra: Partial<RecognitionToken> = {}): RecognitionToken {
  return { text, source, visualConfidence: source === 'visual_confident' ? 0.9 : 0.1, candidates: [], startMs: endMs - 300, endMs, ...extra };
}
