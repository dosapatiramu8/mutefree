import { GEOM_DIM, STATIC_DIM } from '../features/mouth';

/** Frames per resampled word. */
export const T = 24;
const BAND = 5;

export type FeatureSet = 'geometry' | 'geometry+extras';
export const featureDims = (s: FeatureSet) => (s === 'geometry' ? GEOM_DIM : STATIC_DIM);

export interface Sample { word: string; seq: number[][]; durationMs: number; } // T x STATIC_DIM (raw)

/** Linear-interpolate a variable-length frame sequence to exactly T frames (static part only). */
export function resample(frames: Float32Array[], times: number[]): number[][] {
  const out: number[][] = [];
  const t0 = times[0], t1 = times[times.length - 1];
  let j = 0;
  for (let k = 0; k < T; k++) {
    const t = t0 + ((t1 - t0) * k) / (T - 1);
    while (j < times.length - 2 && times[j + 1] < t) j++;
    const a = frames[j], b = frames[Math.min(j + 1, frames.length - 1)];
    const span = times[Math.min(j + 1, times.length - 1)] - times[j];
    const u = span > 0 ? Math.min(Math.max((t - times[j]) / span, 0), 1) : 0;
    const row = new Array<number>(STATIC_DIM);
    for (let d = 0; d < STATIC_DIM; d++) row[d] = a[d] + (b[d] - a[d]) * u;
    out.push(row);
  }
  return out;
}

export interface Model {
  vocab: string[];
  featureSet: FeatureSet;
  dim: number;
  mean: Float32Array;
  std: Float32Array;
  labels: string[];          // label per stored sample
  data: Float32Array[];      // normalized T*dim per sample
  rejectDist: number;        // nearest-class distance above this => "?"
}

function normalize(seq: number[][], dim: number, mean: Float32Array, std: Float32Array): Float32Array {
  const out = new Float32Array(T * dim);
  for (let t = 0; t < T; t++) for (let d = 0; d < dim; d++) out[t * dim + d] = (seq[t][d] - mean[d]) / std[d];
  return out;
}

/** Banded DTW over T x dim sequences; cost is squared Euclidean per frame pair. */
export function dtw(a: Float32Array, b: Float32Array, dim: number): number {
  const INF = 1e30;
  let prev = new Float32Array(T + 1).fill(INF), cur = new Float32Array(T + 1).fill(INF);
  prev[0] = 0;
  for (let i = 1; i <= T; i++) {
    cur.fill(INF);
    const jl = Math.max(1, i - BAND), jh = Math.min(T, i + BAND);
    for (let j = jl; j <= jh; j++) {
      let c = 0;
      const ao = (i - 1) * dim, bo = (j - 1) * dim;
      for (let d = 0; d < dim; d++) { const x = a[ao + d] - b[bo + d]; c += x * x; }
      cur[j] = c / dim + Math.min(prev[j], prev[j - 1], cur[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  return Math.sqrt(prev[T] / (2 * T));
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

export interface Evaluation {
  featureSet: FeatureSet;
  vocab: string[];
  confusion: number[][];      // [true][predicted]
  accuracy: number;
  perWord: number[];
  samples: number;
  rejectDist: number;
  model: Model;
}

/** Fit normalization, build the kNN model, and run leave-one-out cross-validation. */
export async function trainAndEvaluate(
  samples: Sample[], vocab: string[], featureSet: FeatureSet, onProgress?: (f: number) => void,
): Promise<Evaluation> {
  const dim = featureDims(featureSet);
  const mean = new Float32Array(dim), std = new Float32Array(dim);
  let n = 0;
  for (const s of samples) for (const row of s.seq) { n++; for (let d = 0; d < dim; d++) mean[d] += row[d]; }
  for (let d = 0; d < dim; d++) mean[d] /= n;
  for (const s of samples) for (const row of s.seq) for (let d = 0; d < dim; d++) std[d] += (row[d] - mean[d]) ** 2;
  for (let d = 0; d < dim; d++) std[d] = Math.sqrt(std[d] / n) || 1;

  const data = samples.map((s) => normalize(s.seq, dim, mean, std));
  const labels = samples.map((s) => s.word);
  const N = samples.length;
  const D = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) D[i * N + j] = D[j * N + i] = dtw(data[i], data[j], dim);
    if (i % 8 === 0) { onProgress?.(i / N); await tick(); }
  }

  const idx = new Map(vocab.map((w, i) => [w, i]));
  const V = vocab.length;
  const confusion = Array.from({ length: V }, () => new Array<number>(V).fill(0));
  const correctD1: number[] = [];
  for (let i = 0; i < N; i++) {
    const best = new Array<number>(V).fill(Infinity);
    for (let j = 0; j < N; j++) if (j !== i) { const c = idx.get(labels[j])!; best[c] = Math.min(best[c], D[i * N + j]); }
    let p = 0;
    for (let c = 1; c < V; c++) if (best[c] < best[p]) p = c;
    const t = idx.get(labels[i])!;
    confusion[t][p]++;
    if (p === t) correctD1.push(best[p]);
  }
  const perWord = confusion.map((row, i) => { const tot = row.reduce((a, b) => a + b, 0); return tot ? row[i] / tot : 0; });
  const correct = confusion.reduce((a, row, i) => a + row[i], 0);
  correctD1.sort((a, b) => a - b);
  const rejectDist = (correctD1[Math.floor(0.9 * (correctD1.length - 1))] ?? 1) * 1.5;

  return {
    featureSet, vocab, confusion, accuracy: N ? correct / N : 0, perWord, samples: N, rejectDist,
    model: { vocab, featureSet, dim, mean, std, labels, data, rejectDist },
  };
}

export interface Prediction {
  label: string | null; best: string; conf: number; d1: number; d2: number;
  /** Every vocabulary word with its nearest-sample distance, best first. */
  ranked: { word: string; distance: number }[];
}

/**
 * Classify one segment. Confidence is the margin between the nearest class and the
 * runner-up (1 - d1/d2). Returns label=null (show "?") when the nearest distance is
 * beyond what this user's own good samples produce; callers also apply a conf threshold.
 */
export function classify(model: Model, frames: Float32Array[], times: number[]): Prediction {
  const q = normalize(resample(frames, times), model.dim, model.mean, model.std);
  const best = new Array<number>(model.vocab.length).fill(Infinity);
  const idx = new Map(model.vocab.map((w, i) => [w, i]));
  for (let i = 0; i < model.data.length; i++) {
    const c = idx.get(model.labels[i]);
    if (c === undefined) continue;
    best[c] = Math.min(best[c], dtw(q, model.data[i], model.dim));
  }
  let p = 0;
  for (let c = 1; c < best.length; c++) if (best[c] < best[p]) p = c;
  let d2 = Infinity;
  for (let c = 0; c < best.length; c++) if (c !== p) d2 = Math.min(d2, best[c]);
  const d1 = best[p];
  const conf = Number.isFinite(d2) && d2 > 0 ? Math.max(0, 1 - d1 / d2) : 0;
  const ranked = model.vocab.map((word, i) => ({ word, distance: best[i] })).sort((a, b) => a.distance - b.distance);
  return { label: d1 <= model.rejectDist ? model.vocab[p] : null, best: model.vocab[p], conf, d1, d2, ranked };
}
