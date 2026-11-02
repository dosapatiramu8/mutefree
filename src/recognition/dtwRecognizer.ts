import { classify, type Model } from '../model/classifier';
import { trimToMotion } from '../activity/segment';
import type { Candidate, RecognitionToken, VisualRecognitionResult, VisualSpeechInput, VisualSpeechRecognizer } from '../core/types';

const MAX_CANDIDATES = 3;

/**
 * Legacy recognizer: calibrated vocabulary + DTW nearest neighbour, exposed through the
 * VisualSpeechRecognizer interface and enriched with candidates and provenance.
 *
 * Decision rules (identical to the pre-refactor behaviour):
 *  - nearest sample farther than the user's own rejection distance -> `unknown`, text "?", no candidates
 *  - otherwise, margin >= threshold -> `visual_confident`
 *  - otherwise -> `visual_uncertain`, text "?", ranked candidates attached
 */
export class DtwRecognizer implements VisualSpeechRecognizer {
  model: Model | null = null;
  /** Minimum margin (1 - d1/d2) for a word to count as confident. */
  threshold = 0.15;

  async initialize(): Promise<void> { /* model is set after calibration */ }
  async dispose(): Promise<void> { this.model = null; }

  async recognize(input: VisualSpeechInput): Promise<VisualRecognitionResult> {
    const m = this.model;
    if (!m) throw new Error('No calibration: the DTW recognizer has no model');
    const t0 = performance.now();
    const trimmed = trimToMotion(input.frames, input.times);
    const p = classify(m, trimmed.frames, trimmed.times);
    const candidates = scoreCandidates(p.ranked, m.rejectDist);
    const common = { startMs: input.startMs, endMs: input.endMs, visualConfidence: p.conf };

    let token: RecognitionToken;
    if (p.label === null) {
      token = { ...common, text: '?', source: 'unknown', candidates: [] };
    } else if (p.conf >= this.threshold) {
      token = { ...common, text: p.label, source: 'visual_confident', candidates };
    } else {
      token = { ...common, text: '?', source: 'visual_uncertain', candidates };
    }
    return {
      token,
      diagnostics: { distance: p.d1, rejectDistance: m.rejectDist, runnerUpDistance: p.d2 },
      latencyMs: performance.now() - t0,
    };
  }
}

/**
 * Turn DTW distances into normalized candidate scores: exp(-(d - d_best) / s) with
 * s = 0.1 * rejectDist, then normalized over the top candidates. This is a monotone
 * heuristic, NOT a calibrated probability; the evaluation tooling is what calibrates it.
 */
export function scoreCandidates(ranked: { word: string; distance: number }[], rejectDist: number): Candidate[] {
  const top = ranked.filter((r) => Number.isFinite(r.distance)).slice(0, MAX_CANDIDATES);
  if (!top.length) return [];
  const s = Math.max(rejectDist * 0.1, 1e-6);
  const raw = top.map((r) => Math.exp(-(r.distance - top[0].distance) / s));
  const sum = raw.reduce((a, b) => a + b, 0);
  return top.map((r, i) => ({ text: r.word, visualConfidence: raw[i] / sum }));
}
