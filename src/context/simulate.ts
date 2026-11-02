import type { RecognitionToken } from '../core/types';

/**
 * Turns typed text into tokens so the sentence / language-model / speech path can be tried
 * without a camera or calibration. This is a test input, not lip reading, and the UI labels it so.
 *
 *   water              a word the "camera" was sure of
 *   water/waiter/later an unclear word with these candidates (best guess first)
 *   ?                  a movement that matched nothing
 *
 * Example: "I want water/waiter/later please"
 */
export function parseSimulated(text: string, endMs: number, stepMs = 400): RecognitionToken[] {
  const parts = text.trim().split(/\s+/).filter(Boolean);
  return parts.map((p, i): RecognitionToken => {
    const end = endMs - (parts.length - 1 - i) * stepMs;
    const base = { startMs: end - 300, endMs: end };
    if (p === '?') return { ...base, text: '?', source: 'unknown', visualConfidence: 0, candidates: [] };
    const words = p.split('/').map((w) => w.trim()).filter(Boolean);
    if (words.length < 2) return { ...base, text: words[0] ?? '?', source: 'visual_confident', visualConfidence: 0.9, candidates: [] };
    const weights = words.map((_, k) => 1 / (1 + 0.2 * k)); // gently decreasing: the first is only slightly preferred
    const sum = weights.reduce((a, b) => a + b, 0);
    return {
      ...base, text: '?', source: 'visual_uncertain', visualConfidence: 0.05,
      candidates: words.map((w, k) => ({ text: w, visualConfidence: weights[k] / sum })),
    };
  });
}
