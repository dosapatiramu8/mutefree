import type { Expr } from '../features/expression';

export interface VoiceParams { pitch: number; rate: number; volume: number; }
export const FLAT_VOICE: VoiceParams = { pitch: 1, rate: 1, volume: 0.8 };

const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

/**
 * Expression -> voice. Documented mapping (all inputs 0..1):
 *   smile      : pitch +0.30, rate +0.15
 *   frown      : pitch -0.25, rate -0.15
 *   brow up    : pitch +0.20, rate +0.10
 *   brow down  : pitch -0.30, rate -0.30
 *   jaw/mouth  : volume 0.4 + 0.6 * clamp(jaw * gain)   (gain 2 => jaw 0.5 is full)
 * A neutral face with jaw 0.25 gives pitch 1, rate 1, volume 0.7.
 */
export function mapExpression(e: Expr, gain = 2): VoiceParams {
  return {
    pitch: clamp(1 + 0.30 * e.smile - 0.25 * e.frown + 0.20 * e.browUp - 0.30 * e.browDown, 0.5, 2),
    rate: clamp(1 + 0.15 * e.smile - 0.15 * e.frown + 0.10 * e.browUp - 0.30 * e.browDown, 0.6, 1.5),
    volume: clamp(0.4 + 0.6 * clamp(e.jaw * gain, 0, 1), 0.3, 1),
  };
}

/** Exponential smoothing across consecutive words so the voice doesn't jitter. */
export class VoiceSmoother {
  private prev: VoiceParams | null = null;
  constructor(private alpha = 0.6) {} // weight of the new word
  next(p: VoiceParams): VoiceParams {
    const q = this.prev
      ? {
          pitch: this.prev.pitch + (p.pitch - this.prev.pitch) * this.alpha,
          rate: this.prev.rate + (p.rate - this.prev.rate) * this.alpha,
          volume: this.prev.volume + (p.volume - this.prev.volume) * this.alpha,
        }
      : p;
    this.prev = q;
    return q;
  }
  reset() { this.prev = null; }
}
