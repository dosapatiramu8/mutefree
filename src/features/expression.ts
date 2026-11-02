export interface Expr {
  smile: number;     // 0..1, above neutral
  frown: number;     // 0..1, above neutral
  browUp: number;    // 0..1
  browDown: number;  // 0..1
  jaw: number;       // 0..1 jaw opening (raw, intensity proxy)
}
export const NEUTRAL_EXPR: Expr = { smile: 0, frown: 0, browUp: 0, browDown: 0, jaw: 0 };

interface Frame extends Expr { t: number; }
type Cat = { categoryName: string; score: number };

const clamp01 = (x: number) => Math.min(Math.max(x, 0), 1);

/**
 * Reads MediaPipe blendshapes into a handful of expression signals, keeps a few
 * seconds of history so a word can be judged over the time it was mouthed, and
 * subtracts a per-user neutral baseline (so a resting smile line doesn't read as a smile).
 */
export class ExpressionTracker {
  baseline: Expr = { ...NEUTRAL_EXPR };
  private frames: Frame[] = [];
  private live_: Expr = { ...NEUTRAL_EXPR };
  private capture: { until: number; sum: Expr; n: number; done: () => void } | null = null;
  private tauMs = 120;
  private lastT = 0;

  /** Raw (pre-baseline) signals from one blendshape frame. */
  static raw(cats: Cat[]): Expr {
    const m = new Map(cats.map((c) => [c.categoryName, c.score]));
    const g = (k: string) => m.get(k) ?? 0;
    return {
      smile: (g('mouthSmileLeft') + g('mouthSmileRight')) / 2,
      frown: (g('mouthFrownLeft') + g('mouthFrownRight')) / 2,
      browUp: Math.max(g('browInnerUp'), (g('browOuterUpLeft') + g('browOuterUpRight')) / 2),
      browDown: (g('browDownLeft') + g('browDownRight')) / 2,
      jaw: g('jawOpen'),
    };
  }

  push(cats: Cat[], t: number) {
    const r = ExpressionTracker.raw(cats);
    if (this.capture) {
      for (const k of Object.keys(r) as (keyof Expr)[]) this.capture.sum[k] += r[k];
      this.capture.n++;
      if (t >= this.capture.until) {
        const { sum, n, done } = this.capture; this.capture = null;
        for (const k of Object.keys(sum) as (keyof Expr)[]) this.baseline[k] = k === 'jaw' ? 0 : sum[k] / n;
        done();
      }
    }
    const e = this.adjust(r);
    const a = this.lastT ? 1 - Math.exp(-Math.max(t - this.lastT, 1) / this.tauMs) : 1;
    this.lastT = t;
    for (const k of Object.keys(e) as (keyof Expr)[]) this.live_[k] += (e[k] - this.live_[k]) * a;
    this.frames.push({ ...e, t });
    while (this.frames.length && this.frames[0].t < t - 8000) this.frames.shift();
  }

  /** Subtract baseline and rescale so the remaining headroom still maps to 0..1. */
  private adjust(r: Expr): Expr {
    const f = (v: number, b: number) => clamp01((v - b) / Math.max(1 - b, 0.2));
    return {
      smile: f(r.smile, this.baseline.smile), frown: f(r.frown, this.baseline.frown),
      browUp: f(r.browUp, this.baseline.browUp), browDown: f(r.browDown, this.baseline.browDown),
      jaw: clamp01(r.jaw),
    };
  }

  /** Smoothed current expression (for the live meters). */
  get live(): Expr { return { ...this.live_ }; }

  /**
   * Expression over [startMs, endMs]: mean for smile/frown/brows (held expressions),
   * 80th percentile for jaw (how big the mouth movement got). Falls back to the live
   * value if no frames fall in the window.
   */
  window(startMs: number, endMs: number): Expr {
    const fs = this.frames.filter((f) => f.t >= startMs && f.t <= endMs);
    if (!fs.length) return this.live;
    const mean = (k: keyof Expr) => fs.reduce((a, f) => a + f[k], 0) / fs.length;
    const jaws = fs.map((f) => f.jaw).sort((a, b) => a - b);
    return {
      smile: mean('smile'), frown: mean('frown'), browUp: mean('browUp'), browDown: mean('browDown'),
      jaw: jaws[Math.min(jaws.length - 1, Math.floor(0.8 * jaws.length))],
    };
  }

  /** Average the next `ms` of frames as the user's neutral face. */
  captureNeutral(ms: number, now: number, done: () => void) {
    this.capture = { until: now + ms, sum: { ...NEUTRAL_EXPR }, n: 0, done };
  }
  get capturing() { return this.capture !== null; }
}
