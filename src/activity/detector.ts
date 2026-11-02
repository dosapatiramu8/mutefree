export interface Segment {
  startMs: number;
  endMs: number;
  times: number[];
  frames: Float32Array[];
}

export interface DetectorOptions {
  startMult: number;   // start threshold = baseline * startMult (but never below absMin)
  stopMult: number;    // stop threshold  = baseline * stopMult
  absMin: number;      // floor for the start threshold, in energy units
  minStartMs: number;  // energy must stay above start threshold this long to begin
  hangoverMs: number;  // energy must stay below stop threshold this long to end
  minSegmentMs: number;
  prerollMs: number;
  smoothTauMs: number;
}

export const DEFAULTS: DetectorOptions = {
  startMult: 4, stopMult: 2, absMin: 0.15,
  minStartMs: 80, hangoverMs: 350, minSegmentMs: 150, prerollMs: 150, smoothTauMs: 60,
};

/**
 * Hysteresis speech-activity detector over the lip motion-energy signal.
 * Thresholds are relative to an idle-noise baseline, which is measured by the
 * "still" baseline task and then adapts slowly while idle.
 */
export class ActivityDetector {
  state: 'IDLE' | 'ACTIVE' = 'IDLE';
  smooth = 0;
  baseline = 0.05;
  private lastT = 0;
  private aboveSince: number | null = null;
  private belowSince: number | null = null;
  private startMs = 0;
  private buf: { t: number; v: Float32Array }[] = [];

  constructor(public opts: DetectorOptions = { ...DEFAULTS }) {}

  get startThreshold() { return Math.max(this.baseline * this.opts.startMult, this.opts.absMin); }
  get stopThreshold() { return Math.max(this.baseline * this.opts.stopMult, this.opts.absMin * 0.5); }

  setBaseline(b: number) { this.baseline = Math.max(b, 1e-3); }

  reset() {
    this.state = 'IDLE'; this.aboveSince = this.belowSince = null; this.buf = []; this.smooth = 0; this.lastT = 0;
  }

  /** Feed one frame. Returns a finished segment when one ends. */
  push(vec: Float32Array, energy: number, t: number): Segment | null {
    const dt = this.lastT ? Math.max(t - this.lastT, 1) : 33;
    this.lastT = t;
    const a = 1 - Math.exp(-dt / this.opts.smoothTauMs);
    this.smooth += (energy - this.smooth) * a;

    this.buf.push({ t, v: vec });
    if (this.state === 'IDLE') {
      const cutoff = t - this.opts.prerollMs - this.opts.minStartMs - 100;
      while (this.buf.length && this.buf[0].t < cutoff) this.buf.shift();
    }

    if (this.state === 'IDLE') {
      if (Math.min(this.smooth, energy) < this.startThreshold) {
        // slow baseline adaptation (tau ~ 3 s), only on quiet frames
        this.baseline += (this.smooth - this.baseline) * (1 - Math.exp(-dt / 3000));
        this.baseline = Math.max(this.baseline, 1e-3);
        this.aboveSince = null;
        return null;
      }
      this.aboveSince ??= t;
      if (t - this.aboveSince >= this.opts.minStartMs) {
        this.state = 'ACTIVE';
        this.startMs = this.aboveSince - this.opts.prerollMs;
        this.belowSince = null;
      }
      return null;
    }

    // ACTIVE
    if (this.smooth < this.stopThreshold) {
      this.belowSince ??= t;
      if (t - this.belowSince >= this.opts.hangoverMs) {
        const endMs = this.belowSince;
        const seg = this.finish(endMs);
        this.state = 'IDLE'; this.aboveSince = this.belowSince = null; this.buf = [];
        return seg;
      }
    } else {
      this.belowSince = null;
    }
    return null;
  }

  private finish(endMs: number): Segment | null {
    if (endMs - this.startMs < this.opts.minSegmentMs) return null;
    const used = this.buf.filter((b) => b.t >= this.startMs && b.t <= endMs);
    return { startMs: this.startMs, endMs, times: used.map((b) => b.t), frames: used.map((b) => b.v) };
  }
}
