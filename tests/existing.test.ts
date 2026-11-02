import { describe, expect, it } from 'vitest';
import { ActivityDetector } from '../src/activity/detector';
import { trimToMotion } from '../src/activity/segment';
import { mapExpression, VoiceSmoother } from '../src/audio/voice';
import { ExpressionTracker } from '../src/features/expression';
import { wilson } from '../src/ui/evaltest';
import { classify } from '../src/model/classifier';
import { FEATURE_DIM, STATIC_DIM, extractStatic, IDX_WIDTH, IDX_APERTURE } from '../src/features/mouth';
import { makeWords } from './helpers';

function runDetector(sig: (t: number) => number, secs: number) {
  const d = new ActivityDetector();
  d.setBaseline(0.05);
  let n = 0;
  let seed = 11;
  const noise = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.04;
  for (let t = 0; t < secs * 1000; t += 33) if (d.push(new Float32Array(1), sig(t) + noise(), t)) n++;
  return n;
}

describe('ActivityDetector (regression)', () => {
  it('ignores idle noise', () => expect(runDetector(() => 0.05, 20)).toBe(0));
  it('finds 8 words 1.5 s apart', () => {
    expect(runDetector((t) => (t % 1500 < 500 && t > 500 && t < 12500 ? 1 : 0.05), 15)).toBe(8);
  });
  it('does not split a word on a short internal dip', () => {
    expect(runDetector((t) => (t > 500 && t < 1500 && !(t > 900 && t < 1050) ? 1 : 0.05), 3)).toBe(1);
  });
  it('ignores a 50 ms blip and low constant head motion', () => {
    expect(runDetector((t) => (t > 500 && t < 550 ? 2 : 0.05), 2)).toBe(0);
    expect(runDetector(() => 0.12, 10)).toBe(0);
  });
});

describe('trimToMotion', () => {
  it('drops idle lead-in and tail', () => {
    const frames = Array.from({ length: 60 }, (_, i) => {
      const f = new Float32Array(FEATURE_DIM);
      const v = i >= 15 && i < 45 ? 1 : 0.01;
      for (let d = 0; d < 83; d++) f[STATIC_DIM + d] = v;
      return f;
    });
    const r = trimToMotion(frames, frames.map((_, i) => i * 33));
    expect(r.frames.length).toBeGreaterThanOrEqual(30);
    expect(r.frames.length).toBeLessThanOrEqual(36);
  });
});

describe('expression to voice', () => {
  const cats = (o: Record<string, number>) => Object.entries(o).map(([categoryName, score]) => ({ categoryName, score }));
  const face = (o: Record<string, number>) => {
    const tr = new ExpressionTracker();
    tr.captureNeutral(1500, 0, () => {});
    for (let t = 0; t <= 1600; t += 33) tr.push(cats({ mouthSmileLeft: 0.06, mouthSmileRight: 0.06 }), t);
    for (let t = 2000; t <= 2600; t += 33) tr.push(cats({ jawOpen: 0.35, ...o }), t);
    return mapExpression(tr.window(2000, 2600), 2);
  };
  it('smile raises and frown lowers pitch; brow down slows; big mouth is louder', () => {
    const n = face({ mouthSmileLeft: 0.06, mouthSmileRight: 0.06 });
    const s = face({ mouthSmileLeft: 0.7, mouthSmileRight: 0.7 });
    const f = face({ mouthFrownLeft: 0.6, mouthFrownRight: 0.6 });
    const b = face({ browDownLeft: 0.7, browDownRight: 0.7 });
    const big = face({ jawOpen: 0.7 });
    expect(s.pitch).toBeGreaterThan(n.pitch);
    expect(f.pitch).toBeLessThan(n.pitch);
    expect(b.rate).toBeLessThan(n.rate);
    expect(big.volume).toBeGreaterThan(n.volume);
  });
  it('smooths across words', () => {
    const sm = new VoiceSmoother(0.6);
    sm.next({ pitch: 1.2, rate: 1, volume: 0.8 });
    const after = sm.next({ pitch: 0.8, rate: 1, volume: 0.8 });
    expect(after.pitch).toBeGreaterThan(0.8);
    expect(after.pitch).toBeLessThan(1.2);
  });
});

describe('pose normalization', () => {
  it('is invariant to head scale and position', () => {
    const W = 640, H = 480;
    const make = (open: number, s: number, tx: number, ty: number) => {
      const pts: [number, number, number][] = Array.from({ length: 478 }, () => [0, 0.5, 0]);
      pts[33] = [-0.5, 0, 0]; pts[263] = [0.5, 0, 0]; pts[152] = [0, 1.3, 0.05];
      pts[61] = [-0.3, 0.8, 0.05]; pts[291] = [0.3, 0.8, 0.05]; pts[0] = [0, 0.75, 0.1]; pts[17] = [0, 0.85 + open, 0.1];
      pts[13] = [0, 0.78, 0.1]; pts[14] = [0, 0.8 + open, 0.1];
      return pts.map(([x, y, z]) => ({ x: (tx + x * s) / W, y: (ty + y * s) / H, z: (z * s) / W, visibility: 1 }));
    };
    const a = extractStatic(make(0.1, 100, 320, 200), W, H)!;
    const b = extractStatic(make(0.1, 60, 200, 150), W, H)!;
    expect(b[IDX_WIDTH]).toBeCloseTo(a[IDX_WIDTH], 4);
    expect(b[IDX_APERTURE]).toBeCloseTo(a[IDX_APERTURE], 4);
  });
});

describe('classifier and accuracy math', () => {
  it('classifies synthetic words and exposes a full ranking', async () => {
    const w = makeWords(8);
    const ev = await w.train();
    let ok = 0;
    for (let i = 0; i < 8; i++) {
      const u = w.utter(i);
      const p = classify(ev.model, u.frames, u.times);
      if (p.label === w.vocab[i]) ok++;
      expect(p.ranked).toHaveLength(8);
      expect(p.ranked[0].word).toBe(p.best);
    }
    expect(ok).toBe(8);
  });
  it('wilson interval brackets the estimate', () => {
    const [lo, hi] = wilson(80, 100);
    expect(lo).toBeGreaterThan(0.7);
    expect(hi).toBeLessThan(0.9);
    expect(wilson(0, 0)).toEqual([0, 0]);
  });
});
