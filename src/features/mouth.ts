import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

// MediaPipe face-mesh lip loops (20 points each, ordered around the contour).
export const OUTER_LIP = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
export const INNER_LIP = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];

const EYE_R = 33, EYE_L = 263, CHIN = 152;
const CORNER_R = 61, CORNER_L = 291, TOP_OUT = 0, BOT_OUT = 17, TOP_IN = 13, BOT_IN = 14;

/**
 * Layout: [width, outerH, aperture, outer(20x2), inner(20x2) | tongueOut | dark, teeth, red, redY]
 * The first GEOM_DIM entries are pure lip geometry (used for the activity energy);
 * the rest are optional extras that the calibration step evaluates on/off.
 */
export const GEOM_DIM = 3 + OUTER_LIP.length * 2 + INNER_LIP.length * 2;
export const IDX_TONGUE = GEOM_DIM;
export const IDX_APP = GEOM_DIM + 1;
export const APP_DIM = 4;
export const STATIC_DIM = GEOM_DIM + 1 + APP_DIM;
export const FEATURE_DIM = STATIC_DIM * 2; // static + delta
export const IDX_WIDTH = 0, IDX_OUTER_H = 1, IDX_APERTURE = 2;

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a: V3, b: V3) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a: V3, b: V3): V3 => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: V3): V3 => { const n = norm(a) || 1; return [a[0]/n, a[1]/n, a[2]/n]; };

/**
 * Head-pose and scale normalization. Builds a face-local frame from the two outer
 * eye corners (x axis) and the chin (y hint), projects lip points into it, and
 * divides by inter-ocular distance. Output is invariant to head translation,
 * in-plane/out-of-plane rotation (approximately) and distance to the camera.
 */
export function extractStatic(
  lm: NormalizedLandmark[], w: number, h: number, tongueOut = 0, appearance?: ArrayLike<number>,
): Float32Array | null {
  if (lm.length < 468) return null;
  const P = (i: number): V3 => [lm[i].x * w, lm[i].y * h, lm[i].z * w];
  const eR = P(EYE_R), eL = P(EYE_L);
  const iod = norm(sub(eL, eR));
  if (iod < 1e-3) return null;
  const origin: V3 = [(eR[0]+eL[0])/2, (eR[1]+eL[1])/2, (eR[2]+eL[2])/2];
  const ex = unit(sub(eL, eR));
  const yHint = sub(P(CHIN), origin);
  const proj = dot(yHint, ex);
  const ey = unit(sub(yHint, [ex[0]*proj, ex[1]*proj, ex[2]*proj]));
  const ez = cross(ex, ey);
  const to = (i: number): V3 => {
    const d = sub(P(i), origin);
    return [dot(d, ex)/iod, dot(d, ey)/iod, dot(d, ez)/iod];
  };

  const out = new Float32Array(STATIC_DIM);
  out[IDX_WIDTH] = norm(sub(to(CORNER_L), to(CORNER_R)));
  out[IDX_OUTER_H] = norm(sub(to(TOP_OUT), to(BOT_OUT)));
  out[IDX_APERTURE] = norm(sub(to(TOP_IN), to(BOT_IN)));
  let k = 3;
  for (const i of OUTER_LIP) { const p = to(i); out[k++] = p[0]; out[k++] = p[1]; }
  for (const i of INNER_LIP) { const p = to(i); out[k++] = p[0]; out[k++] = p[1]; }
  out[IDX_TONGUE] = tongueOut;
  if (appearance) for (let a = 0; a < APP_DIM; a++) out[IDX_APP + a] = appearance[a];
  return out;
}

/** Adds smoothed per-second velocity features; also exposes a scalar motion energy. */
export class FeatureStream {
  private prev: Float32Array | null = null;
  private prevT = 0;
  private vel = new Float32Array(STATIC_DIM);
  constructor(private smoothing = 0.5) {}

  reset() { this.prev = null; this.vel.fill(0); }

  push(stat: Float32Array, tMs: number): { vec: Float32Array; energy: number } {
    const vec = new Float32Array(FEATURE_DIM);
    vec.set(stat, 0);
    let sq = 0;
    if (this.prev) {
      const dt = Math.max((tMs - this.prevT) / 1000, 1e-3);
      for (let i = 0; i < STATIC_DIM; i++) {
        const v = (stat[i] - this.prev[i]) / dt;
        this.vel[i] = this.smoothing * this.vel[i] + (1 - this.smoothing) * v;
        if (i < GEOM_DIM) sq += this.vel[i] * this.vel[i];
      }
    }
    vec.set(this.vel, STATIC_DIM);
    this.prev = stat; this.prevT = tMs;
    return { vec, energy: Math.sqrt(sq / GEOM_DIM) };
  }
}
