import { GEOM_DIM, STATIC_DIM } from '../features/mouth';

/**
 * Trim a recorded window to the part where the lips actually move, so segments from the
 * hold-to-record button and from the automatic detector look alike (no idle lead-in/tail).
 * Uses the velocity half of each feature vector. Returns the input unchanged if trimming
 * would leave fewer than 8 frames.
 */
export function trimToMotion(frames: Float32Array[], times: number[]): { frames: Float32Array[]; times: number[] } {
  const e = frames.map((f) => {
    let s = 0;
    for (let i = 0; i < GEOM_DIM; i++) s += f[STATIC_DIM + i] ** 2;
    return Math.sqrt(s / GEOM_DIM);
  });
  const smooth = e.map((_, i) => (e[Math.max(0, i - 1)] + e[i] + e[Math.min(e.length - 1, i + 1)]) / 3);
  const peak = Math.max(...smooth, 0);
  if (peak <= 0) return { frames, times };
  const thr = Math.max(0.15 * peak, 0.03);
  let a = smooth.findIndex((v) => v > thr);
  let b = smooth.length - 1; while (b > 0 && smooth[b] <= thr) b--;
  a = Math.max(0, a - 2); b = Math.min(frames.length - 1, b + 2);
  if (b - a + 1 < 8) return { frames, times };
  return { frames: frames.slice(a, b + 1), times: times.slice(a, b + 1) };
}
