import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { INNER_LIP, OUTER_LIP, APP_DIM } from './mouth';

const W = 48, H = 28;

/**
 * Inner-mouth appearance: what fraction of the visible mouth interior is dark cavity,
 * bright low-saturation (teeth), or red-dominant (tongue/lips), and where the red sits
 * vertically. Thresholds are relative to the face's mean brightness so they adapt to
 * lighting; the classifier then learns what the numbers mean for this user.
 * Returns zeros when the mouth is too closed to see inside.
 */
export class MouthAppearance {
  private c = document.createElement('canvas');
  private ctx: CanvasRenderingContext2D;
  constructor() {
    this.c.width = W; this.c.height = H;
    this.ctx = this.c.getContext('2d', { willReadFrequently: true })!;
  }

  compute(video: HTMLVideoElement, lm: NormalizedLandmark[], faceLuma: number): Float32Array {
    const out = new Float32Array(APP_DIM);
    const vw = video.videoWidth, vh = video.videoHeight;
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const i of OUTER_LIP) {
      x0 = Math.min(x0, lm[i].x); x1 = Math.max(x1, lm[i].x);
      y0 = Math.min(y0, lm[i].y); y1 = Math.max(y1, lm[i].y);
    }
    const padX = (x1 - x0) * 0.15, padY = (y1 - y0) * 0.15;
    x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
    const sx = Math.max(0, x0 * vw), sy = Math.max(0, y0 * vh);
    const sw = Math.min(vw - sx, (x1 - x0) * vw), sh = Math.min(vh - sy, (y1 - y0) * vh);
    if (sw < 4 || sh < 4) return out;
    this.ctx.drawImage(video, sx, sy, sw, sh, 0, 0, W, H);
    const px = this.ctx.getImageData(0, 0, W, H).data;

    // inner-lip polygon in crop pixel coordinates
    const poly = INNER_LIP.map((i) => [(lm[i].x * vw - sx) / sw * W, (lm[i].y * vh - sy) / sh * H]);
    let minY = H, maxY = 0;
    for (const p of poly) { minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); }
    let n = 0, dark = 0, teeth = 0, red = 0, redYsum = 0;
    for (let y = Math.max(0, Math.floor(minY)); y <= Math.min(H - 1, Math.ceil(maxY)); y++) {
      for (let x = 0; x < W; x++) {
        if (!inside(poly, x + 0.5, y + 0.5)) continue;
        const k = (y * W + x) * 4, r = px[k], g = px[k + 1], b = px[k + 2];
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        n++;
        if (luma < 0.45 * faceLuma) dark++;
        else if (luma > 1.1 * faceLuma && mx > 0 && (mx - mn) / mx < 0.25) teeth++;
        if (r > g * 1.25 && r > b * 1.15) { red++; redYsum += (y - minY) / Math.max(maxY - minY, 1); }
      }
    }
    if (n < 8) return out;
    out[0] = dark / n; out[1] = teeth / n; out[2] = red / n; out[3] = red ? redYsum / red : 0;
    return out;
  }
}

function inside(poly: number[][], x: number, y: number): boolean {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
