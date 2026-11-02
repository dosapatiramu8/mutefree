import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { OUTER_LIP, INNER_LIP } from '../features/mouth';

export function drawOverlay(canvas: HTMLCanvasElement, lm: NormalizedLandmark[] | null, full: boolean) {
  const c = canvas.getContext('2d')!;
  c.clearRect(0, 0, canvas.width, canvas.height);
  if (!lm) return;
  const W = canvas.width, H = canvas.height;
  if (full) {
    c.fillStyle = 'rgba(120,200,255,0.55)';
    for (const p of lm) c.fillRect(p.x * W - 1, p.y * H - 1, 2, 2);
  }
  const loop = (idx: number[], color: string) => {
    c.beginPath(); c.strokeStyle = color; c.lineWidth = 2;
    idx.forEach((i, k) => { const x = lm[i].x * W, y = lm[i].y * H; if (k) c.lineTo(x, y); else c.moveTo(x, y); });
    c.closePath(); c.stroke();
  };
  loop(OUTER_LIP, '#ff8a5c'); loop(INNER_LIP, '#5fd38a');
}
