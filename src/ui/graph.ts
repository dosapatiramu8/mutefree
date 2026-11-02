export interface Series { name: string; cssVar: string; scale: number; }

/** Rolling time-series plot; each series has a fixed scale (value that maps to full height). */
export class Graph {
  private ctx: CanvasRenderingContext2D;
  private data: number[][];
  constructor(private canvas: HTMLCanvasElement, private series: Series[], private capacity = 150) {
    this.ctx = canvas.getContext('2d')!;
    this.data = series.map(() => []);
  }
  push(values: number[]) {
    values.forEach((v, i) => { const d = this.data[i]; d.push(v); if (d.length > this.capacity) d.shift(); });
  }
  draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(w * dpr)) { this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr); }
    const c = this.ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    const css = getComputedStyle(this.canvas);
    this.series.forEach((s, si) => {
      const d = this.data[si]; if (d.length < 2) return;
      c.beginPath(); c.strokeStyle = css.getPropertyValue(s.cssVar); c.lineWidth = 1.8;
      d.forEach((v, i) => {
        const x = (i / (this.capacity - 1)) * w, y = h - Math.min(v / s.scale, 1) * (h - 6) - 3;
        if (i) c.lineTo(x, y); else c.moveTo(x, y);
      });
      c.stroke();
    });
  }
}
