import type { ActivityDetector, Segment } from './detector';

export interface TaskDef {
  id: string;
  title: string;
  prompt: string;
  durationMs: number;
  /** null = not scored (baseline). 0 = no segments should fire. n = user is asked to mouth n words. */
  expected: number | null;
}

export const TASKS: TaskDef[] = [
  { id: 'baseline', title: '0 · Baseline', prompt: 'Sit still, mouth closed, face the camera. Measuring idle noise.', durationMs: 6000, expected: null },
  { id: 'still', title: '1 · Still', prompt: 'Sit still, mouth closed. (Should trigger nothing.)', durationMs: 15000, expected: 0 },
  { id: 'head', title: '2 · Head moves only', prompt: 'Mouth closed. Nod, turn left/right, tilt, lean in and out. (Should trigger nothing.)', durationMs: 15000, expected: 0 },
  { id: 'mouth', title: '3 · Mouthing, head still', prompt: 'Mouth 8 words silently, about 1.5 s apart (e.g. yes, no, hello, water...). Head still.', durationMs: 15000, expected: 8 },
  { id: 'mouth_head', title: '4 · Mouthing + head moves', prompt: 'Mouth 8 words, 1.5 s apart, while gently moving your head.', durationMs: 15000, expected: 8 },
];

export interface TaskResult {
  id: string;
  title: string;
  expected: number | null;
  detected: number;
  durationS: number;
  falseTriggersPerMin: number | null;
  segmentDurationsMs: number[];
  noFaceFrames: number;
  frames: number;
  energyMedian: number;
  energyP95: number;
  thresholds: { start: number; stop: number; baseline: number };
  trace: [number, number, number][]; // [tRelMs, energy, active(0/1)]
}

const q = (a: number[], p: number) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};

type Phase = 'idle' | 'countdown' | 'run' | 'done';

export class Harness {
  phase: Phase = 'idle';
  results: TaskResult[] = [];
  private idx = 0;
  private phaseStart = 0;
  private energies: number[] = [];
  private segs: Segment[] = [];
  private noFace = 0;
  private frames = 0;
  private trace: [number, number, number][] = [];

  constructor(private det: ActivityDetector, private onText: (title: string, body: string) => void, private onDone: () => void) {}

  start(t: number) {
    this.results = []; this.idx = 0; this.begin('countdown', t);
  }

  private begin(p: Phase, t: number) {
    this.phase = p; this.phaseStart = t;
    if (p === 'countdown') {
      this.energies = []; this.segs = []; this.noFace = 0; this.frames = 0; this.trace = [];
      this.det.reset();
    }
  }

  /** Call once per processed frame. */
  tick(t: number, energy: number, hasFace: boolean, seg: Segment | null) {
    if (this.phase === 'idle' || this.phase === 'done') return;
    const task = TASKS[this.idx];
    const el = t - this.phaseStart;

    if (this.phase === 'countdown') {
      const left = Math.ceil((3000 - el) / 1000);
      this.onText(task.title, `Get ready… ${Math.max(left, 1)}\n${task.prompt}`);
      if (el >= 3000) this.begin('run', t);
      return;
    }

    this.frames++;
    if (!hasFace) this.noFace++;
    else {
      this.energies.push(energy);
      this.trace.push([Math.round(el), +energy.toFixed(4), this.det.state === 'ACTIVE' ? 1 : 0]);
    }
    if (seg && task.expected !== null) this.segs.push(seg);

    const left = Math.max(0, Math.ceil((task.durationMs - el) / 1000));
    const detectedNow = this.segs.length + (this.det.state === 'ACTIVE' ? 1 : 0);
    this.onText(task.title, `${left}s left · segments so far: ${task.expected === null ? '–' : detectedNow}\n${task.prompt}`);

    if (el >= task.durationMs) {
      if (task.id === 'baseline') this.det.setBaseline(q(this.energies, 0.9));
      this.results.push({
        id: task.id, title: task.title, expected: task.expected,
        detected: task.expected === null ? 0 : this.segs.length,
        durationS: task.durationMs / 1000,
        falseTriggersPerMin: task.expected === 0 ? (this.segs.length / (task.durationMs / 60000)) : null,
        segmentDurationsMs: this.segs.map((s) => Math.round(s.endMs - s.startMs)),
        noFaceFrames: this.noFace, frames: this.frames,
        energyMedian: +q(this.energies, 0.5).toFixed(4), energyP95: +q(this.energies, 0.95).toFixed(4),
        thresholds: { start: +this.det.startThreshold.toFixed(4), stop: +this.det.stopThreshold.toFixed(4), baseline: +this.det.baseline.toFixed(4) },
        trace: this.trace,
      });
      this.idx++;
      if (this.idx >= TASKS.length) { this.phase = 'done'; this.onDone(); } else this.begin('countdown', t);
    }
  }

  summary(): string {
    return this.results.map((r) => {
      if (r.expected === null) return `${r.title}: idle noise p95 = ${r.energyP95}, baseline set to ${r.thresholds.baseline}`;
      const ok = r.expected === 0
        ? `${r.detected} false triggers (${r.falseTriggersPerMin!.toFixed(1)}/min)`
        : `${r.detected} segments detected for ${r.expected} words mouthed`;
      return `${r.title}: ${ok}; median energy ${r.energyMedian}, p95 ${r.energyP95}; no-face frames ${r.noFaceFrames}/${r.frames}`;
    }).join('\n');
  }

  json(): string {
    return JSON.stringify({ when: new Date().toISOString(), options: this.det.opts, results: this.results }, null, 1);
  }
}
