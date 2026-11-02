import { classify, type Model } from '../model/classifier';
import { trimToMotion } from '../activity/segment';
import type { Segment } from '../activity/detector';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export type Outcome = 'correct' | 'wrong' | 'rejected' | 'missed';
export interface EvalRow {
  i: number; truth: string; predicted: string; closest: string; conf: number; distance: number;
  outcome: Outcome; processMs: number | null; durationMs: number | null;
}

const GAP_MS = 1200, TIMEOUT_MS = 6000, LEAD_MS = 2000;

/** Wilson 95% interval for a proportion. */
export function wilson(k: number, n: number): [number, number] {
  if (!n) return [0, 0];
  const z = 1.96, p = k / n, d = 1 + (z * z) / n;
  const c = p + (z * z) / (2 * n), m = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [(c - m) / d, (c + m) / d];
}

/**
 * Cued live-accuracy test. The app names a word, the user mouths it, and the normal
 * pipeline (detector -> trim -> classifier -> threshold) decides. Nothing is spoken and
 * nothing is hidden: wrong words, "?" results and words the detector never saw are all
 * counted against accuracy. Ground truth is the cue, so a misread cue counts as wrong.
 */
export class EvalTest {
  active = false;
  rows: EvalRow[] = [];
  private cues: string[] = [];
  private phase: 'idle' | 'gap' | 'prompt' = 'idle';
  private until = 0;
  private promptAt = 0;
  private meta: Record<string, unknown> = {};

  constructor(
    private getModel: () => Model | null,
    private getThreshold: () => number,
    private canStart: () => string | null,
    private getMeta: () => Record<string, unknown>,
  ) {
    $('evStart').onclick = () => this.start();
    $('evStop').onclick = () => this.finish(true);
    $('evCsv').onclick = () => this.download('csv');
    $('evJson').onclick = () => this.download('json');
  }

  private start() {
    const why = this.canStart();
    const model = this.getModel();
    if (why || !model) { $('evOut').textContent = why ?? 'Calibrate first.'; return; }
    const n = Math.min(Math.max(+$<HTMLInputElement>('evN').value || 100, 8), 300);
    // balanced: shuffled rounds so each word is cued about equally often
    this.cues = [];
    while (this.cues.length < n) {
      const round = [...model.vocab].sort(() => Math.random() - 0.5);
      this.cues.push(...round.slice(0, n - this.cues.length));
    }
    this.rows = []; this.active = true; this.phase = 'gap'; this.until = performance.now() + LEAD_MS;
    this.meta = { ...this.getMeta(), threshold: this.getThreshold(), vocab: model.vocab, featureSet: model.featureSet, cued: n, startedAt: new Date().toISOString() };
    ($('evStart') as HTMLButtonElement).disabled = true; ($('evStop') as HTMLButtonElement).disabled = false;
    ($('evCsv') as HTMLButtonElement).disabled = true; ($('evJson') as HTMLButtonElement).disabled = true;
    $('evOut').textContent = 'Get ready…'; this.render('…');
  }

  /** Call every frame. */
  tick(t: number) {
    if (!this.active) return;
    if (this.phase === 'gap' && t >= this.until) {
      if (this.rows.length >= this.cues.length) { this.finish(false); return; }
      this.phase = 'prompt'; this.promptAt = t; this.until = t + TIMEOUT_MS;
      this.render(this.cues[this.rows.length]);
    } else if (this.phase === 'prompt' && t >= this.until) {
      this.log({ outcome: 'missed', predicted: '–', closest: '–', conf: 0, distance: NaN, processMs: null, durationMs: null }, t);
    }
  }

  onSegment(seg: Segment, detectedAt: number) {
    const model = this.getModel();
    if (!this.active || this.phase !== 'prompt' || !model) return;
    if (seg.startMs < this.promptAt - 200) return; // movement from before the cue
    const tr = trimToMotion(seg.frames, seg.times);
    const c0 = performance.now();
    const p = classify(model, tr.frames, tr.times);
    const truth = this.cues[this.rows.length];
    const accepted = p.label !== null && p.conf >= this.getThreshold();
    const outcome: Outcome = !accepted ? 'rejected' : p.label === truth ? 'correct' : 'wrong';
    this.log({
      outcome, predicted: accepted ? p.label! : '?', closest: p.best, conf: p.conf, distance: p.d1,
      processMs: performance.now() - seg.endMs, durationMs: tr.times[tr.times.length - 1] - tr.times[0],
    }, detectedAt + (performance.now() - c0));
  }

  private log(r: Omit<EvalRow, 'i' | 'truth'>, t: number) {
    this.rows.push({ i: this.rows.length + 1, truth: this.cues[this.rows.length], ...r });
    this.phase = 'gap'; this.until = t + GAP_MS;
    this.render('…');
    const n = this.rows.length;
    $('evOut').textContent = `${n}/${this.cues.length} cued · last: said "${r.predicted}" for "${this.rows[n - 1].truth}" (${r.outcome})`;
  }

  private render(word: string) {
    $('evWord').textContent = this.active ? word : '–';
    ($('evProg') as HTMLProgressElement).value = this.cues.length ? this.rows.length / this.cues.length : 0;
  }

  private finish(stopped: boolean) {
    this.active = false; this.phase = 'idle';
    ($('evStart') as HTMLButtonElement).disabled = false; ($('evStop') as HTMLButtonElement).disabled = true;
    ($('evCsv') as HTMLButtonElement).disabled = this.rows.length === 0; ($('evJson') as HTMLButtonElement).disabled = this.rows.length === 0;
    this.render('–');
    $('evOut').textContent = this.summary() + (stopped ? '\n(stopped early)' : '');
  }

  summary(): string {
    const n = this.rows.length;
    if (!n) return 'No results.';
    const c = (o: Outcome) => this.rows.filter((r) => r.outcome === o).length;
    const ok = c('correct'), wrong = c('wrong'), rej = c('rejected'), miss = c('missed');
    const [lo, hi] = wilson(ok, n);
    const spoken = ok + wrong;
    const lat = this.rows.map((r) => r.processMs).filter((x): x is number => x !== null).sort((a, b) => a - b);
    const q = (p: number) => (lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))].toFixed(0) : '–');
    const per = new Map<string, [number, number]>();
    for (const r of this.rows) { const e = per.get(r.truth) ?? [0, 0]; e[1]++; if (r.outcome === 'correct') e[0]++; per.set(r.truth, e); }
    const worst = [...per.entries()].sort((a, b) => a[1][0] / a[1][1] - b[1][0] / b[1][1]).slice(0, 4)
      .map(([w, [k, t]]) => `${w} ${k}/${t}`).join(', ');
    return [
      `Cued words: ${n}${n < 100 ? ' (fewer than the 100 needed for a solid estimate)' : ''}`,
      `Correct (spoken as the cued word): ${ok}/${n} = ${(100 * ok / n).toFixed(1)}%  (95% CI ${(100 * lo).toFixed(0)}–${(100 * hi).toFixed(0)}%)`,
      `Wrong word spoken: ${wrong} · Shown as "?" (not spoken): ${rej} · Movement never detected: ${miss}`,
      `When a word is spoken it is right ${spoken ? (100 * ok / spoken).toFixed(1) : '–'}% of the time; ${(100 * spoken / n).toFixed(0)}% of cues produce speech`,
      `Mouth-stop to decision (before voice start): median ${q(0.5)} ms, p90 ${q(0.9)} ms`,
      `Weakest words: ${worst}`,
      `Threshold ${(+this.meta.threshold!).toFixed(2)} · features ${this.meta.featureSet} · ${(this.meta.vocab as string[]).length} words`,
    ].join('\n');
  }

  private download(kind: 'csv' | 'json') {
    const body = kind === 'json'
      ? JSON.stringify({ meta: this.meta, summary: this.summary(), rows: this.rows }, null, 1)
      : ['i,truth,predicted,closest,confidence,distance,outcome,process_ms,duration_ms',
        ...this.rows.map((r) => [r.i, r.truth, r.predicted, r.closest, r.conf.toFixed(3), Number.isNaN(r.distance) ? '' : r.distance.toFixed(3), r.outcome, r.processMs?.toFixed(0) ?? '', r.durationMs?.toFixed(0) ?? ''].join(','))].join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([body], { type: kind === 'json' ? 'application/json' : 'text/csv' }));
    a.download = `mutefree-accuracy-${Date.now()}.${kind}`; a.click();
  }
}
