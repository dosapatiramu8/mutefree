import { defaultVocab, ambiguousPairs, visemes, QUICK_VOCAB } from '../model/vocab';
import { trimToMotion } from '../activity/segment';
import { resample, trainAndEvaluate, type Sample, type Evaluation, type Model, type FeatureSet } from '../model/classifier';
import { saveCalibration, loadCalibration, deleteCalibration } from '../model/storage';
import type { Segment } from '../activity/detector';

export interface FaceInfo {
  hasFace: boolean; cx: number; cy: number; faceH: number;
  luma: number; leftLuma: number; rightLuma: number;
}

const FULL_REPS = 5, QUICK_REPS = 3;
const MIN_WORDS = 4, MAX_WORDS = 40;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const el = (tag: string, cls?: string, text?: string) => {
  const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e;
};
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

type Mode = 'home' | 'check' | 'record' | 'train' | 'results';

export class Calibration {
  vocab: string[] = defaultVocab();
  samples: Sample[] = [];
  model: Model | null = null;
  private mode: Mode = 'home';
  private queue: string[] = [];
  private qi = 0;
  private wordStart = 0;
  private flash = '';
  private reps = FULL_REPS;
  private quickPlan = false;
  private holdFrames: { v: Float32Array; t: number }[] | null = null;
  private evals: Record<FeatureSet, Evaluation> | null = null;
  private used: FeatureSet = 'geometry';
  private savedAt = 0;

  constructor(private onModel: (m: Model | null) => void, private busy: () => boolean) {
    $('calStart').onclick = () => this.startCheck(false);
    $('calQuick').onclick = () => this.startCheck(true);
    this.bindHold();
    $('calDel').onclick = () => this.wipe();
    $('calReset').onclick = () => { this.vocab = defaultVocab(); this.renderHome(); };
    $('calAdd').onclick = () => this.addWord();
    $<HTMLInputElement>('calWord').onkeydown = (e) => { if (e.key === 'Enter') this.addWord(); };
    $('calBack').onclick = () => this.show('home');
    $('calBegin').onclick = () => {
      this.reps = this.quickPlan ? QUICK_REPS : FULL_REPS;
      if (this.quickPlan) this.vocab = [...QUICK_VOCAB];
      this.beginRecording([...this.vocab], true);
    };
    $('recUndo').onclick = () => this.undo();
    $('recSens').onclick = () => { this.onSensitive?.(); this.flash = 'Detection made more sensitive — try again'; this.renderRecord(); };
    $('recCancel').onclick = () => void this.cancel();
    $('resDone').onclick = () => this.show('home');
    this.show('home');
  }

  /** Abandon an in-progress recording and go back to the last saved state. */
  private async cancel() {
    this.samples = []; this.vocab = defaultVocab();
    await this.init(); this.show('home');
  }

  async init() {
    try {
      const saved = await loadCalibration();
      if (saved) {
        this.vocab = saved.vocab; this.samples = saved.samples; this.savedAt = saved.savedAt;
        const ev = await trainAndEvaluate(saved.samples, saved.vocab, saved.featureSet);
        this.used = saved.featureSet; this.setModel(ev.model);
      }
    } catch (e) { console.warn('could not load saved calibration', e); }
    this.renderHome();
  }

  /** True while the calibration flow owns the segments (checking, recording or training). */
  get active() { return this.mode === 'check' || this.mode === 'record' || this.mode === 'train'; }

  private setModel(m: Model | null) {
    this.model = m; this.onModel(m);
    $('vocabN').textContent = m ? String(m.vocab.length) : '0';
  }

  private count(w: string) { return this.samples.filter((s) => s.word === w).length; }

  private show(m: Mode) {
    this.mode = m;
    for (const [k, id] of [['home', 'calHome'], ['check', 'calCheck'], ['record', 'calRec'], ['train', 'calTrain'], ['results', 'calRes']] as const)
      $(id).hidden = k !== m;
    $('calState').textContent = this.model ? `calibrated · ${this.model.vocab.length} words` : 'not calibrated';
    if (m === 'home') this.renderHome();
  }

  // ---------- home / vocabulary ----------
  private addWord() {
    const inp = $<HTMLInputElement>('calWord');
    const w = inp.value.trim().toLowerCase().replace(/[^a-z' ]/g, '');
    if (!w || this.vocab.some((v) => v.toLowerCase() === w) || this.vocab.length >= MAX_WORDS) return;
    this.vocab.push(w); inp.value = ''; this.renderHome();
  }

  private renderHome() {
    const list = $('vocabList'); list.replaceChildren();
    for (const w of this.vocab) {
      const chip = el('span', 'chip'); chip.append(el('span', undefined, w), el('small', undefined, ' ' + visemes(w)));
      const x = el('button', 'x', '×'); x.title = 'Remove'; x.onclick = () => { this.vocab = this.vocab.filter((v) => v !== w); this.renderHome(); };
      chip.append(x); list.append(chip);
    }
    const amb = ambiguousPairs(this.vocab);
    const a = $('ambig'); a.replaceChildren();
    a.append(el('div', undefined, `Vocabulary size: ${this.vocab.length} words. Mouth-shape codes are a rough guess from spelling.`));
    if (amb.length) {
      a.append(el('div', 'warn', `${amb.length} look-alike pair(s) likely to be confused: ` +
        amb.map((p) => `${p.a}/${p.b}${p.level === 'identical' ? ' (identical)' : ''}`).join(', ') + '. Swap one word of a pair if the confusion matrix confirms it.'));
    } else a.append(el('div', undefined, 'No look-alike pairs flagged by the spelling heuristic.'));

    const st = $('calStatus');
    if (this.model) {
      const ev = this.evals?.[this.used];
      st.textContent = `Calibrated on ${this.samples.length} samples for ${this.model.vocab.length} words` +
        (ev ? ` · cross-validated accuracy ${pct(ev.accuracy)} (leave-one-out on your own samples, not live use)` : '') +
        ` · features: ${this.used}` + (this.savedAt ? ` · saved ${new Date(this.savedAt).toLocaleString()}` : '');
    } else st.textContent = 'No calibration yet. Check the vocabulary, then start.';
    $('calStart').textContent = this.model ? 'Recalibrate (full, 25 words)' : 'Full calibration (25 words × 5)';
    ($('calStart') as HTMLButtonElement).disabled = this.vocab.length < MIN_WORDS;
    ($('calDel') as HTMLButtonElement).disabled = !this.model && this.samples.length === 0;
  }

  private async wipe() {
    if (!confirm('Delete all calibration samples and the trained model from this browser?')) return;
    try { await deleteCalibration(); } catch (e) { console.warn(e); }
    this.samples = []; this.evals = null; this.savedAt = 0; this.setModel(null); this.show('home');
  }

  // ---------- lighting / framing check ----------
  private startCheck(quick: boolean) {
    this.quickPlan = quick;
    if (this.busy()) { $('calStatus').textContent = 'Finish or wait for the M2 test first.'; return; }
    this.show('check');
  }

  updateFace(f: FaceInfo) {
    if (this.mode !== 'check') return;
    const ratio = f.leftLuma > 0 && f.rightLuma > 0 ? Math.min(f.leftLuma, f.rightLuma) / Math.max(f.leftLuma, f.rightLuma) : 0;
    const checks = [
      { ok: f.hasFace, label: 'Face detected', hint: 'Face the camera.' },
      { ok: f.hasFace && Math.abs(f.cx - 0.5) < 0.15 && Math.abs(f.cy - 0.5) < 0.2, label: 'Face centered', hint: 'Move so your face is in the middle of the frame.' },
      { ok: f.hasFace && f.faceH > 0.35 && f.faceH < 0.85, label: 'Distance OK', hint: f.faceH <= 0.35 ? 'Move closer.' : 'Move back a little.' },
      { ok: f.hasFace && f.luma > 70 && f.luma < 200, label: `Brightness OK (${f.luma.toFixed(0)})`, hint: f.luma <= 70 ? 'Too dark: face a light.' : 'Too bright: reduce glare/backlight.' },
      { ok: f.hasFace && ratio > 0.65, label: 'Even lighting', hint: 'One side of your face is much darker: face the light source.' },
    ];
    const ul = $('checks'); ul.replaceChildren();
    for (const c of checks) ul.append(el('li', c.ok ? 'ok' : 'bad', `${c.ok ? '✓' : '✗'} ${c.label}${c.ok ? '' : ' — ' + c.hint}`));
    const all = checks.every((c) => c.ok);
    $('calBegin').textContent = (all ? 'Begin recording' : 'Begin anyway (accuracy may suffer)') + (this.quickPlan ? ` · ${QUICK_VOCAB.length} words × ${QUICK_REPS}` : ` · ${this.vocab.length} words × ${FULL_REPS}`);
  }

  // ---------- recording ----------
  private beginRecording(words: string[], fresh: boolean) {
    if (fresh) this.samples = [];
    this.queue = words; this.qi = 0; this.flash = '';
    this.show('record'); this.nextWord(true);
  }

  private nextWord(first = false) {
    if (this.qi >= this.queue.length) { void this.train(); return; }
    this.wordStart = performance.now() + (first ? 800 : 1500);
    this.flash = first ? '' : 'Next word…';
    this.renderRecord();
  }

  private renderRecord() {
    const w = this.queue[this.qi]; const n = this.count(w);
    $('recWord').textContent = w;
    $('recCode').textContent = `Word ${this.qi + 1} of ${this.queue.length} · mouth it silently, then close your mouth`;
    $('recInfo').textContent = `${n}/${this.reps} recorded` + (this.flash ? ` · ${this.flash}` : '');
    ($('recBar') as HTMLProgressElement).value = (this.qi * this.reps + n) / (this.queue.length * this.reps);
    ($('recUndo') as HTMLButtonElement).disabled = n === 0;
  }

  /** Wired by main: loosen the activity detector's thresholds. */
  onSensitive?: () => void;

  /** Live detector state, shown on the recording screen so a non-triggering detector is visible. */
  updateActivity(active: boolean, energy: number, startThr: number) {
    if (this.mode !== 'record') return;
    $('recAct').textContent = active
      ? `Mouth movement detected (activity ${energy.toFixed(2)}). Finish the word, then close your mouth.`
      : `Waiting for movement — activity ${energy.toFixed(2)}, needs above ${startThr.toFixed(2)}. ` +
        (energy < startThr * 0.5 ? 'If this stays low while you mouth, press "Make detection more sensitive" and exaggerate the lips a little.' : '');
  }

  onSegment(seg: Segment) {
    if (this.mode !== 'record' || this.holdFrames) return; // hold-to-record takes priority
    if (seg.startMs < this.wordStart - 200) { // belongs to the previous word / pause
      this.flash = 'Ignored a movement that started before this word appeared — mouth it again';
      this.renderRecord(); return;
    }
    this.accept(seg.frames, seg.times);
  }

  // ---- hold-to-record: the user marks the word boundaries, so no detector tuning is needed ----
  private bindHold() {
    const btn = $('recHold');
    const down = () => this.holdStart(), up = () => this.holdEnd();
    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); down(); });
    window.addEventListener('pointerup', up);
    window.addEventListener('keydown', (e) => {
      if (e.code !== 'Space' || e.repeat || this.mode !== 'record') return;
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      e.preventDefault(); down();
    });
    window.addEventListener('keyup', (e) => { if (e.code === 'Space') up(); });
  }

  private holdStart() {
    if (this.mode !== 'record' || this.holdFrames) return;
    this.holdFrames = []; $('recHold').classList.add('down'); this.flash = 'Recording… mouth the word, then release';
    this.renderRecord();
  }

  private holdEnd() {
    const h = this.holdFrames;
    if (!h) return;
    this.holdFrames = null; $('recHold').classList.remove('down');
    this.accept(h.map((x) => x.v), h.map((x) => x.t));
  }

  /** Called by main every frame with the feature vector while a face is tracked. */
  pushFrame(v: Float32Array, t: number) { this.holdFrames?.push({ v, t }); }

  /** Shared by the automatic detector and the hold-to-record button. */
  private accept(rawFrames: Float32Array[], rawTimes: number[]) {
    const w = this.queue[this.qi];
    if (rawFrames.length < 8) { this.flash = 'Too short — try again'; this.renderRecord(); return; }
    const { frames, times } = trimToMotion(rawFrames, rawTimes);
    const dur = times[times.length - 1] - times[0];
    if (frames.length < 8 || dur < 200) { this.flash = 'Too short — try again'; this.renderRecord(); return; }
    if (dur > 3000) { this.flash = 'Too long — try again'; this.renderRecord(); return; }
    this.samples.push({ word: w, seq: resample(frames, times), durationMs: Math.round(dur) });
    this.flash = 'Got it';
    if (this.count(w) >= this.reps) { this.qi++; this.nextWord(); } else this.renderRecord();
  }

  private undo() {
    const w = this.queue[this.qi];
    for (let i = this.samples.length - 1; i >= 0; i--) if (this.samples[i].word === w) { this.samples.splice(i, 1); break; }
    this.flash = 'Removed last'; this.renderRecord();
  }

  // ---------- training / results ----------
  private async train() {
    this.show('train');
    const bar = $('trainBar') as HTMLProgressElement;
    try {
      const geo = await trainAndEvaluate(this.samples, this.vocab, 'geometry', (f) => { bar.value = f * 0.5; });
      const ext = await trainAndEvaluate(this.samples, this.vocab, 'geometry+extras', (f) => { bar.value = 0.5 + f * 0.5; });
      this.evals = { geometry: geo, 'geometry+extras': ext };
      this.used = ext.accuracy > geo.accuracy ? 'geometry+extras' : 'geometry';
      this.setModel(this.evals[this.used].model);
      this.savedAt = Date.now();
      await saveCalibration({
        vocab: this.vocab, samples: this.samples, featureSet: this.used,
        accuracy: { geometry: geo.accuracy, 'geometry+extras': ext.accuracy }, savedAt: this.savedAt,
      });
    } catch (e) { $('calStatus').textContent = `Training failed: ${(e as Error).message}`; this.show('home'); return; }
    this.show('results'); this.renderResults();
  }

  private renderResults() {
    const ev = this.evals![this.used], g = this.evals!.geometry, x = this.evals!['geometry+extras'];
    $('resSummary').textContent =
      `${ev.samples} samples, ${ev.vocab.length} words. Leave-one-out accuracy: ${pct(ev.accuracy)} with the model in use (${this.used}). ` +
      `Geometry only: ${pct(g.accuracy)} · plus tongue & inner-mouth features: ${pct(x.accuracy)}. ` +
      `This is cross-validation on your own calibration samples; live accuracy is measured separately in M4/M6.`;

    const tbl = document.createElement('table'); tbl.className = 'cm';
    const head = tbl.insertRow(); head.insertCell();
    ev.vocab.forEach((w) => { const c = head.insertCell(); c.textContent = w; c.className = 'rot'; });
    ev.confusion.forEach((row, i) => {
      const tr = tbl.insertRow(); const lab = tr.insertCell(); lab.textContent = ev.vocab[i]; lab.className = 'lab';
      row.forEach((v, j) => {
        const c = tr.insertCell(); c.textContent = v ? String(v) : '';
        if (v) c.style.background = `color-mix(in srgb, var(${i === j ? '--c3' : '--c2'}) ${Math.round((v / this.reps) * 85) + 15}%, transparent)`;
      });
    });
    const wrap = $('matrix'); wrap.replaceChildren(tbl);

    const pairs: [number, number, number][] = [];
    ev.confusion.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v) pairs.push([i, j, v]); }));
    pairs.sort((a, b) => b[2] - a[2]);
    const conf = $('topConf'); conf.replaceChildren();
    conf.append(el('div', undefined, pairs.length ? 'Most confused (true → predicted):' : 'No confusions in cross-validation.'));
    pairs.slice(0, 6).forEach(([i, j, v]) => conf.append(el('div', 'warn', `${ev.vocab[i]} → ${ev.vocab[j]} (${v}/${this.reps})`)));

    const rows = $('wordRows'); rows.replaceChildren();
    ev.vocab.map((w, i) => ({ w, acc: ev.perWord[i] })).sort((a, b) => a.acc - b.acc).forEach(({ w, acc }) => {
      const r = el('div', 'wrow' + (acc < 0.8 ? ' weak' : '')); r.append(el('span', undefined, `${w} — ${pct(acc)}`));
      const rec = el('button', 'sm', 'Re-record'); rec.onclick = () => this.rerecord(w, w);
      const sw = el('button', 'sm', 'Swap word');
      sw.onclick = () => { const n = prompt(`Replace "${w}" with:`)?.trim().toLowerCase().replace(/[^a-z' ]/g, ''); if (n && !this.vocab.includes(n)) this.rerecord(w, n); };
      r.append(rec, sw); rows.append(r);
    });
  }

  private rerecord(oldW: string, newW: string) {
    this.vocab = this.vocab.map((v) => (v === oldW ? newW : v));
    this.samples = this.samples.filter((s) => s.word !== oldW);
    this.beginRecording([newW], false);
  }
}
