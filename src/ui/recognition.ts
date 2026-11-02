import type { Model } from '../model/classifier';
import type { Segment } from '../activity/detector';
import type { Speaker } from '../audio/speaker';
import { FLAT_VOICE, VoiceSmoother, mapExpression, type VoiceParams } from '../audio/voice';
import type { Expr } from '../features/expression';
import type { Candidate, RecognitionToken, TokenSource } from '../core/types';
import { DtwRecognizer } from '../recognition/dtwRecognizer';
import { SentencePipeline, type FinalSentence } from '../context/sentencePipeline';
import { displayText } from '../context/sentenceBuffer';
import { NullResolver } from '../context/resolver';
import { parseSimulated } from '../context/simulate';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

type Status = 'spoken' | 'muted' | 'unsure' | 'repeat' | 'queued';
interface Entry {
  text: string; closest: string; conf: number; status: Status; source: TokenSource; candidates: Candidate[];
  waitMs: number; classifyMs: number; ttsMs: number | null; totalMs: number | null;
  voice?: VoiceParams; expr?: Expr; test?: boolean;
}

const COOLDOWN_MS = 1200;
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const candText = (c: Candidate[]) => c.map((x) => `${x.text} ${pct(x.visualConfidence)}`).join(', ');

/**
 * Turns finished mouthing segments into spoken output. Nothing is ever guessed:
 *  - Word mode (legacy): a word is spoken at once if it is confident, else "?" is shown and nothing is said.
 *  - Sentence mode: tokens (with their source and candidates) collect in a buffer and the whole
 *    sentence is spoken once, after a pause.
 */
export class Recognition {
  private recognizer = new DtwRecognizer();
  readonly pipeline = new SentencePipeline();
  mode: 'word' | 'sentence' = 'word';
  private history: Entry[] = [];
  private lastWord = '';
  private lastEmit = 0;
  private seen = 0;
  private lastFinal: FinalSentence | null = null;
  private speakingStatus = '';
  expressive = true;
  gain = 2;
  private smoother = new VoiceSmoother();
  /** Expression averaged over [startMs, endMs]; set by main. */
  exprWindow?: (startMs: number, endMs: number) => Expr;
  /** Current smoothed expression; set by main. */
  exprLive?: () => Expr;

  get model(): Model | null { return this.recognizer.model; }
  get threshold(): number { return this.recognizer.threshold; }
  /** Set by main to wire the stop-delay slider to the detector. */
  onStopDelay?: (ms: number) => void;

  constructor(private speaker: Speaker) {
    for (const [id, apply] of [
      ['rgThr', (v: number) => { this.recognizer.threshold = v; }],
      ['rgStop', (v: number) => { this.onStopDelay?.(v); }],
      ['sentTimeout', (v: number) => { this.pipeline.buffer.opts.timeoutMs = v; }],
    ] as const) {
      const inp = $<HTMLInputElement>(id);
      const upd = () => {
        apply(+inp.value);
        $(id + 'V').textContent = id === 'rgThr' ? (+inp.value).toFixed(2) : `${inp.value} ms`;
      };
      inp.addEventListener('input', upd); upd();
    }
    const mute = $<HTMLInputElement>('rgMute');
    mute.onchange = () => { speaker.muted = mute.checked; if (mute.checked) speaker.clear(); };
    const expr = $<HTMLInputElement>('rgExpr');
    expr.onchange = () => { this.expressive = expr.checked; this.smoother.reset(); };
    const gain = $<HTMLInputElement>('rgGain');
    const applyGain = () => { this.gain = +gain.value; $('rgGainV').textContent = gain.value; };
    gain.oninput = applyGain; applyGain();
    $('rgTest').onclick = () => this.testVoice();
    const mode = $<HTMLSelectElement>('rgMode');
    mode.onchange = () => { this.mode = mode.value as 'word' | 'sentence'; $('sentBox').hidden = this.mode !== 'sentence'; this.renderSentence(); };
    $('sentClear').onclick = () => { this.pipeline.clearBuffer(); this.pipeline.clearContext(); this.lastFinal = null; this.renderSentence(); };
    $('sentStop').onclick = () => { speaker.clear(); this.speakingStatus = 'stopped'; this.renderSentence(); };
    // Finalize on a timer too, so the sentence path also works with the camera off (typed simulator).
    setInterval(() => this.tick(performance.now()), 200);
    $('simRun').onclick = () => this.simulate($<HTMLInputElement>('simText').value);
    this.pipeline.onChange = () => this.renderSentence();
    this.pipeline.speak = (text, final) => this.speakSentence(text, final);
    this.pipeline.onFinal = (f) => { this.lastFinal = f; this.renderSentence(); };
    if (!speaker.available) $('rgNote').textContent = 'Speech synthesis is not supported in this browser: words will be shown but not spoken.';
    this.render(null);
    this.renderSentence();
  }

  setModel(m: Model | null) {
    this.recognizer.model = m;
    $('rgNote').textContent = m ? (this.speaker.available ? '' : $('rgNote').textContent)
      : 'Calibrate first: the recognizer only knows the words you recorded.';
  }

  /** Feed typed text through the real sentence / language-model / speech path. Not lip reading. */
  simulate(text: string) {
    if (!text.trim()) return;
    this.speaker.prime(); // button click is a user gesture: unlocks audio
    if (this.mode !== 'sentence') {
      this.mode = 'sentence';
      $<HTMLSelectElement>('rgMode').value = 'sentence';
      $('sentBox').hidden = false;
    }
    this.speakingStatus = '';
    for (const t of parseSimulated(text, performance.now())) this.pipeline.addToken(t);
    $('rgWhy').textContent = 'Simulator: typed input was added to the sentence. It is spoken after the pause below.';
  }

  /** Call every frame: finalizes and speaks the sentence once the speaker has paused. */
  tick(nowMs: number) { if (this.mode === 'sentence') void this.pipeline.tick(nowMs); }

  /** `detectedAt` is performance.now() when the detector handed over the finished segment. */
  async handle(seg: Segment, detectedAt: number) {
    this.seen++;
    if (!this.recognizer.model) {
      $('rgWhy').textContent = `Mouthing detected (${this.seen} so far) but there is no calibration, so nothing can be recognized or spoken. Run Calibration below.`;
      return;
    }
    let result;
    try {
      result = await this.recognizer.recognize({ frames: seg.frames, times: seg.times, startMs: seg.startMs, endMs: seg.endMs });
    } catch (e) {
      $('rgWhy').textContent = `Visual recognition failed: ${(e as Error).message}`;
      return;
    }
    const { token, diagnostics } = result;
    const waitMs = detectedAt - seg.endMs; // time spent waiting to be sure the word had ended
    const accepted = token.source === 'visual_confident';
    const entry: Entry = {
      text: token.text, closest: token.candidates[0]?.text ?? '–', conf: token.visualConfidence, status: 'unsure',
      source: token.source, candidates: token.candidates, waitMs, classifyMs: result.latencyMs, ttsMs: null, totalMs: null,
    };

    if (this.mode === 'sentence') {
      this.pipeline.addToken(token);
      entry.status = 'queued';
      $('rgWhy').textContent = accepted
        ? `Recognized "${token.text}" and added to the sentence. It will be spoken after a ${this.pipeline.buffer.opts.timeoutMs} ms pause.`
        : this.explainRejected(token, diagnostics);
    } else {
      if (accepted) {
        const now = performance.now();
        if (token.text === this.lastWord && now - this.lastEmit < COOLDOWN_MS) {
          entry.status = 'repeat';
        } else {
          this.lastWord = token.text; this.lastEmit = now;
          const ex = this.exprWindow?.(seg.startMs + 150, seg.endMs);
          entry.expr = ex;
          entry.voice = this.voiceFor(ex);
          const queued = this.speaker.speak(token.text, (tStart) => {
            entry.status = 'spoken'; entry.ttsMs = tStart - now;
            entry.totalMs = tStart - seg.endMs; this.render(token.visualConfidence);
          }, entry.voice);
          entry.status = queued ? 'spoken' : 'muted';
        }
      }
      $('rgWhy').textContent = accepted
        ? (entry.status === 'repeat' ? 'Same word repeated within the cooldown: not spoken again.'
          : entry.status === 'muted' ? 'Recognized, but voice is muted or unsupported.' : `Recognized "${token.text}" and sent to the voice.`)
        : this.explainRejected(token, diagnostics);
    }
    this.history.unshift(entry);
    if (this.history.length > 30) this.history.pop();
    this.render(token.visualConfidence);
  }

  private explainRejected(token: RecognitionToken, d: { distance: number; rejectDistance: number }): string {
    return token.source === 'unknown'
      ? `That movement does not match any calibrated word closely enough (distance ${d.distance.toFixed(2)} > ${d.rejectDistance.toFixed(2)}). Not spoken.`
      : `Closest was "${token.candidates[0]?.text}" with confidence ${pct(token.visualConfidence)} < threshold ${pct(this.threshold)}. Lower the threshold slider to speak it.`;
  }

  private voiceFor(ex: Expr | undefined): VoiceParams {
    return this.expressive && ex ? this.smoother.next(mapExpression(ex, this.gain)) : FLAT_VOICE;
  }

  private speakSentence(text: string, final: FinalSentence) {
    const ex = this.exprWindow?.(final.startMs + 150, final.endMs);
    const voice = this.voiceFor(ex);
    const t0 = performance.now();
    this.speakingStatus = 'speaking…';
    const queued = this.speaker.speak(text, (tStart) => {
      this.speakingStatus = `speaking · started ${(tStart - final.endMs).toFixed(0)} ms after your last word`;
      this.renderSentence();
    }, voice, () => { this.speakingStatus = 'done'; this.renderSentence(); });
    if (!queued) this.speakingStatus = 'not spoken (muted or unsupported)';
    void t0;
    this.renderSentence();
  }

  /** Speak "hello" with the face you are making right now, bypassing recognition. */
  testVoice() {
    const ex = this.exprLive?.();
    const voice = this.voiceFor(ex);
    const t0 = performance.now();
    const entry: Entry = { text: 'hello', closest: 'hello', conf: 1, status: 'muted', source: 'visual_confident', candidates: [], waitMs: 0, classifyMs: 0, ttsMs: null, totalMs: null, voice, expr: ex, test: true };
    const queued = this.speaker.speak('hello', (t) => { entry.ttsMs = t - t0; this.render(null); }, voice);
    entry.status = queued ? 'spoken' : 'muted';
    this.history.unshift(entry); this.render(null);
  }

  private renderSentence() {
    if (this.mode !== 'sentence') return;
    const live = this.pipeline.buffer.peek();
    const confident = live.filter((t) => t.source === 'visual_confident').map((t) => t.text).join(' ');
    const unc = live.map((t, i) => ({ t, i })).filter(({ t }) => t.source !== 'visual_confident');
    $('sentRecognized').textContent = confident || '–';
    $('sentUncertain').textContent = unc.length
      ? unc.map(({ t, i }) => (t.source === 'unknown' ? `word ${i + 1}: ? (matches nothing known)` : `word ${i + 1}: ? [${candText(t.candidates)}]`)).join(' · ')
      : '–';
    const res = this.lastFinal?.resolution;
    $('sentSuggest').textContent = this.pipeline.resolver instanceof NullResolver ? 'off (Smart completion is disabled)'
      : !res ? 'decided when the sentence ends'
      : res.fallback ? 'resolver unavailable or too slow: using the raw visual result'
      : res.replacements.length
        ? res.replacements.map((r) => `word ${r.position + 1}: ? → ${r.to} (visual ${pct(r.visualConfidence ?? 0)}, language ${pct(r.languageConfidence ?? 0)}, combined ${pct(r.combinedConfidence ?? 0)})`).join(' · ') + ` · ${res.latencyMs.toFixed(0)} ms`
        : `none: no gap could be filled safely · ${res.latencyMs.toFixed(0)} ms`;
    const f = this.lastFinal;
    $('sentCurrent').textContent = live.length ? displayText(live) : '–';
    $('sentFinal').textContent = f ? `${f.display}${f.spoken ? '' : '  (nothing confident enough to speak)'}` : '–';
    $('sentStatus').textContent = this.pipeline.resolving ? 'completing the sentence with the language model…' : this.speakingStatus || (live.length ? 'listening…' : 'idle');
  }

  private render(conf: number | null) {
    const e = this.history[0];
    $('rgWord').textContent = e ? e.text : '–';
    $('rgWord').classList.toggle('unsure', !!e && e.text === '?');
    ($('rgConf') as HTMLProgressElement).value = conf !== null ? Math.min(conf, 1) : 0;
    $('rgConfV').textContent = conf !== null ? pct(conf) : '–';
    $('rgClosest').textContent = e
      ? (e.text === '?' ? `not spoken · closest guess "${e.closest}" is below threshold or too far from your samples` : e.status === 'repeat' ? 'repeat suppressed (cooldown)' : '')
      : '';

    const list = $('rgHist'); list.replaceChildren();
    for (const h of this.history) {
      const li = document.createElement('li'); li.className = h.status;
      const lat = h.totalMs !== null ? ` · ${h.totalMs.toFixed(0)} ms` : '';
      const v = h.voice ? ` · pitch ${h.voice.pitch.toFixed(2)} rate ${h.voice.rate.toFixed(2)} vol ${h.voice.volume.toFixed(2)}` : '';
      const cands = h.text === '?' && h.candidates.length ? ` [${candText(h.candidates)}]` : '';
      li.textContent = `${h.test ? '[test] ' : ''}${h.text} (${h.test ? '–' : pct(h.conf)}) ${h.text === '?' ? h.source : ''}${cands}${h.status === 'muted' ? ' · muted' : ''}${h.status === 'repeat' ? ' · repeat' : ''}${h.status === 'queued' ? ' · in sentence' : ''}${v}${lat}`;
      list.append(li);
    }

    const done = this.history.filter((h) => h.totalMs !== null);
    const last = done[0];
    const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
    $('rgLat').textContent = last
      ? `Last: ${last.totalMs!.toFixed(0)} ms from mouth stopping to audio (wait ${last.waitMs.toFixed(0)} + classify ${last.classifyMs.toFixed(0)} + voice start ${(last.totalMs! - last.waitMs - last.classifyMs).toFixed(0)}). Median of ${done.length}: ${med(done.map((d) => d.totalMs!)).toFixed(0)} ms. Dropped stale words: ${this.speaker.dropped}.`
      : 'Latency appears after the first spoken word.';
  }
}
