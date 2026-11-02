import type { VoiceParams } from './voice';

interface Pending { text: string; onStart: (tMs: number) => void; voice?: VoiceParams; onEnd?: () => void; }

/**
 * Word-by-word speech queue on top of speechSynthesis. If the user gets more than
 * `maxPending` words ahead of the voice, the OLDEST waiting words are dropped so the
 * audio never lags far behind the mouth. The utterance currently playing is not cut.
 */
export class Speaker {
  muted = false;
  rate = 1; pitch = 1; volume = 1;
  maxPending = 2;
  dropped = 0;
  readonly available = typeof window !== 'undefined' && 'speechSynthesis' in window;
  /** Human-readable speech status for the UI (voice in use, last error). */
  status = '';
  onStatus?: () => void;
  private pending: Pending[] = [];
  private busy = false;
  private watchdog = 0;
  private voiceName = '';
  private lastError = '';

  constructor() {
    if (!this.available) { this.status = 'Speech synthesis is not supported in this browser.'; return; }
    try { this.voiceName = localStorage.getItem('mf-voice') ?? ''; } catch { /* storage unavailable */ }
    speechSynthesis.addEventListener('voiceschanged', () => this.refresh());
    this.refresh();
  }

  voices(): SpeechSynthesisVoice[] { return this.available ? speechSynthesis.getVoices() : []; }

  private chosen(): SpeechSynthesisVoice | undefined {
    const vs = this.voices();
    return vs.find((v) => v.name === this.voiceName)
      ?? vs.find((v) => v.localService && v.lang.toLowerCase().startsWith(navigator.language.slice(0, 2).toLowerCase()))
      ?? vs.find((v) => v.localService)
      ?? vs.find((v) => v.default) ?? vs[0];
  }

  setVoice(name: string) {
    this.voiceName = name;
    try { localStorage.setItem('mf-voice', name); } catch { /* ignore */ }
    this.refresh();
  }

  get voiceInUse() { return this.chosen()?.name ?? ''; }

  private refresh() {
    const n = this.voices().length, v = this.chosen();
    this.status = n === 0
      ? 'No speech voices available yet (the browser has none installed or is still loading them).'
      : `Voice: ${v?.name ?? 'default'}${v && !v.localService ? ' (online voice — may ignore pitch/rate)' : ''} · ${n} voices available`
        + (this.lastError ? ` · last error: ${this.lastError}` : '');
    this.onStatus?.();
  }

  /** Call from a user gesture once: browsers block speech until the page has had one. */
  prime() {
    if (!this.available) return;
    speechSynthesis.cancel(); // clear anything stuck from an earlier page load
    const u = new SpeechSynthesisUtterance('ok'); // not blank: blank utterances can hang the queue
    u.volume = 0;
    speechSynthesis.speak(u);
  }

  /** Returns false if the word was not queued (muted or unsupported). */
  speak(text: string, onStart: (tMs: number) => void, voice?: VoiceParams, onEnd?: () => void): boolean {
    if (!this.available || this.muted) return false;
    this.pending.push({ text, onStart, voice, onEnd });
    while (this.pending.length > this.maxPending) { this.pending.shift()?.onEnd?.(); this.dropped++; }
    this.pump();
    return true;
  }

  private pump() {
    if (this.busy) return;
    const next = this.pending.shift();
    if (!next) return;
    this.busy = true;
    if (speechSynthesis.paused) speechSynthesis.resume();
    const u = new SpeechSynthesisUtterance(next.text);
    const v = this.chosen();
    if (v) { u.voice = v; u.lang = v.lang; }
    u.rate = next.voice?.rate ?? this.rate; u.pitch = next.voice?.pitch ?? this.pitch; u.volume = next.voice?.volume ?? this.volume;
    const done = () => { clearTimeout(this.watchdog); this.busy = false; next.onEnd?.(); this.pump(); };
    u.onstart = () => { this.lastError = ''; this.refresh(); next.onStart(performance.now()); };
    u.onend = done;
    u.onerror = (e) => {
      // 'interrupted'/'canceled' are expected when we clear the queue; anything else is worth showing.
      if (e.error !== 'interrupted' && e.error !== 'canceled') { this.lastError = e.error; this.refresh(); }
      done();
    };
    // Chrome occasionally never fires onend; do not let the queue stall.
    this.watchdog = window.setTimeout(() => {
      this.lastError = 'voice did not start within 4 s (try another voice)'; this.refresh();
      speechSynthesis.cancel(); done();
    }, 4000);
    speechSynthesis.speak(u);
  }

  clear() {
    const dropped = this.pending; this.pending = [];
    if (this.available) speechSynthesis.cancel();
    this.busy = false;
    for (const p of dropped) p.onEnd?.();
  }
}
