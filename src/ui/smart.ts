import type { SentencePipeline } from '../context/sentencePipeline';
import type { LanguageScorer } from '../core/types';
import { CandidateScoringResolver, RESOLVER_DEFAULTS } from '../context/candidateResolver';
import { NullResolver } from '../context/resolver';
import { NgramScorer } from '../context/ngramScorer';
import { CORPUS } from '../context/corpus';
import { WebLlmScorer, WEBLLM_MODEL } from '../context/webllmScorer';
import { OllamaScorer, OLLAMA_URL, isLocalUrl } from '../context/ollamaScorer';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
type Engine = 'ngram' | 'ollama' | 'webllm';

/** Controls for Smart completion: on/off, language engine, connecting/loading, acceptance threshold. */
export class SmartCompletion {
  private ngram = new NgramScorer(CORPUS);
  private llm = new WebLlmScorer();
  private ollama = new OllamaScorer(OLLAMA_URL);
  private opts = { ...RESOLVER_DEFAULTS };

  constructor(private pipeline: SentencePipeline) {
    $('smartOn').onchange = () => this.apply();
    $('smartEngine').onchange = () => this.apply();
    $('smartLoad').onclick = () => void (this.engine === 'ollama' ? this.connectOllama() : this.loadModel());
    $('ollamaModel').onchange = () => { this.ollama.model = $<HTMLSelectElement>('ollamaModel').value; this.status(); };
    $<HTMLInputElement>('ollamaUrl').onchange = () => this.setUrl();
    const acc = $<HTMLInputElement>('smartAccept');
    const upd = () => { this.opts.acceptThreshold = +acc.value; $('smartAcceptV').textContent = (+acc.value).toFixed(2); };
    acc.oninput = upd; upd();
    this.apply();
  }

  private get engine(): Engine { return $<HTMLSelectElement>('smartEngine').value as Engine; }

  private scorer(): LanguageScorer {
    return this.engine === 'webllm' ? this.llm : this.engine === 'ollama' ? this.ollama : this.ngram;
  }

  private setUrl() {
    const url = $<HTMLInputElement>('ollamaUrl').value.trim().replace(/\/+$/, '');
    this.ollama = new OllamaScorer(url || OLLAMA_URL);
    $('ollamaWarn').hidden = isLocalUrl(this.ollama.baseUrl);
    $('ollamaModelBox').hidden = true;
    this.apply();
  }

  private apply() {
    const on = $<HTMLInputElement>('smartOn').checked;
    this.pipeline.resolver = on ? new CandidateScoringResolver(this.scorer(), this.opts) : new NullResolver();
    // The LLM engines make up to two calls per gap, so they get a longer deadline before the raw result is used.
    this.pipeline.resolverTimeoutMs = this.engine === 'ngram' ? 500 : this.engine === 'ollama' ? 12000 : 6000;
    const ready = this.engine === 'ollama' ? this.ollama.state === 'ready' : this.engine === 'webllm' ? this.llm.state === 'ready' : true;
    $('smartLoad').hidden = this.engine === 'ngram' || ready;
    $('smartLoad').textContent = this.engine === 'ollama' ? 'Connect to Ollama' : 'Download and load model';
    $('ollamaModelBox').hidden = !(this.engine === 'ollama' && this.ollama.state === 'ready');
    this.status();
  }

  private status(msg?: string) {
    const s = $('smartStat');
    if (msg) { s.textContent = msg; return; }
    if (!$<HTMLInputElement>('smartOn').checked) { s.textContent = 'Off: uncertain words stay "?".'; return; }
    if (this.engine === 'ngram') {
      s.textContent = 'On: the built-in phrase model picks among the camera\'s own candidates. It only knows a small list of everyday phrases.';
    } else if (this.engine === 'ollama') {
      const o = this.ollama;
      s.textContent = o.state === 'ready' ? `On: ${o.model} on Ollama (${o.baseUrl}) picks among the camera's own candidates. Expect a couple of seconds per uncertain word.`
        : o.state === 'error' ? `${o.error} Uncertain words stay "?" until it is connected.`
        : 'Not connected yet, so uncertain words stay "?". Press "Connect to Ollama".';
    } else if (this.llm.state === 'ready') {
      s.textContent = `On: ${WEBLLM_MODEL} is running in your browser and picks among the camera's own candidates.`;
    } else if (this.llm.state === 'error') {
      s.textContent = `Could not load the language model: ${this.llm.error}. Uncertain words stay "?".`;
    } else {
      s.textContent = 'The language model is not loaded yet, so uncertain words stay "?". Press "Download and load model".';
    }
  }

  private async connectOllama() {
    ($('smartLoad') as HTMLButtonElement).disabled = true;
    this.status('Connecting to Ollama…');
    try {
      await this.ollama.connect();
      const sel = $<HTMLSelectElement>('ollamaModel');
      sel.replaceChildren(...this.ollama.models.map((m) => {
        const o = document.createElement('option');
        o.value = m.name; o.textContent = `${m.name} (${(m.size / 1e9).toFixed(1)} GB)`; return o;
      }));
      sel.value = this.ollama.model;
    } catch { /* the message is in ollama.error and shown by status() */ }
    ($('smartLoad') as HTMLButtonElement).disabled = false;
    this.apply();
  }

  private async loadModel() {
    ($('smartLoad') as HTMLButtonElement).disabled = true;
    const bar = $('smartBar') as HTMLProgressElement;
    bar.hidden = false; bar.value = 0;
    try {
      await this.llm.load((f, text) => { bar.value = f; this.status(`Loading ${WEBLLM_MODEL}: ${(f * 100).toFixed(0)}% ${text}`); });
    } catch { /* state and message are shown by status() */ }
    bar.hidden = true;
    ($('smartLoad') as HTMLButtonElement).disabled = false;
    this.apply();
  }
}
