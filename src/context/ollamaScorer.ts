import type { LanguageScorer, ScoreRequest } from '../core/types';
import { LETTERS, buildMessages, debiased, lettersToProbs, type TopLogprob } from './choicePrompt';

export const OLLAMA_URL = 'http://localhost:11434';

export interface OllamaModel { name: string; size: number; thinking: boolean }
type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

/** True when the URL points at this machine; anything else would send sentence text to another host. */
export function isLocalUrl(url: string): boolean {
  try {
    const h = new URL(url).hostname;
    return h === 'localhost' || h === '127.0.0.1' || h === '[::1]' || h === '::1';
  } catch { return false; }
}

/**
 * Language scorer backed by a local Ollama server. Same multiple-choice method as the in-browser
 * engine: the model only answers with a letter and we read its probabilities for the letters from
 * the token logprobs, so it can only rank the camera's candidates, never write new words.
 *
 * What is sent: the sentence with one gap, the candidate words and up to a few recent sentences,
 * to the configured URL. Nothing else; never video or mouth data. Keep the URL local.
 */
export class OllamaScorer implements LanguageScorer {
  readonly name = 'Ollama';
  state: 'idle' | 'connecting' | 'ready' | 'error' = 'idle';
  error = '';
  models: OllamaModel[] = [];
  model = '';
  debias = true;
  timeoutMs = 10000;

  constructor(public baseUrl = OLLAMA_URL, private fetchFn: Fetch = (u, i) => fetch(u, i)) {}

  /** Contact the server, list usable chat models and pick the best default. */
  async connect(preferred?: string): Promise<void> {
    this.state = 'connecting'; this.error = '';
    try {
      const res = await this.request('/api/tags', undefined, 3000);
      const body = (await res.json()) as { models?: { name: string; size: number; capabilities?: string[] }[] };
      this.models = (body.models ?? [])
        .filter((m) => !m.capabilities || m.capabilities.includes('completion'))
        .map((m) => ({ name: m.name, size: m.size, thinking: !!m.capabilities?.includes('thinking') }));
      if (!this.models.length) throw new Error('Ollama is running but has no chat models. Run "ollama pull qwen3:8b".');
      this.model = this.models.find((m) => m.name === preferred)?.name
        ?? this.models.find((m) => m.name.startsWith('qwen3'))?.name ?? this.models[0].name;
      this.state = 'ready';
    } catch (e) {
      this.state = 'error'; this.error = (e as Error).message;
      throw e;
    }
  }

  async score(req: ScoreRequest): Promise<number[]> {
    if (this.state !== 'ready') throw new Error('Ollama is not connected');
    const n = req.candidates.length;
    if (n < 2 || n > LETTERS.length) throw new Error('Unsupported number of candidates');
    const ask = (ordered: string[]) => this.ask(req, ordered);
    return this.debias ? debiased(req.candidates, ask) : ask(req.candidates);
  }

  private async ask(req: ScoreRequest, ordered: string[]): Promise<number[]> {
    const thinking = this.models.find((m) => m.name === this.model)?.thinking;
    const res = await this.request('/api/chat', {
      model: this.model, stream: false, keep_alive: '10m',
      messages: buildMessages(req, ordered),
      logprobs: true, top_logprobs: 8,
      options: { temperature: 0, num_predict: 1 },
      ...(thinking ? { think: false } : {}), // reasoning models would otherwise spend the one token on <think>
    }, this.timeoutMs);
    const j = (await res.json()) as { logprobs?: { top_logprobs?: TopLogprob[] }[] };
    const top = j.logprobs?.[0]?.top_logprobs;
    if (!top?.length) throw new Error('This Ollama version or model did not return token probabilities (logprobs)');
    return lettersToProbs(top, ordered.length);
  }

  private async request(path: string, body: unknown, timeoutMs: number): Promise<Response> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await this.fetchFn(this.baseUrl + path, {
        method: body ? 'POST' : 'GET', signal: ctl.signal,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!res.ok) {
        let detail = '';
        try { detail = ((await res.json()) as { error?: string }).error ?? ''; } catch { /* not json */ }
        throw new Error(`Ollama returned HTTP ${res.status}${detail ? `: ${detail}` : ''}`);
      }
      return res;
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw new Error(`Ollama did not answer within ${timeoutMs / 1000} s`);
      if (e instanceof TypeError) {
        throw new Error(`Cannot reach Ollama at ${this.baseUrl}. Is it running? If this page is not on localhost, allow its origin with the OLLAMA_ORIGINS environment variable and restart Ollama.`);
      }
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }
}
