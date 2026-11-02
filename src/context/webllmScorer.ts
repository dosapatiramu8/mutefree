import type { LanguageScorer, ScoreRequest } from '../core/types';
import { LETTERS, buildMessages, debiased, lettersToProbs, type TopLogprob } from './choicePrompt';

export { lettersToProbs };
export const WEBLLM_MODEL = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';

interface ChatEngineLike {
  chat: { completions: { create(req: unknown): Promise<{ choices: { logprobs?: { content?: { top_logprobs: TopLogprob[] }[] } | null }[] }> } };
  unload?(): Promise<void>;
}
export type EngineFactory = (modelId: string, onProgress: (fraction: number, text: string) => void) => Promise<ChatEngineLike>;

/** Default factory: loads WebLLM lazily so the library is only fetched when the user opts in. */
const defaultFactory: EngineFactory = async (modelId, onProgress) => {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
  const webllm = await import('@mlc-ai/web-llm');
  return (await webllm.CreateMLCEngine(modelId, {
    initProgressCallback: (r) => onProgress(r.progress, r.text),
  })) as unknown as ChatEngineLike;
};

/**
 * In-browser LLM used as a multiple-choice ranker (see choicePrompt.ts). It never writes free
 * text, so it cannot introduce a word the camera did not consider. By default each question is
 * asked in both candidate orders and averaged to cancel letter-position bias.
 */
export class WebLlmScorer implements LanguageScorer {
  readonly name = 'in-browser LLM';
  state: 'idle' | 'loading' | 'ready' | 'error' = 'idle';
  error = '';
  debias = true;
  private engine: ChatEngineLike | null = null;

  constructor(private factory: EngineFactory = defaultFactory, public modelId = WEBLLM_MODEL) {}

  async load(onProgress: (fraction: number, text: string) => void): Promise<void> {
    this.state = 'loading'; this.error = '';
    try {
      this.engine = await this.factory(this.modelId, onProgress);
      this.state = 'ready';
    } catch (e) {
      this.state = 'error'; this.error = (e as Error).message;
      throw e;
    }
  }

  async dispose(): Promise<void> { await this.engine?.unload?.(); this.engine = null; this.state = 'idle'; }

  async score(req: ScoreRequest): Promise<number[]> {
    if (!this.engine) throw new Error('The language model is not loaded');
    if (req.candidates.length < 2 || req.candidates.length > LETTERS.length) throw new Error('Unsupported number of candidates');
    const ask = (ordered: string[]) => this.ask(req, ordered);
    return this.debias ? debiased(req.candidates, ask, false) : ask(req.candidates);
  }

  private async ask(req: ScoreRequest, ordered: string[]): Promise<number[]> {
    const res = await this.engine!.chat.completions.create({
      messages: buildMessages(req, ordered),
      max_tokens: 1, temperature: 0, logprobs: true, top_logprobs: 8,
    });
    const top = res.choices[0]?.logprobs?.content?.[0]?.top_logprobs;
    if (!top) throw new Error('The model returned no token probabilities');
    return lettersToProbs(top, ordered.length);
  }
}
