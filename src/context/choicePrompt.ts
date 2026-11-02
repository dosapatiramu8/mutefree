import type { ScoreRequest } from '../core/types';

export interface TopLogprob { token: string; logprob: number }

export const LETTERS = 'ABCDEFGH';
const MISSING_LOGPROB = -20;

const SYSTEM = 'A word in a lip-read sentence is unclear. Pick the candidate that makes the most natural sentence. Answer with a single letter only.';

/** The multiple-choice question shared by every LLM engine, so they are comparable. */
export function buildMessages(req: ScoreRequest, ordered: string[]): { role: 'system' | 'user'; content: string }[] {
  const sentence = `${req.left.join(' ')} ____ ${req.right.join(' ')}`.replace(/\s+/g, ' ').trim();
  const context = req.history.length ? `Earlier sentences: ${req.history.join(' ')}\n` : '';
  const options = ordered.map((c, i) => `${LETTERS[i]}) ${c}`).join('\n');
  return [
    { role: 'system', content: SYSTEM },
    { role: 'user', content: `${context}Sentence: "${sentence}" (the underscore marks the unclear word; _ marks other unclear words)\nCandidates:\n${options}\nAnswer:` },
  ];
}

/** Softmax over the logprobs the model gave to the answer letters; absent letters get a floor. */
export function lettersToProbs(top: TopLogprob[], n: number): number[] {
  const lp = Array.from({ length: n }, () => MISSING_LOGPROB);
  for (const t of top) {
    const i = LETTERS.indexOf(t.token.trim().toUpperCase());
    if (t.token.trim().length === 1 && i >= 0 && i < n) lp[i] = Math.max(lp[i], t.logprob);
  }
  const m = Math.max(...lp);
  const e = lp.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/**
 * Ask in the original and the reversed candidate order and average, which cancels the
 * letter-position bias small models show. `ask` returns probabilities for the order it was given.
 */
export async function debiased(
  candidates: string[], ask: (ordered: string[]) => Promise<number[]>, parallel = true,
): Promise<number[]> {
  const rev = [...candidates].reverse();
  const [fwd, back] = parallel ? await Promise.all([ask(candidates), ask(rev)]) : [await ask(candidates), await ask(rev)];
  const b = [...back].reverse();
  return fwd.map((p, i) => (p + b[i]) / 2);
}
