import type { LanguageScorer, ScoreRequest } from '../core/types';

const START = '<s>', END = '</s>', GAP = '_';

/**
 * Built-in language scorer: a smoothed bigram model over a small phrase list. No download,
 * deterministic, instant. It scores P(candidate | previous word) * P(next word | candidate),
 * so it uses both sides of the gap. Words absent from the corpus get only the smoothing floor,
 * which keeps it from favouring candidates it knows nothing about.
 */
export class NgramScorer implements LanguageScorer {
  readonly name = 'phrase model';
  private bigram = new Map<string, number>();
  private unigram = new Map<string, number>();
  private vocab = new Set<string>();

  constructor(corpus: string[], private k = 0.05) {
    for (const line of corpus) {
      const w = [START, ...line.toLowerCase().replace(/[^a-z' ]/g, '').split(/\s+/).filter(Boolean), END];
      for (let i = 0; i < w.length; i++) {
        this.vocab.add(w[i]);
        this.unigram.set(w[i], (this.unigram.get(w[i]) ?? 0) + 1);
        if (i > 0) this.bigram.set(`${w[i - 1]} ${w[i]}`, (this.bigram.get(`${w[i - 1]} ${w[i]}`) ?? 0) + 1);
      }
    }
  }

  private p(next: string, prev: string): number {
    const V = this.vocab.size + 1;
    return ((this.bigram.get(`${prev} ${next}`) ?? 0) + this.k) / ((this.unigram.get(prev) ?? 0) + this.k * V);
  }

  async score(req: ScoreRequest): Promise<number[]> {
    const l = req.left.at(-1);
    const prev = l === undefined ? START : l === GAP ? null : l.toLowerCase();
    const r = req.right[0];
    const next = r === undefined ? END : r === GAP ? null : r.toLowerCase();
    const raw = req.candidates.map((c) => {
      const w = c.toLowerCase();
      let s = prev === null ? ((this.unigram.get(w) ?? 0) + this.k) / (this.vocab.size * this.k + [...this.unigram.values()].reduce((a, b) => a + b, 0)) : this.p(w, prev);
      if (next !== null) s *= this.p(next, w);
      return s;
    });
    const sum = raw.reduce((a, b) => a + b, 0) || 1;
    return raw.map((x) => x / sum);
  }
}
