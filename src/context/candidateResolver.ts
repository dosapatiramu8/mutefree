import type { ContextResolution, ContextResolver, ConversationContext, LanguageScorer, RecognitionToken, Replacement } from '../core/types';

export interface CandidateResolverOptions {
  /** A candidate replaces a "?" only if its combined confidence reaches this. */
  acceptThreshold: number;
  /** Exponent on the visual score. */
  visualWeight: number;
  /** Exponent on the language score. */
  languageWeight: number;
  /** How many previous sentences the scorer may see. */
  historySentences: number;
}

export const RESOLVER_DEFAULTS: CandidateResolverOptions = {
  acceptThreshold: 0.6, visualWeight: 1, languageWeight: 1, historySentences: 2,
};

const FLOOR = 1e-6;

/**
 * Fills "?" gaps by choosing among the VISUAL model's own candidates using language evidence.
 *
 *  - It only touches `visual_uncertain` tokens that carry at least two candidates.
 *  - `visual_confident` tokens are never modified; `unknown` tokens (nothing plausible) stay "?".
 *  - The scorer can only rank the given candidates, so a word the camera did not consider
 *    plausible can never be introduced.
 *  - combined_i is proportional to visual_i^wv * language_i^wl, renormalized over the candidates;
 *    the best one is accepted only if its combined confidence reaches the threshold, otherwise
 *    the gap stays "?".
 *  - Gaps are resolved left to right, so an earlier inference becomes context for a later one.
 *    Every accepted replacement is recorded with its evidence.
 */
export class CandidateScoringResolver implements ContextResolver {
  constructor(private scorer: LanguageScorer, public opts: CandidateResolverOptions = { ...RESOLVER_DEFAULTS }) {}

  async resolve(tokens: RecognitionToken[], context: ConversationContext): Promise<ContextResolution> {
    const t0 = performance.now();
    const out = tokens.map((t) => ({ ...t }));
    const replacements: Replacement[] = [];
    const history = context.sentences.slice(-this.opts.historySentences);

    for (let i = 0; i < out.length; i++) {
      const t = out[i];
      if (t.source !== 'visual_uncertain' || t.candidates.length < 2) continue;

      const word = (x: RecognitionToken) => (x.source === 'visual_confident' || x.source === 'context_inferred' ? x.text : '_');
      const lang = await this.scorer.score({
        left: out.slice(0, i).map(word),
        right: out.slice(i + 1).map(word),
        candidates: t.candidates.map((c) => c.text),
        history,
      });
      if (lang.length !== t.candidates.length) throw new Error(`Scorer "${this.scorer.name}" returned ${lang.length} scores for ${t.candidates.length} candidates`);

      const raw = t.candidates.map((c, k) =>
        Math.pow(Math.max(c.visualConfidence, FLOOR), this.opts.visualWeight) * Math.pow(Math.max(lang[k], FLOOR), this.opts.languageWeight));
      const sum = raw.reduce((a, b) => a + b, 0);
      const combined = raw.map((r) => r / sum);
      let best = 0;
      for (let k = 1; k < combined.length; k++) if (combined[k] > combined[best]) best = k;
      if (combined[best] < this.opts.acceptThreshold) continue; // not safe: leave the "?"

      const chosen = t.candidates[best];
      replacements.push({
        position: i, from: t.text, to: chosen.text,
        visualConfidence: chosen.visualConfidence, languageConfidence: lang[best], combinedConfidence: combined[best],
      });
      out[i] = { ...t, text: chosen.text, source: 'context_inferred', languageConfidence: lang[best] };
    }

    const confidence = replacements.length
      ? replacements.reduce((a, r) => a + (r.combinedConfidence ?? 0), 0) / replacements.length
      : 1;
    return { tokens: out, replacements, confidence, fallback: false, latencyMs: performance.now() - t0 };
  }
}
