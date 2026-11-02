/**
 * Shared contracts for the recognition -> context -> speech pipeline.
 * Nothing here touches the DOM, so every implementation behind these interfaces
 * (DTW today, a pretrained visual speech model or an LLM later) is unit-testable.
 */

/** Where a token's text came from. The UI shows each differently; speech rules depend on it. */
export type TokenSource =
  | 'visual_confident'   // the visual model is sure
  | 'visual_uncertain'   // the visual model has candidates but no clear winner
  | 'context_inferred'   // a context resolver filled/replaced it (never silently)
  | 'unknown';           // the movement matched nothing the model knows

export interface Candidate {
  text: string;
  /** Normalized across the candidates of one token (sums to ~1). A heuristic score, not a calibrated probability. */
  visualConfidence: number;
}

export interface RecognitionToken {
  /** The word, or "?" when unresolved. */
  text: string;
  source: TokenSource;
  /** Confidence in `text` from the visual side. For the DTW recognizer this is the margin between the best and runner-up word. */
  visualConfidence: number;
  /** Set only when a context resolver chose the text. */
  languageConfidence?: number;
  /** Ranked alternatives from the visual model (best first). Empty for `unknown`. */
  candidates: Candidate[];
  startMs: number;
  endMs: number;
}

// ---------- visual speech recognition ----------

export interface VisualSpeechInput {
  /** Per-frame feature vectors (the same layout the DTW model is trained on). */
  frames: Float32Array[];
  times: number[];
  startMs: number;
  endMs: number;
}

export interface VisualRecognitionResult {
  token: RecognitionToken;
  /** Raw evidence, kept for debugging and evaluation. */
  diagnostics: { distance: number; rejectDistance: number; runnerUpDistance: number };
  latencyMs: number;
}

export interface VisualSpeechRecognizer {
  initialize(): Promise<void>;
  recognize(input: VisualSpeechInput): Promise<VisualRecognitionResult>;
  dispose(): Promise<void>;
}

// ---------- context resolution ----------

export interface ConversationContext {
  /** Previously finalized sentences, oldest first. */
  sentences: string[];
}

export interface Replacement {
  position: number;
  from: string;
  to: string;
  /** The chosen candidate's normalized visual score. */
  visualConfidence?: number;
  /** The language scorer's probability for the chosen candidate among the candidates. */
  languageConfidence?: number;
  /** Visual and language evidence combined and renormalized over the candidates. */
  combinedConfidence?: number;
}

export interface ContextResolution {
  tokens: RecognitionToken[];
  replacements: Replacement[];
  confidence: number;
  /** True when the resolver failed or timed out and `tokens` are the unmodified input. */
  fallback: boolean;
  latencyMs: number;
}

export interface ContextResolver {
  resolve(tokens: RecognitionToken[], context: ConversationContext): Promise<ContextResolution>;
}

// ---------- language scoring ----------

export interface ScoreRequest {
  /** Words before the gap; `_` marks an earlier word that is still unresolved. */
  left: string[];
  /** Words after the gap; `_` marks a later unresolved word. */
  right: string[];
  /** The visual model's candidates for the gap. The scorer may only rank these. */
  candidates: string[];
  /** Recently finalized sentences, oldest first. */
  history: string[];
}

/**
 * Ranks a FIXED set of candidate words in context. It cannot invent words: it returns one
 * probability per candidate (summing to 1), in the same order as `request.candidates`.
 */
export interface LanguageScorer {
  readonly name: string;
  score(request: ScoreRequest): Promise<number[]>;
}

// ---------- speech ----------

export interface SpeechOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
  /** Called with performance.now() when audio actually starts. */
  onStart?: (tMs: number) => void;
}

export interface SpeechEngine {
  /** Resolves when the utterance ends, is dropped, or is stopped. */
  speak(text: string, options?: SpeechOptions): Promise<void>;
  stop(): void;
}
