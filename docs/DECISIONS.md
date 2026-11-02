# MuteFree decision log

Each entry: problem, options, decision, why, trade-offs, replacement path.

## D1. Test runner

- **Problem.** The repo had no automated tests; earlier checks were throwaway scripts.
- **Options.** Keep ad-hoc scripts; add Vitest; add Jest.
- **Decision.** Vitest (`npm test`), tests in `tests/`, node environment.
- **Why.** Already on Vite, no extra config, fast, handles TypeScript.
- **Trade-offs.** One more dev dependency. UI modules that touch the DOM are not covered; logic is kept out of them for that reason.
- **Replace.** Add `@vitest/browser` or Playwright later for camera-free UI tests.

## D2. Phase order: model-agnostic layers first

- **Problem.** The target design needs an open-vocabulary visual model, but its license, size and browser feasibility are unmeasured.
- **Options.** (a) Start with the model. (b) Build interfaces, uncertainty tokens, sentence buffer and context layer on the existing DTW recognizer first, spike the model separately.
- **Decision.** (b). Phases 1-3 are done on the DTW recognizer; the open-vocabulary model needs a Phase 0 spike and a license check first.
- **Why.** Phases 1-4 deliver value now (candidates, provenance, sentence speech) and are needed no matter which model is chosen later.
- **Trade-offs.** Until a new model exists, the system is still closed-vocabulary.
- **Replace.** Implement `VisualSpeechRecognizer` with the new model; nothing downstream changes.

## D3. Candidates come from DTW distances

- **Problem.** The context layer needs alternatives, not just a best word.
- **Decision.** `classify()` now returns the full ranking; `scoreCandidates()` converts the top 3 into normalized scores `exp(-(d - d_best) / (0.1 * rejectDist))`.
- **Why.** Free, monotone in distance, no extra model.
- **Trade-offs.** The scores are a heuristic, not calibrated probabilities. They must not be read as "41% chance".
- **Replace.** Calibrate against the evaluation log (Phase 8), or use CTC posteriors from a pretrained model.

## D4. Token provenance and the "unknown" distinction

- **Decision.** Four sources: `visual_confident`, `visual_uncertain`, `context_inferred`, `unknown`. A movement farther from every sample than the user's own rejection distance is `unknown` and carries no candidates, because candidates for something unrecognizable would be misleading. Below-threshold but plausible movements are `visual_uncertain` with candidates.
- **Why.** Only `visual_uncertain` tokens are safe to hand to a context resolver. `unknown` ones stay `?`.

## D5. Sentence speech policy

- **Problem.** If a sentence contains `?` gaps, speaking only the confident words can change the meaning ("I want ? please" becomes "I please").
- **Options.** Skip the gaps; hold the whole sentence.
- **Decision.** Default `skip` (the screen always shows the `?`), with a `hold` policy implemented and tested but not yet exposed in the UI.
- **Why.** Matches the existing rule that uncertain words are not spoken; avoids a silent app. The visible `?` keeps the user in control.
- **Replace.** Expose the policy as a setting once real use shows which one users prefer.

## D6. Resolver safety

- **Decision.** Every resolver call goes through `resolveSafely()` with a timeout. A failing or slow resolver returns the unmodified visual tokens flagged `fallback: true`.
- **Why.** The brief requires that the app stays usable when the AI layer fails.

## D7. Language-model strategy (built in Phase 4)

- **Problem.** Fill uncertain words from sentence context without letting a language model invent text.
- **Options.** Free-form generation; rewriting the whole sentence; ranking only the camera's own candidates.
- **Decision.** The resolver (`CandidateScoringResolver`) only ranks the visual candidates of `visual_uncertain` tokens. It combines `visual^1 * language^1`, renormalizes, and accepts the best only if its combined confidence reaches a threshold (default 0.6). `visual_confident` and `unknown` tokens are never touched. Replacements are recorded with their evidence and shown in [brackets?].
- **Why.** A scorer that can only pick among candidates cannot introduce a word the camera did not consider, and every correction is measurable (false-correction rate).
- **Trade-offs.** It cannot fix a confident-but-wrong visual word, and it cannot fill a gap whose true word is not among the three candidates. Correcting confident words is intentionally not implemented; the brief allows it only with strong evidence, and we have no measured basis for that yet.
- **Replace.** The `LanguageScorer` interface is the seam: any engine returning one probability per candidate fits.

## D9. Two language engines

- **Built-in phrase model** (default, ~100 hand-written phrases, bigram scoring, no download, instant). Honest limit: it only knows phrases like those in `src/context/corpus.ts`.
- **In-browser LLM** (WebLLM, Qwen2.5-0.5B-Instruct q4f16_1; about 945 MB GPU memory per the library's model list; WebGPU required). Used as a multiple-choice ranker: the prompt lists candidates as A/B/C and we read the probabilities of the answer letters from the first token's logprobs. It never writes free text. Small models favour letter positions, so each question is asked in the original and reversed order and averaged (doubles LLM calls; can be disabled with `debias`).
- **Why a 0.5B model.** Smallest option in the prebuilt list that follows a multiple-choice format; larger ones raise download, memory and latency. The choice is untested on this task and should be revisited with measurements.
- **Not chosen.** A local service (Ollama) breaks the browser-only goal; an external API sends text off the device and needs a key. Neither is built.

## D10. Privacy of the LLM path

- Video and mouth features never leave the device. The only network traffic added is the one-time download of model files from huggingface.co after an explicit button press. Candidate words are given to the local model only. Nothing is stored beyond the browser's own model cache.

## D11. Local Ollama engine (measured on the project owner's machine, Ollama 0.33.3)

- **Problem.** The user already runs Ollama, which can host far stronger models than the in-browser 0.5B option.
- **Decision.** Added `OllamaScorer` as a third engine, using the same multiple-choice question and logprob reading as the in-browser engine (shared in `choicePrompt.ts`). Default model: the first `qwen3:*` found, else the first chat model. `think:false` is sent only to models that report the "thinking" capability.
- **Verified against the real server.** `/api/chat` returns per-token `logprobs` with `top_logprobs`. CORS preflight from `http://localhost:5173` is allowed by default. The first request takes about 20 s while the model loads; `keep_alive: 10m` keeps it warm.
- **Measured (12 synthetic gap questions, small sample, NOT an accuracy claim).**

  | Model | single order | both orders averaged | confident (>=0.6) answers wrong | latency per call (median / p90) |
  | --- | --- | --- | --- | --- |
  | llama3.1:8b | 7/12 | 8/12 | 0 of 5 | 967 ms / 1171 ms |
  | qwen3:8b | 10/12 | 12/12 | 0 of 10 | 1800 ms / 2997 ms |

  Averaging both candidate orders helped both models, so it stays on by default. The two calls run in parallel; end-to-end, one gap took about 1.8 s and two gaps about 2.8 s.
- **Observed behaviour.** On genuinely ambiguous gaps the model is honestly unsure (e.g. "I want ___": water 0.53 vs waiter 0.47), and the 0.6 acceptance threshold then leaves the "?" in place. That is the intended safety behaviour, not a failure.
- **Trade-offs.** Roughly 2-3 s added when a sentence has an uncertain word. Sentences with none are unaffected. The sentence-timeout and the resolver deadline (12 s for Ollama) apply.
- **Privacy.** Sentence text with one gap, the candidate words and up to two earlier sentences go to the configured Ollama URL, default this computer. The UI warns when the URL is not local. Never video or mouth data.
- **Unverified.** Behaviour when the page is served over https from another origin to `http://localhost:11434` (browser local-network permission prompts, `OLLAMA_ORIGINS`); behaviour on other machines and GPUs.

## D8. Open-vocabulary visual model (deferred)

- **Open questions.** License of the pretrained weights (the Auto-AVSR family states non-commercial terms in at least one repository), ONNX export of a 3D-conv + Conformer model, WebGPU operator support, and whether models trained on filmed speech work on silent mouthing.
- **Plan.** Phase 0 spike on the user's machine; record the numbers here.

## Status of unmeasured claims

- LLM latency, download size, real-world accuracy gain and false-correction rate on real lip movements are NOT measured. The only numbers so far are from a synthetic in-sample check (tests/phase4.test.ts) and prove the mechanism, not the benefit.
