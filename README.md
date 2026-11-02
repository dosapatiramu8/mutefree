# MuteFree

Live lip-to-voice in the browser. You silently mouth words at your webcam; the app
recognizes them from a small vocabulary you calibrated yourself and speaks them aloud,
with the voice's pitch, speed and volume following your facial expression.

Everything runs client-side: no server, no API keys, no video upload. Calibration data
lives in your browser's IndexedDB and can be deleted from the page.

## Run

```bash
npm install      # also copies MediaPipe WASM and downloads the 3.7 MB face model into public/
npm run dev      # http://localhost:5173
npm run build    # static site in dist/ (relative paths; host anywhere over https)
```

If the model files are missing (offline install), run `npm run setup` once with a connection.
The camera needs `https` or `localhost`. Chrome/Edge recommended.

## Use

1. **Start camera** and allow access.
2. **Calibration** → *Quick start* (8 words × 3 samples, about 1 min) or *Full* (25 words × 5).
   Pass the lighting check, then for each word hold the button (or Space) while mouthing it.
   The confusion matrix shows which words get mixed up; re-record or swap weak words.
3. Mouth a calibrated word and pause. It is spoken if it is confident enough, otherwise `?` is shown and nothing is said.
4. **Set neutral face**, then smile / frown / raise or lower your brows while mouthing: the voice changes.
5. **Accuracy test** measures real live accuracy (see below).

## How it works

| Stage | File | Notes |
|---|---|---|
| Landmarks + blendshapes | `src/vision/landmarker.ts` | MediaPipe Face Landmarker, GPU with CPU fallback |
| Mouth features | `src/features/mouth.ts` | Lip contours in a face-local frame, scaled by inter-eye distance (pose/distance invariant), plus velocity, `tongueOut`, and inner-mouth appearance (`appearance.ts`) |
| Speech activity | `src/activity/detector.ts` | Hysteresis on lip-motion energy with an adaptive idle baseline; hold-to-record bypasses it during calibration |
| Recognizer | `src/model/classifier.ts` | Each word resampled to 24 frames, z-scored, banded DTW + nearest neighbour on *your* samples. Confidence = margin between the nearest and second-nearest word. Too far from your samples → `?`. Trains in under a second |
| Expression | `src/features/expression.ts` | Smile, frown, brows, jaw from blendshapes, averaged over the time the word was mouthed, neutral-face baseline |
| Voice | `src/audio/voice.ts`, `speaker.ts` | `speechSynthesis`. Mapping: smile ↑ pitch/rate, frown ↓, brow-down ↓ pitch and rate, brow-up ↑ pitch, bigger mouth ↑ volume. Smoothed across words; stale queued words are dropped |

Calibration compares geometry-only features against geometry + tongue + inner-mouth
appearance and keeps whichever cross-validates better, so those extra features are only
used if they help *for you*.

## Honest limits

- **Lips alone are ambiguous.** Many sounds look identical, so the vocabulary is small and fixed by design. Look-alike word pairs are flagged (a rough spelling-based guess; the confusion matrix on your own samples is the real test).
- **Speaker-dependent.** It works after you calibrate on your own lips; it will not work for someone else.
- **Lighting / angle.** Even light, face centered and front-on. A lighting check runs before calibration.
- **Tongue.** Only the tongue-out blendshape and visible-interior appearance are available; tongue position behind the teeth is not observable from a webcam.
- **Latency.** The word is only known after your mouth stops (the *word-end delay*, default 350 ms), then classification and the browser's voice start-up are added. The on-screen latency panel reports the measured value; the 500 ms target is not guaranteed and depends on your browser and voice.
- **Voice quality / expressiveness** depends on the OS voice; some online voices ignore pitch and rate. Pick a local voice in the Voice dropdown.
- **Streaming** here means word-by-word at the end of each word, not mid-word.

## Measured results

None are recorded in this repository. Accuracy depends on the user, lighting and vocabulary,
and has to be measured by you: run the **Accuracy test** (≥100 cued words). It reports
correct / wrong word / shown as `?` / never detected, a 95% interval, and mouth-stop-to-decision
latency, and exports CSV/JSON. Add your own numbers below.

| Date | User | Vocabulary | Samples/word | Cued | Correct | Wrong | `?` | Missed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

## Development notes

- Vite + TypeScript, one runtime dependency (`@mediapipe/tasks-vision`).
- `npm run build` type-checks first (`tsc --noEmit`).
- The M2 test panel (speech-activity test with head movement) can be used to tune the detector thresholds.
- Not implemented: open-vocabulary mode, Hindi/Telugu packs, higher-quality TTS, optional LLM suggestions.
