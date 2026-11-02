import { startCamera, describeCameraError } from './capture/camera';
import { createLandmarker, type Landmarker } from './vision/landmarker';
import { extractStatic, FeatureStream, IDX_WIDTH, IDX_APERTURE } from './features/mouth';
import { ActivityDetector } from './activity/detector';
import { Harness } from './activity/harness';
import { MouthAppearance } from './features/appearance';
import { Calibration, type FaceInfo } from './ui/calibration';
import { Speaker } from './audio/speaker';
import { Recognition } from './ui/recognition';
import { SmartCompletion } from './ui/smart';
import { ExpressionTracker } from './features/expression';
import { mapExpression } from './audio/voice';
import { EvalTest } from './ui/evaltest';
import { Graph } from './ui/graph';
import { drawOverlay } from './ui/overlay';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video'), overlay = $<HTMLCanvasElement>('overlay');
const status = $('status'), startBtn = $<HTMLButtonElement>('start');
const showLm = $<HTMLInputElement>('showLandmarks');
const graph = new Graph($<HTMLCanvasElement>('graph'), [
  { name: 'width', cssVar: '--c1', scale: 1.2 },
  { name: 'aperture', cssVar: '--c2', scale: 0.5 },
  { name: 'energy', cssVar: '--c3', scale: 2 },
]);
const stream = new FeatureStream();
const detector = new ActivityDetector();
const appearance = new MouthAppearance();
const tiny = document.createElement('canvas'); tiny.width = 80; tiny.height = 60;
const tinyCtx = tiny.getContext('2d', { willReadFrequently: true })!;
const speaker = new Speaker();
const recognition = new Recognition(speaker);
new SmartCompletion(recognition.pipeline);
const expression = new ExpressionTracker();
try { const b = localStorage.getItem('mf-neutral'); if (b) expression.baseline = JSON.parse(b); } catch { /* storage unavailable */ }
recognition.exprWindow = (a, b) => expression.window(a, b);
recognition.exprLive = () => expression.live;
const setBar = (id: string, v: number, bi = false) => {
  const i = document.getElementById(id) as HTMLElement;
  if (bi) { i.style.left = v < 0 ? `${50 + v * 50}%` : '50%'; i.style.width = `${Math.abs(v) * 50}%`; }
  else { i.style.left = '0'; i.style.width = `${Math.min(Math.max(v, 0), 1) * 100}%`; }
};
$('neutralBtn').onclick = () => {
  const btn = $<HTMLButtonElement>('neutralBtn'); btn.disabled = true; btn.textContent = 'Hold a relaxed face…';
  expression.captureNeutral(1500, performance.now(), () => {
    btn.disabled = false; btn.textContent = 'Set neutral face (1.5 s)';
    try { localStorage.setItem('mf-neutral', JSON.stringify(expression.baseline)); } catch { /* ignore */ }
  });
};
const voiceSel = $<HTMLSelectElement>('rgVoice');
const refreshVoices = () => {
  const vs = speaker.voices();
  if (voiceSel.options.length !== vs.length) {
    voiceSel.replaceChildren(...vs.map((v) => { const o = document.createElement('option'); o.value = v.name; o.textContent = `${v.name} (${v.lang}${v.localService ? '' : ', online'})`; return o; }));
  }
  voiceSel.value = speaker.voiceInUse;
  $('rgVoiceStat').textContent = speaker.status;
};
voiceSel.onchange = () => speaker.setVoice(voiceSel.value);
speaker.onStatus = refreshVoices; refreshVoices();
recognition.onStopDelay = (ms) => { detector.opts.hangoverMs = ms; };
const calib = new Calibration((m) => recognition.setModel(m), () => harness.phase === 'countdown' || harness.phase === 'run');
calib.onSensitive = () => {
  const set = (id: string, v: number) => { const i = $<HTMLInputElement>(id); i.value = String(v); i.dispatchEvent(new Event('input')); };
  set('absMin', Math.max(0.02, +(detector.opts.absMin * 0.6).toFixed(3)));
  set('startMult', Math.max(1.5, +(detector.opts.startMult * 0.8).toFixed(2)));
};
const evalTest = new EvalTest(
  () => recognition.model,
  () => recognition.threshold,
  () => calib.active ? 'Finish calibration first.'
    : !lmk ? 'Start the camera first.'
    : (harness.phase === 'countdown' || harness.phase === 'run') ? 'Finish the M2 test first.' : null,
  () => ({ hangoverMs: detector.opts.hangoverMs, startMult: detector.opts.startMult, absMin: detector.opts.absMin, userAgent: navigator.userAgent }),
);
void calib.init();

// Surface unexpected failures instead of failing silently.
const banner = $('errBanner');
const showErr = (m: string) => { banner.textContent = m; banner.hidden = false; };
window.addEventListener('error', (e) => showErr(`Unexpected error: ${e.message}`));
window.addEventListener('unhandledrejection', (e) => showErr(`Unexpected error: ${(e.reason as Error)?.message ?? e.reason}`));
if (!window.isSecureContext) showErr('Camera needs https or localhost. Open the site over https.');
const badge = $('badge'), testOut = $('testOut'), runBtn = $<HTMLButtonElement>('runTest'), dlBtn = $<HTMLButtonElement>('dl');

let segCount = 0;
const harness = new Harness(
  detector,
  (title, body) => { testOut.textContent = `${title}\n${body}`; },
  () => {
    testOut.textContent = harness.summary();
    runBtn.disabled = false; dlBtn.disabled = false;
  },
);

// Threshold sliders
for (const [id, key] of [['startMult', 'startMult'], ['stopMult', 'stopMult'], ['absMin', 'absMin']] as const) {
  const el = $<HTMLInputElement>(id);
  const apply = () => { detector.opts[key] = +el.value; $(id + 'V').textContent = el.value; };
  el.addEventListener('input', apply); apply();
}

function setStatus(msg: string, err = false) { status.textContent = msg; status.classList.toggle('err', err); }

/** Face box + brightness from a tiny copy of the frame (used for lighting checks and mouth appearance). */
function faceInfo(lm: NonNullable<ReturnType<Landmarker['detect']>['faceLandmarks'][0]> | null): FaceInfo {
  if (!lm) return { hasFace: false, cx: 0, cy: 0, faceH: 0, luma: 0, leftLuma: 0, rightLuma: 0 };
  let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
  for (const p of lm) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
  tinyCtx.drawImage(video, 0, 0, 80, 60);
  const d = tinyCtx.getImageData(0, 0, 80, 60).data;
  const px0 = Math.max(0, Math.floor(x0 * 80)), px1 = Math.min(79, Math.ceil(x1 * 80));
  const py0 = Math.max(0, Math.floor(y0 * 60)), py1 = Math.min(59, Math.ceil(y1 * 60));
  const mid = (px0 + px1) / 2;
  let sum = 0, n = 0, ls = 0, ln = 0, rs = 0, rn = 0;
  for (let y = py0; y <= py1; y++) for (let x = px0; x <= px1; x++) {
    const k = (y * 80 + x) * 4, l = 0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2];
    sum += l; n++; if (x < mid) { ls += l; ln++; } else { rs += l; rn++; }
  }
  return { hasFace: true, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, faceH: y1 - y0, luma: n ? sum / n : 0, leftLuma: ln ? ls / ln : 0, rightLuma: rn ? rs / rn : 0 };
}

let lmk: Landmarker, lastVideoT = -1, frames = 0, fpsT = performance.now(), inferAcc = 0, inferN = 0;

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoT) return;
  lastVideoT = video.currentTime;
  if (overlay.width !== video.videoWidth) { overlay.width = video.videoWidth; overlay.height = video.videoHeight; }

  const t0 = performance.now();
  const res = lmk.detect(video, t0);
  inferAcc += performance.now() - t0; inferN++; frames++;

  const lm = res.faceLandmarks[0] ?? null;
  drawOverlay(overlay, lm, showLm.checked);
  const tongue = res.faceBlendshapes?.[0]?.categories.find((c) => c.categoryName === 'tongueOut')?.score ?? 0;
  const cats = res.faceBlendshapes?.[0]?.categories;
  if (cats) expression.push(cats, t0);
  const info = faceInfo(lm);
  calib.updateFace(info);
  calib.updateActivity(detector.state === 'ACTIVE', detector.smooth, detector.startThreshold);
  const app = lm ? appearance.compute(video, lm, info.luma) : undefined;
  const stat = lm && extractStatic(lm, video.videoWidth, video.videoHeight, tongue, app);
  let energy = 0, seg = null as ReturnType<ActivityDetector['push']>;
  if (stat) {
    const f = stream.push(stat, t0);
    energy = f.energy;
    seg = detector.push(f.vec, energy, t0);
    calib.pushFrame(f.vec, t0);
    graph.push([stat[IDX_WIDTH], stat[IDX_APERTURE], energy]);
    $('face').textContent = 'yes'; setStatus('Face tracked.');
  } else {
    // Face lost: never leave the detector latched on stale data.
    stream.reset(); detector.reset(); graph.push([0, 0, 0]);
    $('face').textContent = 'no'; setStatus('No face detected. Face the camera in good light.', true);
  }
  if (seg) {
    segCount++;
    if (evalTest.active) evalTest.onSegment(seg, performance.now());
    else if (calib.active) calib.onSegment(seg);
    else if (harness.phase === 'idle' || harness.phase === 'done') recognition.handle(seg, performance.now());
  }
  harness.tick(t0, energy, !!stat, seg);
  evalTest.tick(t0);
  recognition.tick(t0);

  const ex = expression.live;
  setBar('mSmile', Math.max(ex.smile - ex.frown, 0)); setBar('mBrow', ex.browUp - ex.browDown, true); setBar('mJaw', Math.min(ex.jaw * recognition.gain, 1));
  $('vSmile').textContent = ex.frown > ex.smile ? `frown ${ex.frown.toFixed(2)}` : ex.smile.toFixed(2);
  $('vBrow').textContent = (ex.browUp - ex.browDown).toFixed(2); $('vJaw').textContent = ex.jaw.toFixed(2);
  const vp = mapExpression(ex, recognition.gain);
  $('voiceNow').textContent = recognition.expressive ? `voice right now → pitch ${vp.pitch.toFixed(2)} · rate ${vp.rate.toFixed(2)} · volume ${vp.volume.toFixed(2)}` : 'expressive mode off: flat voice';
  badge.textContent = detector.state === 'ACTIVE' ? 'MOUTHING' : 'IDLE';
  badge.classList.toggle('on', detector.state === 'ACTIVE');
  $('thr').textContent = `energy ${detector.smooth.toFixed(2)} · start>${detector.startThreshold.toFixed(2)} stop<${detector.stopThreshold.toFixed(2)} · baseline ${detector.baseline.toFixed(3)} · segments ${segCount}`;
  graph.draw();

  const now = performance.now();
  if (now - fpsT > 1000) {
    $('fps').textContent = (frames * 1000 / (now - fpsT)).toFixed(0);
    $('infer').textContent = (inferAcc / Math.max(inferN, 1)).toFixed(1);
    frames = 0; inferAcc = 0; inferN = 0; fpsT = now;
  }
}

runBtn.addEventListener('click', () => {
  if (!lmk) { testOut.textContent = 'Start the camera first.'; return; }
  runBtn.disabled = true; dlBtn.disabled = true;
  harness.start(performance.now());
});
dlBtn.addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([harness.json()], { type: 'application/json' }));
  a.download = `mutefree-m2-${Date.now()}.json`; a.click();
});

startBtn.addEventListener('click', async () => {
  speaker.prime(); // speech needs a user gesture
  startBtn.disabled = true; setStatus('Loading model…');
  try {
    lmk = await createLandmarker();
  } catch (e) {
    setStatus(`Could not load the face model (${(e as Error)?.message ?? e}). Check your connection/WebGL and reload; run "npm run setup" if the model files are missing.`, true);
    startBtn.disabled = false; return;
  }
  $('delegate').textContent = lmk.delegate;
  try {
    await startCamera(video);
  } catch (e) {
    setStatus(describeCameraError(e), true); startBtn.disabled = false; return;
  }
  startBtn.textContent = 'Running';
  loop();
});
