// Vendors MediaPipe WASM + face model into public/ so the app has no runtime CDN dependency.
import { cpSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';

const wasmSrc = 'node_modules/@mediapipe/tasks-vision/wasm';
if (existsSync(wasmSrc)) {
  cpSync(wasmSrc, 'public/wasm', { recursive: true });
  console.log('copied wasm');
}
const model = 'public/models/face_landmarker.task';
if (!existsSync(model)) {
  mkdirSync('public/models', { recursive: true });
  const url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
  const res = await fetch(url);
  if (!res.ok) { console.error('model download failed', res.status); process.exit(1); }
  writeFileSync(model, Buffer.from(await res.arrayBuffer()));
  console.log('downloaded model');
}
