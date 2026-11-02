import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

export interface Landmarker {
  detect(video: HTMLVideoElement, tsMs: number): FaceLandmarkerResult;
  delegate: 'GPU' | 'CPU';
}

export async function createLandmarker(): Promise<Landmarker> {
  const fileset = await FilesetResolver.forVisionTasks(`${import.meta.env.BASE_URL}wasm`);
  const make = (delegate: 'GPU' | 'CPU') =>
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    });
  let delegate: 'GPU' | 'CPU' = 'GPU';
  let lm: FaceLandmarker;
  try {
    lm = await make('GPU');
  } catch {
    delegate = 'CPU';
    lm = await make('CPU');
  }
  return { detect: (v, t) => lm.detectForVideo(v, t), delegate };
}
