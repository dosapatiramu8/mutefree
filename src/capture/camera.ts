export async function startCamera(video: HTMLVideoElement): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera API unavailable (needs https or localhost).');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

export function describeCameraError(e: unknown): string {
  const name = (e as DOMException)?.name;
  if (name === 'NotAllowedError') return 'Camera permission denied. Allow camera access and retry.';
  if (name === 'NotFoundError') return 'No camera found.';
  if (name === 'NotReadableError') return 'Camera is in use by another application.';
  return `Camera error: ${(e as Error)?.message ?? e}`;
}
