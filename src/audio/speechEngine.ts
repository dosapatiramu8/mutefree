import type { SpeechEngine, SpeechOptions } from '../core/types';
import type { Speaker } from './speaker';

/** SpeechEngine backed by the browser's speechSynthesis queue (Speaker). */
export class BrowserSpeechEngine implements SpeechEngine {
  constructor(private speaker: Speaker) {}

  speak(text: string, options: SpeechOptions = {}): Promise<void> {
    return new Promise<void>((resolve) => {
      const voice = options.rate !== undefined || options.pitch !== undefined || options.volume !== undefined
        ? { rate: options.rate ?? 1, pitch: options.pitch ?? 1, volume: options.volume ?? 1 }
        : undefined;
      const queued = this.speaker.speak(text, (t) => options.onStart?.(t), voice, resolve);
      if (!queued) resolve(); // muted or unsupported: nothing will play
    });
  }

  stop(): void { this.speaker.clear(); }
}
