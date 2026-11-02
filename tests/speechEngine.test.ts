import { describe, expect, it } from 'vitest';
import { BrowserSpeechEngine } from '../src/audio/speechEngine';
import type { Speaker } from '../src/audio/speaker';

function fakeSpeaker(queue: boolean) {
  const calls: { text: string; voice: unknown }[] = [];
  let end: (() => void) | undefined;
  let start: ((t: number) => void) | undefined;
  const s = {
    speak: (text: string, onStart: (t: number) => void, voice: unknown, onEnd?: () => void) => {
      calls.push({ text, voice });
      start = onStart;
      end = onEnd;
      return queue;
    },
    clear: () => end?.(),
  };
  return { s: s as unknown as Speaker, calls, finish: () => { start?.(123); end?.(); } };
}

describe('BrowserSpeechEngine', () => {
  it('resolves when the utterance ends and reports start time', async () => {
    const f = fakeSpeaker(true);
    const e = new BrowserSpeechEngine(f.s);
    let started = 0;
    let done = false;
    const p = e.speak('hello', { rate: 1.2, pitch: 0.9, volume: 0.5, onStart: (t) => { started = t; } }).then(() => { done = true; });
    expect(done).toBe(false);
    f.finish();
    await p;
    expect(done).toBe(true);
    expect(started).toBe(123);
    expect(f.calls[0].voice).toEqual({ rate: 1.2, pitch: 0.9, volume: 0.5 });
  });
  it('resolves immediately when speech is muted or unsupported', async () => {
    const f = fakeSpeaker(false);
    await expect(new BrowserSpeechEngine(f.s).speak('x')).resolves.toBeUndefined();
  });
  it('stop() releases a pending speak()', async () => {
    const f = fakeSpeaker(true);
    const e = new BrowserSpeechEngine(f.s);
    const p = e.speak('long sentence');
    e.stop();
    await expect(p).resolves.toBeUndefined();
  });
});
