import { describe, expect, it } from 'vitest';
import { SentenceBuffer, displayText, spokenText } from '../src/context/sentenceBuffer';
import { tok } from './helpers';

describe('SentenceBuffer', () => {
  it('does not finalize an empty buffer', () => {
    expect(new SentenceBuffer({ timeoutMs: 1000, maxTokens: 5 }).shouldFinalize(99999)).toBe(false);
  });
  it('waits for the pause after the last word', () => {
    const b = new SentenceBuffer({ timeoutMs: 1000, maxTokens: 5 });
    b.add(tok('I', 'visual_confident', 1000)); b.add(tok('want', 'visual_confident', 2000));
    expect(b.shouldFinalize(2900)).toBe(false); // short pause: more speech likely
    expect(b.shouldFinalize(3000)).toBe(true);
  });
  it('finalizes a very long sentence regardless of pause', () => {
    const b = new SentenceBuffer({ timeoutMs: 1000, maxTokens: 3 });
    for (let i = 1; i <= 3; i++) b.add(tok('x', 'visual_confident', i * 100));
    expect(b.shouldFinalize(350)).toBe(true);
  });
  it('take() drains the buffer', () => {
    const b = new SentenceBuffer({ timeoutMs: 1000, maxTokens: 5 });
    b.add(tok('hi', 'visual_confident', 100));
    expect(b.take()).toHaveLength(1);
    expect(b.size).toBe(0);
  });
});

describe('text rendering', () => {
  const mixed = [tok('I', 'visual_confident', 1), tok('want', 'visual_confident', 2), tok('?', 'visual_uncertain', 3), tok('please', 'visual_confident', 4)];
  it('shows unresolved tokens as ? and inferred ones bracketed', () => {
    expect(displayText(mixed)).toBe('I want ? please');
    expect(displayText([tok('I', 'visual_confident', 1), tok('water', 'context_inferred', 2)])).toBe('I [water?]');
  });
  it('speaks only words the system actually has', () => {
    expect(spokenText(mixed, 'skip')).toBe('I want please.');
    expect(spokenText(mixed, 'hold')).toBe('');
  });
  it('never speaks "?" or unknown tokens', () => {
    expect(spokenText([tok('?', 'unknown', 1), tok('?', 'visual_uncertain', 2)])).toBe('');
  });
  it('speaks context-inferred words', () => {
    const t = [tok('I', 'visual_confident', 1), tok('want', 'visual_confident', 2), tok('water', 'context_inferred', 3)];
    expect(spokenText(t)).toBe('I want water.');
  });
});
