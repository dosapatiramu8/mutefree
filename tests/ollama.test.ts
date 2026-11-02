import { describe, expect, it } from 'vitest';
import { OllamaScorer, isLocalUrl } from '../src/context/ollamaScorer';
import { buildMessages, debiased, lettersToProbs } from '../src/context/choicePrompt';

type Handler = (url: string, body: Record<string, unknown> | undefined) => unknown | Response;

function fakeFetch(h: Handler) {
  const calls: { url: string; body?: Record<string, unknown> }[] = [];
  const f = async (url: string, init?: RequestInit): Promise<Response> => {
    const body = init?.body ? (JSON.parse(init.body as string) as Record<string, unknown>) : undefined;
    calls.push({ url, body });
    const out = h(url, body);
    if (out instanceof Response) return out;
    return new Response(JSON.stringify(out), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  return { f, calls };
}

const tags = {
  models: [
    { name: 'llama3.1:8b', size: 4.9e9, capabilities: ['completion', 'tools'] },
    { name: 'qwen3:8b', size: 5.2e9, capabilities: ['completion', 'tools', 'thinking'] },
    { name: 'nomic-embed-text:latest', size: 3e8, capabilities: ['embedding'] },
  ],
};
const lp = (letter: string) => ({ logprobs: [{ top_logprobs: [{ token: letter, logprob: -0.05 }, { token: 'C', logprob: -7 }] }] });

describe('OllamaScorer', () => {
  it('lists chat models only, prefers qwen3, and notes which can think', async () => {
    const { f } = fakeFetch(() => tags);
    const s = new OllamaScorer('http://localhost:11434', f);
    await s.connect();
    expect(s.models.map((m) => m.name)).toEqual(['llama3.1:8b', 'qwen3:8b']);
    expect(s.model).toBe('qwen3:8b');
    expect(s.state).toBe('ready');
  });

  it('sends a one-token logprobs request and turns thinking off for reasoning models', async () => {
    const { f, calls } = fakeFetch((url) => (url.endsWith('/api/tags') ? tags : lp('A')));
    const s = new OllamaScorer('http://localhost:11434', f);
    await s.connect();
    await s.score({ left: ['I', 'want'], right: [], candidates: ['water', 'later'], history: [] });
    const chat = calls.filter((c) => c.url.endsWith('/api/chat'));
    expect(chat).toHaveLength(2); // both candidate orders
    expect(chat[0].body).toMatchObject({ model: 'qwen3:8b', stream: false, logprobs: true, think: false, options: { temperature: 0, num_predict: 1 } });
    const user = (chat[0].body!.messages as { content: string }[])[1].content;
    expect(user).toContain('A) water'); expect(user).toContain('B) later');
    expect((chat[1].body!.messages as { content: string }[])[1].content).toContain('A) later'); // reversed
  });

  it('does not send think for models that cannot think', async () => {
    const { f, calls } = fakeFetch((url) => (url.endsWith('/api/tags') ? tags : lp('A')));
    const s = new OllamaScorer('http://localhost:11434', f);
    await s.connect('llama3.1:8b');
    await s.score({ left: [], right: [], candidates: ['a', 'b'], history: [] });
    expect(calls.find((c) => c.url.endsWith('/api/chat'))!.body).not.toHaveProperty('think');
  });

  it('cancels letter bias: a model that always says A shows no preference', async () => {
    const { f } = fakeFetch((url) => (url.endsWith('/api/tags') ? tags : lp('A')));
    const s = new OllamaScorer('http://localhost:11434', f);
    await s.connect();
    const p = await s.score({ left: [], right: [], candidates: ['water', 'later'], history: [] });
    expect(Math.abs(p[0] - p[1])).toBeLessThan(0.05);
  });

  it('explains an unreachable server and CORS', async () => {
    const s = new OllamaScorer('http://localhost:11434', async () => { throw new TypeError('Failed to fetch'); });
    await expect(s.connect()).rejects.toThrow(/Cannot reach Ollama.*OLLAMA_ORIGINS/);
    expect(s.state).toBe('error');
  });

  it('reports a server that returns no logprobs', async () => {
    const { f } = fakeFetch((url) => (url.endsWith('/api/tags') ? tags : { message: { content: 'A' } }));
    const s = new OllamaScorer('http://localhost:11434', f);
    await s.connect();
    await expect(s.score({ left: [], right: [], candidates: ['a', 'b'], history: [] })).rejects.toThrow(/logprobs/);
  });

  it('reports HTTP errors with the server message', async () => {
    const s = new OllamaScorer('http://localhost:11434', async () => new Response(JSON.stringify({ error: 'model not found' }), { status: 404 }));
    await expect(s.connect()).rejects.toThrow(/HTTP 404: model not found/);
  });

  it('refuses to score before connecting', async () => {
    await expect(new OllamaScorer().score({ left: [], right: [], candidates: ['a', 'b'], history: [] })).rejects.toThrow(/not connected/);
  });
});

describe('local URL check', () => {
  it('flags non-local hosts so the UI can warn that text would leave the device', () => {
    expect(isLocalUrl('http://localhost:11434')).toBe(true);
    expect(isLocalUrl('http://127.0.0.1:11434')).toBe(true);
    expect(isLocalUrl('http://192.168.1.20:11434')).toBe(false);
    expect(isLocalUrl('https://example.com')).toBe(false);
    expect(isLocalUrl('not a url')).toBe(false);
  });
});

describe('shared choice prompt', () => {
  it('labels candidates and marks the gap', () => {
    const m = buildMessages({ left: ['I', 'want'], right: ['please'], candidates: ['water', 'later'], history: ['Hello.'] }, ['water', 'later']);
    expect(m[1].content).toContain('Earlier sentences: Hello.');
    expect(m[1].content).toContain('"I want ____ please"');
  });
  it('averages both orders back into the original candidate order', async () => {
    // a model that answers by letter, always preferring the candidate "x" wherever it sits
    const p = await debiased(['x', 'y', 'z'], async (ordered) => lettersToProbs([{ token: 'ABC'[ordered.indexOf('x')], logprob: -0.01 }], 3));
    expect(p[0]).toBeGreaterThan(0.9);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });
});
