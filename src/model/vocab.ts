/**
 * Vocabulary helpers. Visemes are a ROUGH guess derived from spelling, used only to
 * flag word pairs that are likely to look alike on the lips. The real test of
 * distinctness is the confusion matrix measured on the user's own samples.
 *
 * Codes: P bilabial (p b m) · F labiodental (f v) · T tongue-visible (th)
 *        R/O rounded (w r o u) · A open · E spread (e i y)
 *        L alveolar, tongue mostly hidden (t d n l) · S sibilant (s z c j x sh ch)
 *        K back, mostly invisible (k g q)   ('h' is dropped: no visible gesture)
 */
const LETTER: Record<string, string> = {
  p: 'P', b: 'P', m: 'P', f: 'F', v: 'F', w: 'R', r: 'R', o: 'O', u: 'O', a: 'A',
  e: 'E', i: 'E', y: 'E', t: 'L', d: 'L', n: 'L', l: 'L', s: 'S', z: 'S', c: 'S',
  j: 'S', x: 'S', k: 'K', g: 'K', q: 'K', h: '',
};
const DIGRAPH: Record<string, string> = {
  th: 'T', sh: 'S', ch: 'S', ph: 'F', ck: 'K', wh: 'R', kn: 'L', ng: 'K',
  oo: 'O', ou: 'O', ow: 'O', ee: 'E', ea: 'E', ai: 'A', ay: 'A',
};

export function visemes(word: string): string {
  let w = word.toLowerCase().replace(/[^a-z]/g, '');
  if (w.length > 3 && w.endsWith('e') && !'aeiouy'.includes(w[w.length - 2])) w = w.slice(0, -1);
  let out = '';
  for (let i = 0; i < w.length; i++) {
    const two = w.slice(i, i + 2);
    if (DIGRAPH[two]) { out += DIGRAPH[two]; i++; } else out += LETTER[w[i]] ?? '';
  }
  return out.replace(/(.)\1+/g, '$1');
}

function lev(a: string, b: string): number {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}

export interface Ambiguity { a: string; b: string; level: 'identical' | 'close'; }

/** Pairs whose viseme strings are identical or one edit apart. */
export function ambiguousPairs(words: string[]): Ambiguity[] {
  const v = words.map(visemes);
  const out: Ambiguity[] = [];
  for (let i = 0; i < words.length; i++)
    for (let j = i + 1; j < words.length; j++) {
      const d = lev(v[i], v[j]);
      if (d === 0) out.push({ a: words[i], b: words[j], level: 'identical' });
      else if (d === 1) out.push({ a: words[i], b: words[j], level: 'close' });
    }
  return out;
}

/** Words to offer as swaps. */
export const POOL = [
  'yes', 'no', 'hello', 'help', 'thanks', 'water', 'stop', 'I', 'you', 'want', 'need', 'please',
  'sorry', 'more', 'hot', 'cold', 'food', 'pain', 'home', 'doctor', 'okay', 'good', 'bad', 'call',
  'tired', 'go', 'come', 'sit', 'eat', 'drink', 'toilet', 'medicine', 'family', 'love', 'wait',
  'now', 'later', 'again', 'open', 'close', 'light', 'turn', 'phone', 'mother', 'father',
];

export function defaultVocab(): string[] {
  return [...DEFAULT_VOCAB];
}

// Chosen greedily from POOL: the core words first, then words that add no new look-alike
// pair. The core words themselves still contain flagged pairs (e.g. yes / I / you).
const DEFAULT_VOCAB: string[] = [
  'yes', 'no', 'hello', 'help', 'thanks', 'water', 'stop', 'I', 'you', 'want', 'need', 'please',
  'sorry', 'more', 'hot', 'doctor', 'okay', 'tired', 'drink', 'toilet', 'medicine', 'family',
  'again', 'open', 'turn',
];

/** Small, mutually distinct set for a ~1 minute first calibration. */
export const QUICK_VOCAB = ['yes', 'no', 'water', 'stop', 'thanks', 'please', 'toilet', 'medicine'];
