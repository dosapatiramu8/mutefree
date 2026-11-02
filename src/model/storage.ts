import type { FeatureSet, Sample } from './classifier';

/** Everything needed to rebuild the model; raw samples are kept so retraining is cheap. */
export interface Saved {
  vocab: string[];
  samples: Sample[];
  featureSet: FeatureSet;
  accuracy: { geometry: number; 'geometry+extras': number };
  savedAt: number;
}

const DB = 'mutefree', STORE = 'kv', KEY = 'calibration';

function open(): Promise<IDBDatabase> {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function run<T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  return new Promise((res, rej) => {
    const req = f(db.transaction(STORE, mode).objectStore(STORE));
    req.onsuccess = () => { db.close(); res(req.result); };
    req.onerror = () => { db.close(); rej(req.error); };
  });
}

export const saveCalibration = (s: Saved) => run('readwrite', (st) => st.put(s, KEY)).then(() => undefined);
export const loadCalibration = () => run<Saved | undefined>('readonly', (st) => st.get(KEY));
export const deleteCalibration = () => run('readwrite', (st) => st.delete(KEY)).then(() => undefined);
