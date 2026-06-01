/* eslint-disable no-restricted-globals */
/**
 * worker.ts — Web Worker entry.
 *
 * The model is NOT embedded here (that would double the bundled size). The main
 * thread transfers the decoded model bytes once at "init"; classify payloads
 * carry raw pixels (the worker has no DOM to decode images itself).
 *
 * Protocol (request/response keyed by reqId):
 *   init:         { reqId, modelBuffer: ArrayBuffer, cfg, labels, thresholds, opts }
 *   classify:     { reqId, payload: { data, width, height } }
 *   classifyBatch:{ reqId, payloads: [{ data, width, height }, ...] }
 *   ->            { reqId, ok: true, result } | { reqId, ok: true, results: [...] }
 *                 | { reqId, ok: false, error }
 */
import * as core from "./core.js";
import type { PreprocessConfig, Thresholds } from "./types.js";

// Typed view of the worker global that sidesteps DOM-vs-WebWorker lib conflicts.
const ctx = self as unknown as {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
};

let session: core.Session | null = null;
let cfg: PreprocessConfig | null = null;
let labels: string[] = [];
let thresholds: Thresholds = { nsfw: 0.5 };

ctx.onmessage = (event: MessageEvent) => {
  const msg = event.data || {};
  const reqId = msg.reqId;

  void (async () => {
    try {
      if (msg.type === "init") {
        cfg = msg.cfg as PreprocessConfig;
        labels = msg.labels as string[];
        if (msg.thresholds) thresholds = msg.thresholds as Thresholds;
        session = await core.loadSession(new Uint8Array(msg.modelBuffer), msg.opts || {});
        ctx.postMessage({ reqId, ok: true, backend: session.backend });
        return;
      }

      if (msg.type === "classify") {
        if (!session || !cfg) throw new Error("worker not initialized");
        const result = await core.classifyImageData(session, msg.payload, cfg, labels, thresholds);
        ctx.postMessage({ reqId, ok: true, result });
        return;
      }

      if (msg.type === "classifyBatch") {
        if (!session || !cfg) throw new Error("worker not initialized");
        const results = await core.classifyBatch(session, msg.payloads, cfg, labels, thresholds);
        ctx.postMessage({ reqId, ok: true, results });
        return;
      }
    } catch (err) {
      ctx.postMessage({
        reqId,
        ok: false,
        error: String((err && (err as Error).message) || err),
      });
    }
  })();
};
