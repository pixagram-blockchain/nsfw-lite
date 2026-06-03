/**
 * core.ts — runtime-agnostic inference helpers shared by the Web Worker and the
 * main-thread fallback. No DOM beyond OffscreenCanvas + createImageBitmap (both
 * available in window and worker scopes).
 */
// The /webgpu bundle includes BOTH the WebGPU and WASM execution providers,
// so we can prefer GPU and fall back to CPU from a single import.
import * as ort from "onnxruntime-web/webgpu";
import type { PreprocessConfig, Thresholds, NsfwResult } from "./types.js";

function now(): number {
  return typeof performance !== "undefined" && performance.now
    ? performance.now()
    : Date.now();
}

/** Raw pixels the main thread hands to inference. */
export interface PixelData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface Session {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
  backend: string; // the execution provider that actually started, e.g. "webgpu"
}

export interface RuntimeOptions {
  wasmPaths?: string | Record<string, string>;
  numThreads?: number;
  /** "auto" tries WebGPU then WASM (default). Force one with "webgpu"/"wasm". */
  backend?: "auto" | "webgpu" | "wasm";
}

let configured = false;
export function configureRuntime(opts: RuntimeOptions = {}): void {
  if (configured) return;
  // >1 thread requires cross-origin isolation (COOP/COEP); 1 is always safe.
  try {
    ort.env.wasm.numThreads = opts.numThreads ?? 1;
  } catch {
    /* ignore */
  }
  try {
    ort.env.wasm.simd = true;
  } catch {
    /* ignore */
  }
  // We run inside a worker already (or accept main-thread cost): no proxy worker.
  try {
    (ort.env.wasm as { proxy?: boolean }).proxy = false;
  } catch {
    /* ignore */
  }
  if (opts.wasmPaths) {
    try {
      ort.env.wasm.wasmPaths = opts.wasmPaths as never;
    } catch {
      /* ignore */
    }
  }
  configured = true;
}

export async function loadSession(
  bytes: Uint8Array,
  opts: RuntimeOptions = {}
): Promise<Session> {
  configureRuntime(opts);

  const pref = opts.backend ?? "auto";
  const hasGPU =
    typeof navigator !== "undefined" &&
    !!(navigator as unknown as { gpu?: unknown }).gpu;

  // For the GPU path, pass ["webgpu","wasm"] as ONE provider list. ORT runs each
  // op on the WebGPU EP when it has a kernel and falls back to CPU per-op
  // otherwise — within a single session. This is what lets an INT8/QDQ model run
  // on WebGPU at all: listing "webgpu" alone would fail on any unsupported
  // quantize/dequantize op and drop the whole model to CPU. Pure "wasm" is used
  // when the GPU is unwanted or the WebGPU API isn't present.
  const useGPU = pref === "webgpu" || (pref === "auto" && hasGPU);
  const providers = useGPU ? ["webgpu", "wasm"] : ["wasm"];

  let session: ort.InferenceSession;
  let used = useGPU ? "webgpu" : "wasm";
  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    // WebGPU couldn't initialise at all (no adapter, etc.). If we weren't forced
    // onto it, retry pure CPU; if the caller forced "webgpu", surface the error.
    if (useGPU && pref !== "webgpu") {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      used = "wasm";
    } else {
      throw e;
    }
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error("@pixagram/nsfw-lite: model has no input/output names");
  }
  // `used` is the PRIMARY provider; with the GPU path some ops may still run on
  // CPU via fallback. The measured ms is the real signal of how well it landed.
  return { session, inputName, outputName, backend: used };
}

/**
 * Reusable scratch canvases.
 *
 * Allocating an OffscreenCanvas + 2D context for every image is wasteful when
 * the whole point is to classify many images. We keep one canvas per role and
 * only resize it when the required dimensions change (resizing also clears it).
 * Sharing across calls is safe: a worker handles one message at a time, and in
 * classifyBatch the loop reads each canvas back with getImageData (which
 * copies) before the next iteration overwrites it. There is no `await` between
 * a draw and its read, so the draw+read is atomic even if classify() calls
 * overlap on the main thread.
 */
interface Scratch {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
}
const scratches: Record<string, Scratch> = {};

function getScratch(slot: string, w: number, h: number): Scratch {
  let s = scratches[slot];
  if (!s) {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("@pixagram/nsfw-lite: 2D canvas context unavailable");
    s = { canvas, ctx: ctx as OffscreenCanvasRenderingContext2D };
    scratches[slot] = s;
  } else if (s.canvas.width !== w || s.canvas.height !== h) {
    s.canvas.width = w; // resizing clears the canvas — which is what we want
    s.canvas.height = h;
  }
  return s;
}

/**
 * Resize source pixels to the model's square input.
 *
 * The plain-resize path (this model: doCenterCrop=false) uses createImageBitmap,
 * which scales off the main thread and can be hardware-accelerated. The resize
 * filter follows cfg.interpolation via resizeQuality: "nearest" -> "pixelated"
 * (crisp blocks, the pixel-art choice), otherwise "medium" (smooth). A single
 * 1:1 readback canvas then extracts the RGBA bytes (ImageBitmap pixels can't be
 * read without a canvas; there is no scaling on it, so smoothing is irrelevant).
 *
 * PARITY CAVEAT: resizeQuality is browser-dependent and not uniformly supported
 * — notably Firefox's support for it has been incomplete, so "pixelated" may be
 * silently ignored and fall back to a smooth resize, breaking nearest parity
 * there. Where pixel-exact nearest across browsers matters more than off-thread
 * decode, do the scaling on a canvas with imageSmoothingEnabled=false instead
 * (the readback canvas below already does the 1:1 blit). For bit-exact parity
 * vs the Python pipeline, bake resize+normalize into the ONNX graph.
 *
 * The center-crop branch (unused by the binary mnv4 model) stays on the canvas:
 * createImageBitmap crops-then-resizes whereas the Python side resizes-then-crops,
 * and replicating that exactly isn't worth it for a dead branch.
 */
async function resizeToSquare(
  px: PixelData,
  cfg: PreprocessConfig
): Promise<Uint8ClampedArray> {
  const S = cfg.size;
  const dst = getScratch("dst", S, S);

  if (cfg.doCenterCrop && cfg.cropSize) {
    const smooth = cfg.interpolation !== "nearest";
    const src = getScratch("src", px.width, px.height);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    src.ctx.putImageData(new ImageData(px.data as any, px.width, px.height), 0, 0);
    const c = cfg.cropSize;
    const scale = c / Math.min(px.width, px.height);
    const rw = Math.round(px.width * scale);
    const rh = Math.round(px.height * scale);
    const tmp = getScratch("tmp", rw, rh);
    tmp.ctx.imageSmoothingEnabled = smooth;
    tmp.ctx.clearRect(0, 0, rw, rh);
    tmp.ctx.drawImage(src.canvas, 0, 0, px.width, px.height, 0, 0, rw, rh);
    const cx = Math.floor((rw - S) / 2);
    const cy = Math.floor((rh - S) / 2);
    dst.ctx.imageSmoothingEnabled = smooth;
    dst.ctx.clearRect(0, 0, S, S);
    dst.ctx.drawImage(tmp.canvas, cx, cy, S, S, 0, 0, S, S);
    return dst.ctx.getImageData(0, 0, S, S).data;
  }

  // Plain square resize via createImageBitmap (off-thread, HW-accelerated).
  const quality: ResizeQuality =
    cfg.interpolation === "nearest" ? "pixelated" : "medium";
  // Cast to any: newer TS types Uint8ClampedArray as generic over ArrayBufferLike
  // (incl. SharedArrayBuffer) and rejects it for ImageData; runtime is a plain
  // ArrayBuffer. Type-only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const srcData = new ImageData(px.data as any, px.width, px.height);
  const bmp = await createImageBitmap(srcData, {
    resizeWidth: S,
    resizeHeight: S,
    resizeQuality: quality,
    premultiplyAlpha: "none", // match getImageData's non-premultiplied RGBA
  });
  // 1:1 blit purely to read pixels back; no scaling here, so smoothing is moot.
  dst.ctx.imageSmoothingEnabled = false;
  dst.ctx.clearRect(0, 0, S, S);
  dst.ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return dst.ctx.getImageData(0, 0, S, S).data;
}

/** RGBA bytes -> NCHW float32, applying rescale / offset / normalize / include_top. */
function toTensorData(rgba: Uint8ClampedArray, cfg: PreprocessConfig): Float32Array {
  const S = cfg.size;
  const area = S * S;
  const out = new Float32Array(3 * area);
  const rO = 0;
  const gO = area;
  const bO = 2 * area;

  const m = cfg.mean;
  const s = cfg.std;
  const rf = cfg.rescaleFactor;
  const offset = cfg.rescaleOffset === true;
  const normalize = cfg.doNormalize !== false;
  const top = cfg.includeTop === true;

  for (let p = 0, j = 0; p < area; p++, j += 4) {
    let r = rgba[j]! * rf;
    let g = rgba[j + 1]! * rf;
    let b = rgba[j + 2]! * rf;
    if (offset) {
      r -= 1;
      g -= 1;
      b -= 1;
    }
    if (normalize) {
      r = (r - m[0]) / s[0];
      g = (g - m[1]) / s[1];
      b = (b - m[2]) / s[2];
      if (top) {
        r /= s[0];
        g /= s[1];
        b /= s[2];
      }
    }
    out[rO + p] = r;
    out[gO + p] = g;
    out[bO + p] = b;
  }
  return out;
}

/**
 * Softmax logits -> per-class scores, then apply the single NSFW gate.
 *
 * The `nsfw` probability is looked up BY LABEL NAME (not by a hardcoded index),
 * because ImageFolder sorts class folders alphabetically — "nsfw" < "sfw" — so
 * the model's index 0 is `nsfw`. Reading by name keeps this correct no matter
 * how the labels were ordered at train time.
 */
export function decode(
  logits: Float32Array | number[],
  labels: string[],
  t: Thresholds
): Pick<NsfwResult, "scores" | "top" | "nsfw" | "triggers"> {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;

  let sum = 0;
  const exps = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i]! - max);
    exps[i] = e;
    sum += e;
  }

  const scores: Record<string, number> = {};
  let topLabel = (labels[0] ?? "0").toLowerCase();
  let topScore = -1;
  for (let i = 0; i < exps.length; i++) {
    const prob = exps[i]! / sum;
    const label = (labels[i] ?? String(i)).toLowerCase();
    scores[label] = prob;
    if (prob > topScore) {
      topScore = prob;
      topLabel = label;
    }
  }

  const nsfw = scores["nsfw"] ?? 0;
  const triggers: string[] = [];
  if (nsfw >= t.nsfw) triggers.push("nsfw>=" + t.nsfw);

  return {
    scores,
    top: { label: topLabel, score: topScore },
    nsfw: triggers.length > 0,
    triggers,
  };
}

function simdTag(): string {
  try {
    return ort.env.wasm.simd ? "+simd" : "";
  } catch {
    return "";
  }
}

/** Full pipeline for one image: preprocess -> run -> decode. */
export async function classifyImageData(
  sess: Session,
  px: PixelData,
  cfg: PreprocessConfig,
  labels: string[],
  thresholds: Thresholds
): Promise<NsfwResult> {
  const tStart = now();

  const rgba = await resizeToSquare(px, cfg);
  const data = toTensorData(rgba, cfg);
  const S = cfg.size;
  const tensor = new ort.Tensor("float32", data, [1, 3, S, S]);

  const feeds: Record<string, ort.Tensor> = {};
  feeds[sess.inputName] = tensor;
  const results = await sess.session.run(feeds);
  const output = results[sess.outputName];
  if (!output) throw new Error("@pixagram/nsfw-lite: missing model output");

  const decoded = decode(output.data as Float32Array, labels, thresholds);
  const backend = sess.backend === "wasm" ? "wasm" + simdTag() : sess.backend;

  return { ...decoded, ms: Math.round(now() - tStart), backend };
}

/**
 * Classify N images in a SINGLE inference. Each is preprocessed to [3,S,S] and
 * stacked into an [N,3,S,S] batch, so the (expensive) GPU upload / kernel launch
 * / readback is paid once for the whole batch instead of per image. The model's
 * batch axis is dynamic (see export), so any N works. Returns one result per
 * input, in order; `ms` is the wall time of the whole batched run.
 */
export async function classifyBatch(
  sess: Session,
  pxs: PixelData[],
  cfg: PreprocessConfig,
  labels: string[],
  thresholds: Thresholds
): Promise<NsfwResult[]> {
  const tStart = now();
  const n = pxs.length;
  if (n === 0) return [];

  const S = cfg.size;
  const stride = 3 * S * S; // floats per image (NCHW)
  const data = new Float32Array(n * stride);
  for (let i = 0; i < n; i++) {
    const rgba = await resizeToSquare(pxs[i]!, cfg);
    data.set(toTensorData(rgba, cfg), i * stride);
  }

  const tensor = new ort.Tensor("float32", data, [n, 3, S, S]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[sess.inputName] = tensor;
  const out = await sess.session.run(feeds);
  const output = out[sess.outputName];
  if (!output) throw new Error("@pixagram/nsfw-lite: missing model output");

  const all = output.data as Float32Array;
  const classesPerRow = Math.floor(all.length / n); // == 2 for the binary head
  const backend = sess.backend === "wasm" ? "wasm" + simdTag() : sess.backend;
  const ms = Math.round(now() - tStart);

  const results: NsfwResult[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const logits = all.subarray(i * classesPerRow, (i + 1) * classesPerRow);
    const decoded = decode(logits, labels, thresholds);
    results[i] = { ...decoded, ms, backend };
  }
  return results;
}
