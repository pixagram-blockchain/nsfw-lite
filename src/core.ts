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
  /**
   * The backend that actually INITIALISED: "webgpu" or "wasm". On "webgpu",
   * individual ops without a GPU kernel (the quantize/dequantize nodes) still
   * run on CPU per-op inside the same session — that fallback is intrinsic to
   * ORT's graph partitioning, not something this label tracks.
   */
  backend: string;
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
  // No-op on ort-web >= ~1.19 (SIMD is always built in and the flag is
  // ignored); kept for older runtimes where it still matters.
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

/**
 * True only if WebGPU is genuinely usable here. `navigator.gpu` existing is NOT
 * enough: Chromium exposes the object even when the GPU is blocklisted, hardware
 * acceleration is off, or the platform rollout hasn't reached this machine —
 * requestAdapter() then resolves null. Only a non-null adapter proves it works.
 * Valid in both window and dedicated-worker scopes (workers without WebGPU
 * support simply have no `gpu` or return a null adapter).
 */
async function webgpuAvailable(): Promise<boolean> {
  try {
    const nav = (
      globalThis as { navigator?: { gpu?: { requestAdapter(): Promise<unknown> } } }
    ).navigator;
    const gpu = nav?.gpu;
    if (!gpu) return false;
    const adapter = await gpu.requestAdapter();
    return adapter !== null && adapter !== undefined;
  } catch {
    return false;
  }
}

export async function loadSession(
  bytes: Uint8Array,
  opts: RuntimeOptions = {}
): Promise<Session> {
  configureRuntime(opts);
  // Launch-time: learn once whether createImageBitmap honours "pixelated", so
  // the first real image doesn't pay for the probe (memoised; see core helpers).
  void probeBitmapCaps();

  const pref = opts.backend ?? "auto";
  const useGPU =
    pref === "webgpu" || (pref === "auto" && (await webgpuAvailable()));

  // Request ONE backend at a time. Per-op CPU fallback for the INT8/QDQ ops is
  // intrinsic to the session — ORT always registers the CPU EP last during
  // graph partitioning, so ["webgpu"] alone still runs unsupported
  // quantize/dequantize nodes on CPU within the same session. Listing "wasm"
  // alongside it added only silent BACKEND fallback at init time: when the
  // WebGPU backend failed to start, ORT quietly created the session on wasm
  // and this code still labelled it "webgpu" (and the catch below never ran).
  let session: ort.InferenceSession;
  let used: "webgpu" | "wasm" = useGPU ? "webgpu" : "wasm";
  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: [used],
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    // WebGPU backend couldn't initialise after all (device request refused,
    // adapter lost, etc.). Auto mode retries pure CPU; an explicit "webgpu"
    // request surfaces the error instead of hiding it.
    if (used === "webgpu" && pref !== "webgpu") {
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
  // `used` is the backend that actually initialised; with "webgpu" some ops may
  // still run on CPU via per-op fallback. The measured ms is the real signal of
  // how well it landed.
  return { session, inputName, outputName, backend: used };
}

/**
 * Reusable scratch canvases.
 *
 * Allocating an OffscreenCanvas + 2D context for every image is wasteful when
 * the whole point is to classify many images. We keep one canvas per role and
 * only resize it when the required dimensions change (resizing also clears it).
 * Sharing across calls is safe because resizeToSquare resolves all of its async
 * work (capability probe, bitmap decode) BEFORE touching a scratch canvas: the
 * fill -> draw -> read section is fully synchronous, so overlapping calls on
 * the main thread cannot interleave inside it. (The worker is sequential by
 * construction — one message at a time.)
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

// --- createImageBitmap capability probe (run once at launch) ----------------
// resizeQuality:"pixelated" is a browser-dependent HINT — some engines (notably
// Firefox, historically) silently ignore it and resize smoothly, which breaks
// nearest-neighbour parity for pixel art. We detect that empirically, once: a
// known black/white pattern upscaled with "pixelated" stays pure if the hint is
// honoured, and grows gray midtones if it fell back to smoothing. When it's not
// honoured (or createImageBitmap can't resize at all) we scale on a 2D canvas
// with imageSmoothingEnabled=false instead — the reliable nearest path.
interface BitmapCaps {
  /** createImageBitmap exists AND obeys resizeWidth/resizeHeight. */
  resize: boolean;
  /** resizeQuality:"pixelated" produces true nearest-neighbour (not smoothing). */
  pixelated: boolean;
}

let bitmapCapsPromise: Promise<BitmapCaps> | null = null;

function probeBitmapCaps(): Promise<BitmapCaps> {
  if (bitmapCapsPromise) return bitmapCapsPromise;
  bitmapCapsPromise = (async (): Promise<BitmapCaps> => {
    if (
      typeof createImageBitmap === "undefined" ||
      typeof OffscreenCanvas === "undefined"
    ) {
      return { resize: false, pixelated: false };
    }
    try {
      // 4 source px (black, white, black, white) upscaled 4× to 16 px wide.
      const probe = new Uint8ClampedArray(4 * 4); // 4 px · RGBA
      for (let i = 0; i < 4; i++) {
        const v = i % 2 === 0 ? 0 : 255;
        probe[i * 4] = v;
        probe[i * 4 + 1] = v;
        probe[i * 4 + 2] = v;
        probe[i * 4 + 3] = 255;
      }
      const W = 16;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const bmp = await createImageBitmap(new ImageData(probe as any, 4, 1), {
        resizeWidth: W,
        resizeHeight: 1,
        resizeQuality: "pixelated",
        premultiplyAlpha: "none",
      });
      const resize = bmp.width === W && bmp.height === 1;

      let pixelated = false;
      const cx = new OffscreenCanvas(W, 1).getContext("2d", {
        willReadFrequently: true,
      }) as OffscreenCanvasRenderingContext2D | null;
      if (cx && resize) {
        cx.imageSmoothingEnabled = false; // 1:1 readback blit; no scaling here
        cx.drawImage(bmp, 0, 0);
        const out = cx.getImageData(0, 0, W, 1).data;
        pixelated = true;
        for (let i = 0; i < out.length; i += 4) {
          // True nearest => every pixel is exactly 0 or 255 (allow ±1 rounding).
          if (out[i]! > 1 && out[i]! < 254) {
            pixelated = false;
            break;
          }
        }
      }
      bmp.close();
      return { resize, pixelated };
    } catch {
      // resize options unsupported, OffscreenCanvas readback blocked, etc.
      return { resize: false, pixelated: false };
    }
  })();
  return bitmapCapsPromise;
}

/**
 * Resize source pixels to the model's square input by LETTERBOXING: preserve the
 * source aspect ratio, fit the whole image inside S x S, and pad the margins with
 * cfg.padColor (default black). This is the ONLY fit mode — cropping would drop
 * content at the edges and squashing would distort wide images, both of which
 * cause missed detections in a moderation classifier; letterboxing keeps every
 * pixel at its true shape. Pad margins only appear for non-square inputs.
 *
 * Resize backend: createImageBitmap (off-thread, HW-accelerated) is used when it
 * can resize AND — for the "nearest" filter — actually honours "pixelated"
 * (decided once by probeBitmapCaps). Otherwise we scale on a 2D canvas with
 * imageSmoothingEnabled forced off, the reliable nearest path on engines that
 * ignore the hint. The scaled image is composited centred over the pad. For
 * bit-exact parity vs the Python pipeline, bake resize+normalize into the ONNX
 * graph.
 *
 * Ordering note: ALL awaits happen before the scratch canvases are touched, so
 * the pad-fill -> composite -> getImageData section is atomic with respect to
 * other in-flight resizes sharing the same scratches (see Scratch docs).
 */
async function resizeToSquare(
  px: PixelData,
  cfg: PreprocessConfig
): Promise<Uint8ClampedArray> {
  const S = cfg.size;
  const nearest = cfg.interpolation === "nearest";

  // Letterbox geometry: fit inside S x S, centre, pad the rest. Dims use round
  // and offsets use floor — mirrored by letterbox / RandomLetterbox in the
  // Python pipeline, so train, calibration, and serve share one geometry.
  const scale = Math.min(S / px.width, S / px.height);
  const dw = Math.max(1, Math.round(px.width * scale));
  const dh = Math.max(1, Math.round(px.height * scale));
  const ox = Math.floor((S - dw) / 2);
  const oy = Math.floor((S - dh) / 2);

  // Resolve everything asynchronous up front.
  const caps = await probeBitmapCaps();
  const useBitmap = caps.resize && (!nearest || caps.pixelated);
  let bmp: ImageBitmap | null = null;
  if (useBitmap) {
    // Cast to any: newer TS types Uint8ClampedArray as generic over
    // ArrayBufferLike (incl. SharedArrayBuffer) and rejects it for ImageData;
    // runtime is a plain ArrayBuffer. Type-only.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const srcData = new ImageData(px.data as any, px.width, px.height);
    bmp = await createImageBitmap(srcData, {
      resizeWidth: dw,
      resizeHeight: dh,
      resizeQuality: nearest ? "pixelated" : "medium",
      premultiplyAlpha: "none", // match getImageData's non-premultiplied RGBA
    });
  }

  // --- Synchronous from here: no await between pad fill and readback. ---
  const dst = getScratch("dst", S, S);
  const pad: [number, number, number] = cfg.padColor ?? [0, 0, 0];
  dst.ctx.fillStyle = `rgb(${pad[0]}, ${pad[1]}, ${pad[2]})`;
  dst.ctx.fillRect(0, 0, S, S);

  if (bmp) {
    dst.ctx.imageSmoothingEnabled = false; // 1:1 composite at (ox,oy); no scaling
    dst.ctx.drawImage(bmp, ox, oy);
    bmp.close();
  } else {
    // Canvas fallback: the scale happens in this drawImage. smoothing off =
    // nearest, on = the browser's smooth filter.
    const src = getScratch("src", px.width, px.height);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    src.ctx.putImageData(new ImageData(px.data as any, px.width, px.height), 0, 0);
    dst.ctx.imageSmoothingEnabled = !nearest;
    dst.ctx.drawImage(src.canvas, 0, 0, px.width, px.height, ox, oy, dw, dh);
  }

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
  // ort-web >= ~1.19 always ships SIMD and ignores (or no longer exposes)
  // env.wasm.simd — so a missing flag means SIMD is on, not off.
  try {
    return (ort.env.wasm as { simd?: boolean }).simd === false ? "" : "+simd";
  } catch {
    return "+simd";
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
