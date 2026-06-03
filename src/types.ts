/**
 * Public types for @pixagram/nsfw-lite.
 *
 * Binary classifier: two classes, `sfw` and `nsfw`. An image is flagged when
 * P(nsfw) crosses a single threshold — far simpler than the multi-class gating
 * scheme it was adapted from.
 */

/** Single confidence gate. An image is flagged NSFW if P(nsfw) >= `nsfw`. */
export interface Thresholds {
  /** Flag when the softmax probability of the `nsfw` class >= this. Default 0.5. */
  nsfw: number;
}

/**
 * Preprocessing parameters. These MUST match the exported model's training-time
 * transforms; the build pipeline (scripts/export_model.py) reads them from the
 * timm checkpoint and emits the real values into the embedded assets.
 *
 * For MobileNetV4 (timm) the path is: plain resize -> rescale (1/255) ->
 * normalize(mean, std). The EfficientNet-only quirks (`rescaleOffset`,
 * `includeTop`) are kept for completeness but are off for this model.
 */
export interface PreprocessConfig {
  /** Square model input edge, e.g. 224. */
  size: number;
  /** If center-cropping: resize shorter side to this, then crop to `size`. */
  cropSize?: number | null;
  doCenterCrop?: boolean;
  /** Pixel rescale, e.g. 1/255. */
  rescaleFactor: number;
  /** Offset rescaled values by -1 (EfficientNet only; off here). */
  rescaleOffset?: boolean;
  doNormalize: boolean;
  mean: [number, number, number];
  std: [number, number, number];
  /** EfficientNet image-classification quirk: divide by std a SECOND time. Off here. */
  includeTop?: boolean;
  /**
   * Canvas resize filter, read from preprocess.json: "nearest" = pixelated
   * (smoothing off, crisp pixel blocks — the pixel-art choice), "bilinear" =
   * smooth (the default when absent). Set to match the filter the model was
   * trained/calibrated with so train/serve preprocessing stays in parity.
   */
  interpolation?: "nearest" | "bilinear";
}

/** Class label -> probability (labels are lowercased). Here: { sfw, nsfw }. */
export type NsfwScores = Record<string, number>;

export interface NsfwResult {
  /** True if P(nsfw) crossed the threshold. */
  nsfw: boolean;
  /** Softmax probabilities for both classes, keyed by lowercased label. */
  scores: NsfwScores;
  /** Highest-probability class. */
  top: { label: string; score: number };
  /** Which gate(s) tripped, e.g. ["nsfw>=0.5"]. Useful for debugging. */
  triggers: string[];
  /** Inference time in milliseconds. */
  ms: number;
  /** Backend that ran inference, e.g. "wasm+simd" or "webgpu". */
  backend: string;
}

/** Anything the main thread can turn into pixels. (The worker only sees pixels.) */
export type ImageSource =
  | ImageData
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | Blob
  | string; // URL (fetched then decoded)

export interface DetectorOptions {
  /** "auto" (default) uses a Worker when available, else main-thread. */
  useWorker?: "auto" | boolean;
  /** Execution provider: "auto" tries WebGPU then WASM (default). */
  backend?: "auto" | "webgpu" | "wasm";
  /** Where onnxruntime-web's own .wasm/.mjs are served from. */
  wasmPaths?: string | Record<string, string>;
  /** WASM threads. >1 requires cross-origin isolation (COOP/COEP). Default 1. */
  numThreads?: number;
  /** Max images per batched inference call (default 8). */
  maxBatch?: number;
  /** Coalescing window in ms: classify() calls within it run as one batch (default 12). */
  batchDelayMs?: number;
  /** Override the default NSFW gate. */
  thresholds?: Partial<Thresholds>;
  /** Provide your own model bytes instead of the embedded one. */
  modelBytes?: Uint8Array;
  /** Override embedded preprocessing parameters. */
  preprocess?: Partial<PreprocessConfig>;
  /** Override embedded class label order (must match the model's training order). */
  labels?: string[];
}
