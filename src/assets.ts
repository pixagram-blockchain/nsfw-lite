import {
  MODEL_B64,
  LABELS as GEN_LABELS,
  PREPROCESS as GEN_PREPROCESS,
  EMBEDDED,
} from "./assets.generated.js";
import type { PreprocessConfig } from "./types.js";

/**
 * MobileNetV4 (timm) preprocessing fallback. These are a fallback only — the
 * build pipeline (scripts/export_model.py) reads the REAL values from the timm
 * checkpoint's data config and overrides these. Do not rely on them for
 * accuracy without verifying against the emitted preprocess.json.
 */
export const DEFAULT_PREPROCESS: PreprocessConfig = {
  size: 224,
  cropSize: null,
  doCenterCrop: false,
  rescaleFactor: 1 / 255,
  rescaleOffset: false,
  doNormalize: true,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  includeTop: false,
};

/**
 * Fallback label order. ImageFolder sorts class folders alphabetically, so for
 * folders `nsfw/` and `sfw/` the order is ["nsfw", "sfw"] (nsfw == index 0).
 * The REAL order is emitted into labels.json by the build pipeline; this is the
 * sensible default but always prefer the embedded value.
 */
export const DEFAULT_LABELS = ["nsfw", "sfw"];

export const EMBEDDED_LABELS: string[] =
  GEN_LABELS && GEN_LABELS.length ? GEN_LABELS : DEFAULT_LABELS;

export const EMBEDDED_PREPROCESS: PreprocessConfig = {
  ...DEFAULT_PREPROCESS,
  ...(GEN_PREPROCESS as Partial<PreprocessConfig>),
};

export function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

export function getModelBytes(): Uint8Array {
  if (!EMBEDDED || !MODEL_B64) {
    throw new Error(
      "@pixagram/nsfw-lite: no model is embedded. Train + export your .onnx and " +
        "run `npm run embed-model` (see README), or pass options.modelBytes."
    );
  }
  return b64ToBytes(MODEL_B64);
}
