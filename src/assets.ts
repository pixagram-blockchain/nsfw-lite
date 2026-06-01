import type { PreprocessConfig } from "./types.js";

/**
 * Shape of a per-variant `src/variants/<v>/assets.generated.ts` module. The
 * embed step (scripts/embed-assets.mjs) overwrites that file with the real
 * base64 model + labels/preprocess; a freshly cloned repo ships a stub with
 * `EMBEDDED = false` (imports + type-checks fine, throws only at runtime).
 */
export interface GeneratedAssets {
  MODEL_B64: string;
  LABELS: readonly string[];
  PREPROCESS: Readonly<Record<string, unknown>>;
  EMBEDDED: boolean;
}

/** Resolved, ready-to-use embedded assets for a single model variant. */
export interface Embedded {
  /** Decoded INT8 ONNX bytes. Throws if this variant has no embedded model. */
  getModelBytes(): Uint8Array;
  /** Class label order (lowercased), e.g. ["nsfw", "sfw"]. */
  labels: string[];
  /** Preprocessing parameters merged over DEFAULT_PREPROCESS. */
  preprocess: PreprocessConfig;
}

/**
 * timm preprocessing fallback. These are a fallback ONLY — the build pipeline
 * (scripts/export_model.py) reads the REAL values from each backbone's data
 * config and the embed step bakes them into assets.generated.ts, overriding
 * these. They are deliberately generic and are NOT correct for every backbone:
 * e.g. `tinynet_e` is 106×106 (not 224) and `lcnet_050` uses inception
 * mean/std (0.5/0.5/0.5). They only take effect if a variant somehow has no
 * embedded preprocess — and such a variant has no embedded model either, so it
 * throws at getModelBytes() before any inference runs. Do not rely on them for
 * accuracy; verify against the emitted preprocess.json.
 */
export const DEFAULT_PREPROCESS: PreprocessConfig = {
  size: 224, // generic fallback only — real size comes from preprocess.json (tinynet_e = 106)
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

export function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/**
 * Resolve a variant's generated module into ready-to-use {@link Embedded}
 * assets: merge its preprocess over {@link DEFAULT_PREPROCESS}, prefer its
 * labels (falling back to {@link DEFAULT_LABELS}), and defer the
 * "no model embedded" error to {@link Embedded.getModelBytes} call time so an
 * un-built variant still imports and type-checks.
 */
export function makeEmbedded(gen: GeneratedAssets): Embedded {
  const labels =
    gen.LABELS && gen.LABELS.length ? [...gen.LABELS] : [...DEFAULT_LABELS];
  const preprocess: PreprocessConfig = {
    ...DEFAULT_PREPROCESS,
    ...(gen.PREPROCESS as Partial<PreprocessConfig>),
  };
  return {
    labels,
    preprocess,
    getModelBytes(): Uint8Array {
      if (!gen.EMBEDDED || !gen.MODEL_B64) {
        throw new Error(
          "@pixagram/nsfw-lite: no model is embedded. Train + export your .onnx and " +
            "run `npm run embed-model` (see README), or pass options.modelBytes."
        );
      }
      return b64ToBytes(gen.MODEL_B64);
    },
  };
}
