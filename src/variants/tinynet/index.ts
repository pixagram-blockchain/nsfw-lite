/**
 * @pixagram/nsfw-lite/tinynet — TinyNet-E variant.
 *
 * Fast on-device binary (sfw / nsfw) image classification for the browser,
 * powered by a TinyNet-E backbone via onnxruntime-web. Runs off-thread in a
 * Web Worker (the sibling dist/tinynet/worker.js) with an automatic main-thread
 * fallback; this variant's INT8 model is base64-embedded into this bundle.
 */
import { makeNsfwApi } from "../../api.js";
import type { NsfwDetectorInstance } from "../../api.js";
import { makeEmbedded } from "../../assets.js";
import * as generated from "./assets.generated.js";

const api = makeNsfwApi(makeEmbedded(generated));

// Value + type deliberately share the name `NsfwDetector` (declaration
// merging): consumers get both the `new`-free `NsfwDetector.create()` value and
// a usable `NsfwDetector` instance type.
export const NsfwDetector = api.NsfwDetector;
export type NsfwDetector = NsfwDetectorInstance;

export const classify = api.classify;
export const warmup = api.warmup;
export const disposeShared = api.disposeShared;

export type {
  DetectorOptions,
  ImageSource,
  NsfwResult,
  NsfwScores,
  PreprocessConfig,
  Thresholds,
} from "../../types.js";
