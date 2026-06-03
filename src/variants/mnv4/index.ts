/**
 * @pixagram/nsfw-lite/mnv4 — MobileNetV4-conv-small-050 variant.
 *
 * Fast on-device binary (sfw / nsfw) image classification for the browser,
 * powered by a MobileNetV4-conv-small-050 backbone via onnxruntime-web. Runs off-thread in a
 * Web Worker (the sibling dist/mnv4/worker.js) with an automatic main-thread
 * fallback; this build's uint8-quantized model is base64-embedded into this bundle.
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
