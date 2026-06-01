/**
 * @pixagram/nsfw-lite — fast on-device binary (sfw / nsfw) image classification
 * for the browser, powered by a MobileNetV4-small-050 backbone via onnxruntime-web.
 */
export { NsfwDetector } from "./detector.js";
export { classify, warmup, disposeShared } from "./oneshot.js";

export type {
  DetectorOptions,
  ImageSource,
  NsfwResult,
  NsfwScores,
  PreprocessConfig,
  Thresholds,
} from "./types.js";
