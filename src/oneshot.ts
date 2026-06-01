/**
 * oneshot.ts — convenience API backed by a lazily-created shared detector.
 * For repeated use, prefer creating and reusing your own NsfwDetector.
 */
import { NsfwDetector } from "./detector.js";
import type { DetectorOptions, ImageSource, NsfwResult } from "./types.js";

let singleton: Promise<NsfwDetector> | null = null;

function shared(opts?: DetectorOptions): Promise<NsfwDetector> {
  if (!singleton) singleton = NsfwDetector.create(opts);
  return singleton;
}

/** Load the model ahead of time (worker spawn + ORT session warm-up). */
export async function warmup(opts?: DetectorOptions): Promise<void> {
  await shared(opts);
}

/** Classify a single image with the shared detector. */
export async function classify(source: ImageSource, opts?: DetectorOptions): Promise<NsfwResult> {
  const detector = await shared(opts);
  return detector.classify(source);
}

/** Dispose the shared detector (next call recreates it). */
export function disposeShared(): void {
  if (!singleton) return;
  const pending = singleton;
  singleton = null;
  void pending.then((d) => d.dispose()).catch(() => undefined);
}
