# @pixagram/nsfw-lite

Fast, on-device **binary** (`sfw` / `nsfw`) image classification for the browser,
tuned for **pixel art**. About **65 ms per image on CPU** from a model **under
1 MB gzipped**.

A uint8-quantized **MobileNetV4-conv-small-050** runs through
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) at **160×160**,
off the main thread in a **Web Worker** (with an automatic **main-thread
fallback**). The model is **base64-embedded** in the bundle, so there's no
separate model fetch at runtime. Nothing leaves the device.

| Backbone | Input | Normalization | Quant | Model |
| --- | --- | --- | --- | --- |
| `mobilenetv4_conv_small_050` | 160×160 | inception (0.5/0.5/0.5) | uint8 | < 1 MB gzipped |

> **Heads up:** the repo ships the *code*, not weights. A fresh clone has an
> empty model stub and throws `no model is embedded` until you run the
> train → export → embed pipeline (see [Building your own model](#building-your-own-model)).
> The model is *your* model.

## Install

```bash
npm install @pixagram/nsfw-lite onnxruntime-web
```

`onnxruntime-web` is a **peer dependency** so your app controls the ORT version.

## Quick start

```ts
import { classify } from "@pixagram/nsfw-lite";

const result = await classify(myImageElement);
// {
//   nsfw: true,
//   scores: { sfw: 0.04, nsfw: 0.96 },
//   top: { label: "nsfw", score: 0.96 },
//   triggers: ["nsfw>=0.5"],
//   ms: 64,
//   backend: "wasm+simd"
// }

if (result.nsfw) {
  // block / blur / flag
}
```

For repeated use, create one detector and reuse it (one worker, one warm session):

```ts
import { NsfwDetector } from "@pixagram/nsfw-lite";

const detector = await NsfwDetector.create({
  wasmPaths: "/ort/", // where you serve ORT's own .wasm/.mjs assets
});

const a = await detector.classify(imageBitmap);
const b = await detector.classify(blob);
const c = await detector.classify("https://same-origin/your.png");

detector.dispose(); // terminates the worker / releases the session
```

`classify()` accepts `ImageData`, `ImageBitmap`, `HTMLImageElement`,
`HTMLCanvasElement`, `OffscreenCanvas`, `Blob`/`File`, or a (same-origin) URL
string. The main thread decodes the source to pixels; only pixels are sent to the
worker.

## Options

```ts
await NsfwDetector.create({
  useWorker: "auto",         // "auto" (default) | true | false
  backend: "auto",           // "auto" (WebGPU→WASM) | "webgpu" | "wasm"
  wasmPaths: "/ort/",        // dir (or per-file URL map) for ORT's wasm assets
  numThreads: 1,             // >1 needs cross-origin isolation (COOP/COEP)
  maxBatch: 8,               // images per batched inference
  batchDelayMs: 12,          // coalescing window for concurrent classify() calls
  thresholds: { nsfw: 0.5 }, // flag when P(nsfw) >= 0.5
});
```

An image is flagged `nsfw: true` when **`P(nsfw) ≥ thresholds.nsfw`**. Raise the
threshold for fewer false positives (more misses); lower it for the reverse — no
retraining needed.

Concurrent `classify()` calls within `batchDelayMs` are coalesced into a single
batched inference (up to `maxBatch`), amortizing the fixed per-call cost.

## Performance

- **~65 ms per image on CPU** (WASM + SIMD), from a **< 1 MB gzipped** embedded
  model.
- The backbone is **~0.96M params / ~65 MFLOPs** at 160px — the cheap end of the
  MobileNetV4 curve (~0.14× the compute of full `conv_small` at 256).
- Ships **uint8** for the WASM/CPU backend (ORT recommends uint8 on CPU; fp16 is
  slow there). On WebGPU it runs with CPU fallback for the quantize/dequantize
  nodes.
- For a model this small, also benchmark plain FP32-on-WASM and `numThreads`
  (with COOP/COEP) — threading is often a bigger lever than quantization.

## Building your own model

The repo ships code, not weights — produce them once. Lay your data out as a
two-class ImageFolder (`data/nsfw/`, `data/sfw/`), then:

```bash
# 1) Fine-tune at 160px (--interp nearest keeps crisp pixel blocks for pixel art)
python scripts/train_nsfw_hf_mnv4_160.py --data-dir data/ --interp nearest

# 2) Export to ONNX (self-verifies vs PyTorch) + quantize to uint8
python scripts/export_model.py --variant mnv4 --calib-data data/

# 3) Base64-embed the uint8 model + bake in labels/preprocess
npm run embed:mnv4

# 4) Build dual ESM/CJS + types
npm run build
```

Input size, normalization, and the resize filter are read from the trained
checkpoint and baked into the export, so train- and serve-time preprocessing
always match. The trainer also runs on Hugging Face Jobs (it's a self-contained
UV script) and supports optional knowledge distillation from a larger teacher —
see the script's flags for details.

**Pixel art note:** use `--interp nearest` for sprites at/below 160px (it
preserves the grid); it aliases when downscaling high-res sources, so prefer the
`bilinear` default for those. Canvas resizing isn't bit-identical to PIL on
non-integer scales — bake resize+normalize into the ONNX graph if you need
pixel-exact parity.

## Notes

- `numThreads > 1` requires a cross-origin-isolated page
  (`Cross-Origin-Opener-Policy: same-origin` +
  `Cross-Origin-Embedder-Policy: require-corp`). The default `1` always works.
- Serve ORT's `.wasm`/`.mjs` files and point `wasmPaths` at them; see the
  [onnxruntime-web docs](https://onnxruntime.ai/docs/tutorials/web/).
- Vite / webpack 5 / Rollup detect `new Worker(new URL("./worker.js",
  import.meta.url))` statically and bundle the worker for you.
- In CJS, SSR, or any environment without `Worker`, the detector falls back to
  main-thread inference — same API, same results.

## Responsible use

NSFW training data scraped from the web can contain illegal material, including
CSAM. Use a reputable, screened source, run perceptual-hash matching against
known-bad sets **before** training, and comply with your local law — you are
legally responsible for the data you hold. This classifier is a moderation aid,
not a guarantee; always pair automated flags with appropriate review.

## License

Apache-2.0. Your trained weights are your own, subject to your dataset's licensing
and your local law.
