# @pixagram/nsfw-lite

Fast, on-device **binary** (`sfw` / `nsfw`) image classification for the browser.
Runs a **MobileNetV4-conv-small-050** classifier (~2.2M params, ~0.1 GMACs —
roughly 4× less compute than an EfficientNet-b0 head) through
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) in a
**Web Worker**, with an automatic **main-thread fallback** when a worker isn't
available. The model is **bundled in the package** (base64-embedded), so there's
no separate model fetch at runtime.

Nothing leaves the device — classification happens entirely client-side.

This is a slimmed-down sibling of `@pixagram/nsfw`: one binary head instead of a
five-class gating scheme, a much lighter backbone, and trained on your own data.

## Install

```bash
npm install @pixagram/nsfw-lite onnxruntime-web
```

`onnxruntime-web` is a **peer dependency** — you install it in your app so there's
a single ORT copy and you control its version.

> **Heads up:** a freshly cloned copy of this package ships with an *empty* model
> stub and will throw `no model is embedded` at runtime until you run the
> train → export → embed pipeline below. The model is *your* model — this repo
> ships the code, not weights.

## Quick start

```ts
import { classify } from "@pixagram/nsfw-lite";

const result = await classify(myImageElement);
// {
//   nsfw: true,
//   scores: { sfw: 0.04, nsfw: 0.96 },
//   top: { label: "nsfw", score: 0.96 },
//   triggers: ["nsfw>=0.5"],
//   ms: 6,
//   backend: "webgpu"
// }

if (result.nsfw) {
  // block / blur / flag
}
```

For repeated use, create one detector and reuse it (one worker, one warm session):

```ts
import { NsfwDetector } from "@pixagram/nsfw-lite";

const detector = await NsfwDetector.create({
  // serve ORT's own wasm assets from somewhere your bundler can reach:
  wasmPaths: "/ort/",
});

console.log(detector.backend); // "worker" or "direct"

const a = await detector.classify(imageBitmap);
const b = await detector.classify(blob);
const c = await detector.classify("https://same-origin/your.jpg");

detector.dispose(); // terminates the worker / releases the session
```

`classify()` accepts `ImageData`, `ImageBitmap`, `HTMLImageElement`,
`HTMLCanvasElement`, `OffscreenCanvas`, `Blob`/`File`, or a URL string. The main
thread decodes the source to pixels; only pixels are sent to the worker.

## Options

```ts
await NsfwDetector.create({
  useWorker: "auto",        // "auto" (default) | true | false
  backend: "auto",          // "auto" (WebGPU→WASM) | "webgpu" | "wasm"
  wasmPaths: "/ort/",       // string dir, or per-file URL map for ORT's wasm
  numThreads: 1,            // >1 needs cross-origin isolation (COOP/COEP)
  maxBatch: 8,              // images per batched inference
  batchDelayMs: 12,         // coalescing window for concurrent classify() calls
  thresholds: { nsfw: 0.5 },// flag when P(nsfw) >= 0.5
  modelBytes,               // Uint8Array — override the embedded model
  preprocess,               // override embedded preprocessing params
  labels,                   // override embedded class order
});
```

An image is flagged `nsfw: true` when **`P(nsfw) ≥ thresholds.nsfw`**. Raise the
threshold to flag fewer images (fewer false positives, more misses); lower it to
flag more (more false positives, fewer misses). Pick the operating point from the
per-class precision/recall the training script prints.

## Building the package (train → export → embed → build)

The repo ships code but not weights. Produce them once:

```bash
# 0) Lay your data out as ImageFolder with exactly two classes:
#      data/nsfw/   (your 10k nsfw images)
#      data/sfw/    (your 10k sfw images)

# 1) Fine-tune MobileNetV4-small-050. Reads the backbone's native mean/std/size
#    and validates with a plain resize (matching the browser's canvas resize).
pip install "timm>=1.0.0" torch torchvision onnx onnxruntime pillow numpy
python scripts/train_nsfw.py --data-dir data/ --epochs 12
# -> model/nsfw_mnv4.pt, model/labels.json, model/preprocess.json

# 2) Export to ONNX, verify it matches PyTorch, then static-INT8-quantize it
#    (per-channel INT8 weights + UInt8 activations) using calibration images.
python scripts/export_model.py --calib-data data/
# -> model/nsfw.onnx (FP32, verified) and model/nsfw.int8.onnx (verified)

# 3) Base64-embed the INT8 model + bake in labels/preprocess.
npm run embed-model
# -> overwrites src/assets.generated.ts (EMBEDDED = true)

# 4) Build dual ESM/CJS + types.
npm run build
# -> dist/
```

To sanity-check the exported models on real images, *before* touching the
browser:

```bash
python scripts/sanity_check.py some_sfw.jpg some_nsfw.jpg
```

It prints `P(nsfw)` for both the FP32 and INT8 graphs so you can confirm they
agree and that neither collapsed.

### Training on Hugging Face

`scripts/train_nsfw_hf.py` is the same trainer, but it loads data from the
Hugging Face `datasets` library instead of a local folder, and is written as a
self-contained **UV script** so it runs directly on **HF Jobs** (GPU, no setup).
It emits the identical `nsfw_mnv4.pt` / `labels.json` / `preprocess.json`, so
steps 2–4 above are unchanged. It accepts three data layouts:

- **Two separate single-class datasets** (`--nsfw-dataset` + `--sfw-dataset`) —
  each is labelled wholesale. This matches the `civitai-top-*-images` datasets,
  which are single-class image sets with no label column.
- **One labelled dataset** (`--hf-dataset`) with a label column / class subfolders.
- **A local imagefolder** (`--data-dir`).

```bash
# Two single-class Hub datasets (the civitai layout):
python scripts/train_nsfw_hf.py \
    --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
    --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata

# On HF Jobs (Pro/Team feature; `hf auth login` first). Job storage is EPHEMERAL,
# so --push-repo uploads model/ to the Hub when training finishes:
hf jobs uv run --flavor a100-large -s HF_TOKEN scripts/train_nsfw_hf.py \
    -- --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
       --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata \
       --push-repo you/nsfw-lite-mnv4 --epochs 12 --batch-size 256 --num-workers 16

# Then pull the result down and continue the normal pipeline locally:
hf download you/nsfw-lite-mnv4 --repo-type model --local-dir model/
python scripts/export_model.py --calib-data ./calib && npm run embed-model && npm run build
```

**On picking a GPU.** Flavors go `t4-small → l4x1 → a10g-small → a10g-large →
a100-large` (80 GB), with H100/H200/B200 tiers if your plan exposes them. This
backbone is tiny (~2.2M params / 0.1 GMACs), so past an L4/A10G the GPU is idle
and epoch time is bound by **image decoding + data loading**, not compute. A
bigger card mainly lets you raise `--batch-size`; the real throughput lever is
`--num-workers` (decode parallelism). `a10g-large` is the cost-sensible choice;
`a100-large` only if you want a large batch and the headroom.

**You don't pre-create any repo.** The two input datasets already exist — just
reference them by id. The output `--push-repo` is created for you on first push
(`create_repo(exist_ok=True)`), as long as it's under your namespace and your
`HF_TOKEN` has write scope. Locally (no `--push-repo`) nothing is created; the
model just lands in `./model/`.

The classes still normalize to exactly `{nsfw, sfw}` (aliases are mapped); the JS
side reads `nsfw` by name, so index order never matters.

**Tuning knobs** (defaults are reasonable for fine-tuning on a balanced ~20k set):
`--epochs` (12 is a safe ceiling — the best-by-macro-F1 checkpoint is kept, not
the last), `--lr` (1e-3; drop to 5e-4 if the first epoch looks unstable),
`--rrc-min` (RandomResizedCrop lower scale, default 0.8 — gentle on purpose, so
cropping doesn't remove the explicit region from an `nsfw` image or drift from the
full-frame resize used at serve time), `--patience N` (early-stop after N epochs
without improvement; 0 = off), and `--ema` (track an averaged copy of the weights
and keep whichever of raw/EMA scores higher each epoch — pure upside, tune the
window with `--ema-decay`, default 0.998). If missing NSFW is costlier than
over-flagging, don't overtrain for recall — lower the serve-time `thresholds.nsfw`
below 0.5, which trades precision for recall with no retraining.

> **No-code alternative — AutoTrain.** AutoTrain trains image classifiers from a
> zip of per-class folders or a Hub dataset, but it owns the preprocessing
> (ImageNet resize + center-crop) and outputs a Transformers checkpoint, not this
> repo's timm `.pt`. That breaks the deliberate train/serve **resize parity**
> (the browser does a plain square resize, no crop) and won't feed
> `export_model.py` as-is. Prefer the script above if you want the lightweight
> MobileNetV4 + INT8-in-browser path intact.

### Label order matters

`ImageFolder` sorts class folders **alphabetically**, so `nsfw/` and `sfw/` give
the order `["nsfw", "sfw"]` — i.e. the model's output index 0 is `nsfw`. The
pipeline reads this order from the checkpoint and bakes it into the package, and
the JS decoder looks up the `nsfw` probability **by name**, so the gate stays
correct regardless of ordering. Don't hardcode an index.

## Why these choices

**MobileNetV4-small-050, not EfficientNet.** At ~2.2M params / ~0.1 GMACs it's
the lightest MobileNetV4 with timm ImageNet weights — a ~4× compute cut versus an
EfficientNet-b0 head, which is the whole point of this package. timm's
`resolve_model_data_config` hands the script the model's own mean/std/input size,
so train- and serve-time preprocessing always match.

**Binary head.** Two classes (`sfw`/`nsfw`) with a single threshold is simpler
and faster to reason about than per-class gates, and maps cleanly onto a two-
folder `ImageFolder` dataset. (If you'd rather a single sigmoid logit, retrain
with `num_classes=1` and a `BCEWithLogitsLoss` head — but the 2-class softmax
here needs no JS changes.)

**INT8, not FP8/FP16.** FP8 tensor types are rejected by onnxruntime-web's WASM
backend at session creation; FP16 gives no benefit on CPU/WASM. INT8 is the lever
that works in-browser. `export_model.py` uses **static** quantization with a
calibration set (correct for CNNs — dynamic quantization mostly helps matmul
ops). It verifies the INT8 graph still agrees with FP32 and falls back to
shipping FP32 if quantization degraded the model.

**No AvgPool patch needed.** Unlike HF EfficientNet (whose oversized fixed-kernel
global pooler traces wrong to ONNX `AveragePool`), timm MobileNetV4/V3 heads pool
cleanly to `GlobalAveragePool`. The export still sweeps several configurations and
keeps only the first whose logits match PyTorch within `1e-3`.

**Worker is truly off-thread only for ESM consumers.** The worker is spawned via
`new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which
Vite / webpack 5 / Rollup detect statically and bundle. In CJS, SSR, or any
environment without `Worker` (or where that construction throws), the detector
**automatically falls back to main-thread inference** — same API, same results.
Force one mode with `useWorker: true | false`.

**Batching for throughput.** Concurrent `classify()` calls within `batchDelayMs`
are coalesced into one `[N,3,S,S]` inference (up to `maxBatch`), so the fixed
per-call cost (GPU upload / kernel launch / readback) is amortised. The model's
batch axis is exported as dynamic.

**Bundled model = bigger tarball.** Because the model is base64-embedded and the
package builds both ESM and CJS, the model bytes appear in *both* `dist/index.js`
and `dist/index.cjs`. Your app bundle only pulls in the format it imports. The
worker does **not** embed the model; it receives the bytes via a zero-copy
transfer at init. With a ~2.2M-param INT8 model this is small (~2–3 MB), but if
tarball size matters more than self-containment, ship the `.onnx` as an external
asset and load it with `modelBytes` instead.

**Canvas resampling ≠ PIL.** Preprocessing resizes via `OffscreenCanvas`
(bilinear), not bit-identical to PIL. Predictions can drift slightly at the
margins. The training/export pipeline validates with a plain resize to mirror
this; if you need exact parity, bake resize+normalize into the ONNX graph.

## Notes

- `numThreads > 1` requires your page to be cross-origin isolated
  (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`).
  Default is `1`, which always works.
- Serve ORT's `.wasm`/`.mjs` files and point `wasmPaths` at them; see the
  [onnxruntime-web docs](https://onnxruntime.ai/docs/tutorials/web/).

## Responsible use

NSFW training data scraped from the web can contain illegal material, including
CSAM. Use a reputable, screened source, run perceptual-hash matching against
known-bad sets **before** training, and comply with your local law. You are
legally responsible for the data you hold. This classifier is a moderation aid,
not a guarantee — always pair automated flags with appropriate review.

## License

Apache-2.0. Your trained weights are your own and subject to your dataset's
licensing and your local law.
