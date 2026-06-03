# @pixagram/nsfw-lite

Fast, on-device **binary** (`sfw` / `nsfw`) image classification for the browser,
tuned for **pixel art**. Runs a tiny **MobileNetV4-conv-small-050** (timm)
classifier through
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) in a
**Web Worker**, with an automatic **main-thread fallback** when a worker isn't
available. The **uint8-quantized** model is **bundled in the package**
(base64-embedded), so there's no separate model fetch at runtime.

| Backbone (timm)                              | Input | Normalization        | Quant |
| -------------------------------------------- | ----- | -------------------- | ----- |
| `mobilenetv4_conv_small_050.e3000_r224_in1k` | **160×160** | inception (0.5/0.5/0.5) | uint8 (U8U8) |

Input size, normalization, and the **resize filter** are read straight from the
trained checkpoint — you never set them by hand. The trainer bakes the resolved
values into `preprocess.json`, the export embeds them, and the browser resizes to
whatever the embedded values say, so train- and serve-time preprocessing always
match.

Nothing leaves the device — classification happens entirely client-side.

This is a slimmed-down sibling of `@pixagram/nsfw`: one binary head instead of a
five-class gating scheme, a much lighter backbone, and trained on your own data.

## Install

```bash
npm install @pixagram/nsfw-lite onnxruntime-web
```

`onnxruntime-web` is a **peer dependency** — you install it in your app so there's
a single ORT copy and you control its version.

> **Heads up:** a freshly cloned copy ships with an *empty* model stub and will
> throw `no model is embedded` at runtime until you run the train → export →
> embed pipeline below. The model is *your* model — this repo ships the code, not
> weights.

## Quick start

```ts
import { classify } from "@pixagram/nsfw-lite";
// equivalently: import { classify } from "@pixagram/nsfw-lite/mnv4";

const result = await classify(myImageElement);
// {
//   nsfw: true,
//   scores: { sfw: 0.04, nsfw: 0.96 },
//   top: { label: "nsfw", score: 0.96 },
//   triggers: ["nsfw>=0.5"],
//   ms: 6,
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
  // serve ORT's own wasm assets from somewhere your bundler can reach:
  wasmPaths: "/ort/",
});

console.log(detector.backend); // "worker" or "direct"

const a = await detector.classify(imageBitmap);
const b = await detector.classify(blob);
const c = await detector.classify("https://same-origin/your.png");

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
flag more. Pick the operating point from the per-class precision/recall the
training script prints.

## Building the package (train → export → embed → build)

The repo ships code but not weights. Produce them once. Everything lands under
`model/mnv4/`.

```bash
# 0) Lay your data out as ImageFolder with exactly two classes:
#      data/nsfw/   (your nsfw images)
#      data/sfw/    (your sfw images)

# 1) Fine-tune mobilenetv4_conv_small_050 at 160px. --interp nearest gives the
#    pixelated (crisp-block) resize for pixel art; it's recorded in the checkpoint
#    so export + serve inherit it. Saves model/mnv4/nsfw_mnv4.pt (+ labels/preprocess).
pip install "timm>=1.0.0" torch torchvision "datasets>=2.14.0" pillow numpy huggingface_hub
python scripts/train_nsfw_hf_mnv4_160.py --data-dir data/ --interp nearest

# 2) Export to ONNX, verify it matches PyTorch, then quantize. Emits THREE files:
#      model/mnv4/nsfw.onnx        FP32 reference (verified, self-contained)
#      model/mnv4/nsfw.uint8.onnx  U8U8 static-quant for the WASM EP  <-- embedded
#      model/mnv4/nsfw.fp16.onnx   float16 for the WebGPU EP
#    Calibration must be representative — for pixel art, point it at pixel art.
python scripts/export_model.py --variant mnv4 --calib-data data/

# 3) Base64-embed the uint8 model + bake in labels/preprocess.
npm run embed:mnv4           # or `npm run embed-model`
# -> overwrites src/variants/mnv4/assets.generated.ts (EMBEDDED = true)

# 4) Build (dual ESM/CJS + types).
npm run build
# -> dist/mnv4/index.{js,cjs,d.ts}  dist/mnv4/worker.{js,d.ts}
```

The export self-verifies: it sweeps export configs and keeps only the first whose
logits match PyTorch within `1e-3`, and after quantization it checks the uint8
graph still agrees with FP32 on a probe — falling back to shipping FP32 as the
WASM file if quantization collapsed. So a bad trace or a degenerate quantization
can't ship silently.

### Pixel art: the resize filter

The resize filter is the one preprocessing knob that matters more for pixel art
than for photos, and it must be identical at train and serve time. It's a single
source of truth: `--interp` on the trainer flows into the checkpoint, then to the
export's calibration resize, then into `preprocess.json`, and the browser honors
it (`"nearest"` ⇒ canvas smoothing off ⇒ crisp pixel blocks; `"bilinear"` ⇒
smooth, the default). Use `--interp nearest` when your sources are sprites at or
below 160px (it preserves the grid); note that nearest **aliases when
downscaling** large images, so weigh it if you feed high-res sources. Canvas
nearest/bilinear aren't bit-identical to PIL's on non-integer scale factors — a
negligible drift for a 2-class head, but bake preprocessing into the ONNX graph
if you need pixel-exact parity.

### Training on Hugging Face

`scripts/train_nsfw_hf_mnv4_160.py` is the same trainer as the local one but
loads data from the Hugging Face `datasets` library, and is written as a
self-contained **UV script** so it runs directly on **HF Jobs** (GPU, no setup).
It accepts three data layouts:

- **Two separate single-class datasets** (`--nsfw-dataset` + `--sfw-dataset`) —
  each labelled wholesale. Matches the `civitai-top-*-images` sets (single-class,
  no label column).
- **One labelled dataset** (`--hf-dataset`) with a label column / class subfolders.
- **A local imagefolder** (`--data-dir`).

```bash
# On HF Jobs (Pro/Team feature; `hf auth login` first). Job storage is EPHEMERAL,
# so --push-repo uploads model/mnv4/ to the Hub when training finishes.
hf jobs uv run --flavor a100-large --timeout 2h -s HF_TOKEN scripts/train_nsfw_hf_mnv4_160.py \
    -- --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
       --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata \
       --push-repo primerz/nsfw_mnv4_160 --interp nearest \
       --batch-size 160 --num-workers 16 --patience 5

# Then pull the model down and continue locally:
hf download primerz/nsfw_mnv4_160 --repo-type model --local-dir model/mnv4/
python scripts/export_model.py --variant mnv4 --calib-data ./calib
npm run embed:mnv4 && npm run build
```

**On picking a GPU.** Flavors go `t4-small → l4x1 → a10g-small → a10g-large →
a100-large` (80 GB), with H100/H200/B200 tiers if your plan exposes them. This
backbone is ~0.96M params, so past an L4/A10G the GPU is idle and epoch time is
bound by **image decoding + data loading**, not compute. A bigger card mainly
lets you raise `--batch-size`; the real throughput lever is `--num-workers`
(decode parallelism). `a10g-large` is the cost-sensible choice.

**You don't pre-create the repo.** The two input datasets already exist — just
reference them. The output `--push-repo` is created for you on first push
(`create_repo(exist_ok=True)`) under your namespace with a write-scoped
`HF_TOKEN`. Locally (no `--push-repo`) the model just lands in `./model/mnv4/`.

**Optional knowledge distillation.** To squeeze more accuracy into the small
`_050` student, train a full `mobilenetv4_conv_small` teacher and pass
`--teacher-checkpoint model/mnv4_full/nsfw_mnv4.pt`; the student learns the
teacher's softened logits on top of the labels. Off by default.

**Tuning knobs** (defaults are reasonable for a balanced ~20k set): `--epochs`
(12 ceiling — best-by-macro-F1 is kept, not the last), `--lr` (1e-3; drop to 5e-4
if the first epoch is unstable), `--rrc-min` (RandomResizedCrop lower scale,
default 0.8), `--patience N` (early-stop after N stale epochs; 0 = off),
`--img-size` (default 160), `--interp` (default bilinear; use nearest for pixel
art), `--ema` (track an averaged copy, keep whichever of raw/EMA scores higher).
If missing NSFW is costlier than over-flagging, lower the serve-time
`thresholds.nsfw` below 0.5 — trades precision for recall with no retraining.

### Label order

`ImageFolder` sorts class folders **alphabetically**, so `nsfw/` and `sfw/` give
order `["nsfw", "sfw"]` (output index 0 = `nsfw`). The pipeline reads this from
the checkpoint and bakes it into the package, and the JS decoder looks up the
`nsfw` probability **by name**, so the gate stays correct regardless of ordering.
Don't hardcode an index.

## What ends up in `dist/`

```
dist/mnv4/  index.js  index.cjs  index.d.ts  worker.js  worker.cjs  …maps
```

- `dist/mnv4/index.*` **embeds the model**; the worker file does **not** — it
  receives the bytes via a zero-copy transfer at init. Vite / webpack 5 / Rollup
  detect `new Worker(new URL("./worker.js", import.meta.url))` statically and
  bundle the worker.
- The bare specifier `@pixagram/nsfw-lite` and `@pixagram/nsfw-lite/mnv4` resolve
  to the same bundle.

## Why these choices

**MobileNetV4-conv-small-050.** Among the lightest ImageNet-pretrained CNNs timm
offers (~0.96M params), and architecturally stronger per-parameter than the older
small CNN families. At 160px it's the cheap end of the curve — ~0.14× the compute
of the full `conv_small` at 256 — which is the whole point of this package.

**uint8 for WASM, fp16 for WebGPU.** ONNX Runtime Web recommends **uint8**
quantized models on the WASM/CPU backend and warns that **float16 is slow on
CPU** (not natively supported). So the export produces a **U8U8** model for the
WASM EP and a **fp16** model for the WebGPU EP. This single-file bundle embeds the
**uint8** model, which runs on WASM and also on WebGPU with per-op CPU fallback
for the quantize/dequantize nodes. For a *pure* WebGPU path (no QDQ fallback) the
bundle would embed + select both models per backend — a contained loader change;
ask if you want it. For a model this small, also benchmark plain FP32-on-WASM and
`numThreads` (with COOP/COEP) — at ~65 MFLOPs the quantization overhead doesn't
always pay off, and threading is often the bigger lever.

**Binary head.** Two classes (`sfw`/`nsfw`) with a single threshold is simpler and
faster to reason about than per-class gates, and maps cleanly onto a two-folder
`ImageFolder`. (Prefer a single sigmoid? Retrain with `num_classes=1` +
`BCEWithLogitsLoss`; the 2-class softmax here needs no JS changes.)

**No AvgPool patch needed.** timm's `SelectAdaptivePool2d` head traces cleanly to
ONNX `GlobalAveragePool` (unlike HF EfficientNet's oversized fixed-kernel pooler).

**Worker is off-thread only for ESM consumers.** Spawned via `new Worker(new
URL("./worker.js", import.meta.url), { type: "module" })`. In CJS, SSR, or any
environment without `Worker`, the detector **falls back to main-thread**
inference — same API, same results. Force a mode with `useWorker: true | false`.

**Batching for throughput.** Concurrent `classify()` calls within `batchDelayMs`
are coalesced into one `[N,3,S,S]` inference (up to `maxBatch`), amortising the
fixed per-call cost. The model's batch axis is exported as dynamic.

**Bundled model = bigger tarball.** The model is base64-embedded and the package
builds both ESM and CJS, so the bytes appear in both `dist/mnv4/index.js` and
`index.cjs`. Your app bundle pulls in only the format it imports. If tarball size
matters more than self-containment, ship the `.onnx` as an external asset and
load it with `modelBytes`.

**Canvas resize honors `interpolation`.** Preprocessing resizes via
`OffscreenCanvas`, with smoothing on (bilinear) or off (nearest) per the embedded
`interpolation` value, to match the trained filter. It's not bit-identical to
PIL on non-integer scales; for exact parity, bake resize+normalize into the ONNX
graph.

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
