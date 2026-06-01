# @pixagram/nsfw-lite

Fast, on-device **binary** (`sfw` / `nsfw`) image classification for the browser.
Runs a tiny **timm** classifier through
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) in a
**Web Worker**, with an automatic **main-thread fallback** when a worker isn't
available. The model is **bundled in the package** (base64-embedded), so there's
no separate model fetch at runtime.

It ships **four interchangeable backbones** — pick the size/architecture you
want and import that one:

| Import path                    | Backbone (timm)                              | Params / GMACs | Native input |
| ------------------------------ | -------------------------------------------- | -------------- | ------------ |
| `@pixagram/nsfw-lite/mnv4`     | `mobilenetv4_conv_small_050.e3000_r224_in1k` | ~2.2M / 0.1    | **224×224** (test 256) |
| `@pixagram/nsfw-lite/mnv3`     | `mobilenetv3_small_050.lamb_in1k`            | ~1.6M / 0.0    | **224×224**  |
| `@pixagram/nsfw-lite/tinynet`  | `tinynet_e.in1k`                             | ~2.0M / 0.0    | **106×106**  |
| `@pixagram/nsfw-lite/lcnet`    | `lcnet_050.ra2_in1k`                         | ~1.89M / 0.0   | **224×224** (test 256) |

Input sizes and normalization are **per-backbone** and read straight from each
timm checkpoint — `tinynet_e` runs at **106×106**, not 224, and `lcnet_050` uses
inception normalization (mean/std `0.5/0.5/0.5`) while the others use the
ImageNet defaults. You never set any of this by hand: the trainer bakes the
resolved values into `preprocess.json`, they're embedded per variant, and the
browser resizes to whatever the embedded value says. The bare specifier
`@pixagram/nsfw-lite` is an alias for `@pixagram/nsfw-lite/mnv4`.

Every variant exposes the **same API and the same result shape** — they differ
only in the embedded weights and the per-backbone preprocessing above.

Nothing leaves the device — classification happens entirely client-side.

This is a slimmed-down sibling of `@pixagram/nsfw`: one binary head instead of a
five-class gating scheme, much lighter backbones, and trained on your own data.

## Install

```bash
npm install @pixagram/nsfw-lite onnxruntime-web
```

`onnxruntime-web` is a **peer dependency** — you install it in your app so there's
a single ORT copy and you control its version.

> **Heads up:** a freshly cloned copy of this package ships with *empty* model
> stubs and every variant will throw `no model is embedded` at runtime until you
> run the train → export → embed pipeline below for that variant. The model is
> *your* model — this repo ships the code, not weights.

## Quick start

Import the variant you trained (here, the default MobileNetV4):

```ts
import { classify } from "@pixagram/nsfw-lite/mnv4";
// equivalently: import { classify } from "@pixagram/nsfw-lite";

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

Swapping backbones is a one-line import change — `…/mnv3`, `…/tinynet`, or
`…/lcnet` — with no other code changes:

```ts
import { classify } from "@pixagram/nsfw-lite/lcnet";
```

For repeated use, create one detector and reuse it (one worker, one warm session):

```ts
import { NsfwDetector } from "@pixagram/nsfw-lite/mnv4";

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

> **Don't mix variants in one app unless you mean to.** Each import pulls in its
> own embedded model, so importing two variants ships two models. The one-shot
> `classify` / `warmup` / `disposeShared` helpers are *per variant* (each module
> has its own shared singleton); if you need more than one at once, create your
> own `NsfwDetector` from each and manage them yourself.

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

The repo ships code but not weights. Produce them once **per variant** you want.
Everything is keyed on a short variant id — `mnv4`, `mnv3`, `tinynet`, `lcnet` —
and each variant keeps its own folder under `model/<variant>/` so the four never
clobber each other's `labels.json` / `preprocess.json` / checkpoints.

```bash
# 0) Lay your data out as ImageFolder with exactly two classes:
#      data/nsfw/   (your nsfw images)
#      data/sfw/    (your sfw images)

# 1) Fine-tune. There's one HF trainer per variant; each defaults --out-dir to
#    model/<variant> and saves nsfw_<variant>.pt. Each reads its backbone's
#    native mean/std/size and validates with a plain resize (matching the
#    browser's canvas resize), so train- and serve-time preprocessing match.
pip install "timm>=1.0.0" torch torchvision "datasets>=2.14.0" pillow numpy huggingface_hub
python scripts/train_nsfw_hf_lcnet.py --data-dir data/ --epochs 12
# -> model/lcnet/nsfw_lcnet.pt, model/lcnet/labels.json, model/lcnet/preprocess.json

# 2) Export to ONNX, verify it matches PyTorch, then static-INT8-quantize it
#    (per-channel INT8 weights + UInt8 activations) using calibration images.
#    export_model.py recreates the right backbone from each checkpoint, so the
#    same script handles every variant. --variant takes one, several, or `all`;
#    each derives its own checkpoint (model/<v>/nsfw_<v>.pt) and out-dir
#    (model/<v>). With exactly one target you may override via --checkpoint/--out-dir.
python scripts/export_model.py --variant all --calib-data data/   # or: mnv3 lcnet, or a single name
# -> model/<v>/nsfw.onnx (FP32, verified) + model/<v>/nsfw.int8.onnx (verified), per variant

# 3) Base64-embed the INT8 model + bake in labels/preprocess for this variant.
npm run embed:lcnet          # or `npm run embed-model` to (re)embed all four
# -> overwrites src/variants/lcnet/assets.generated.ts (EMBEDDED = true)

# 4) Build all variants (dual ESM/CJS + types).
npm run build
# -> dist/mnv4/  dist/mnv3/  dist/tinynet/  dist/lcnet/
```

The four trainer scripts are:

```
scripts/train_nsfw_hf_mnv4.py        scripts/train_nsfw_hf_tinynet.py
scripts/train_nsfw_hf_mnv3.py        scripts/train_nsfw_hf_lcnet.py
```

They are byte-for-byte the same trainer apart from the backbone id, the
`--out-dir model/<variant>` default, and the saved `nsfw_<variant>.pt` name. The
trainer **prints the resolved input size / mean / std** for its backbone at
startup — don't hardcode those anywhere; the export + embed steps carry the real
values through to the browser.

`npm run embed-model` embeds **every** variant that has a
`model/<variant>/nsfw.int8.onnx` and writes an empty stub for any that don't, so
a partial build (say, only `lcnet` trained) still compiles — the un-built
variants just throw `no model is embedded` if you actually import them.

To sanity-check an exported variant on real images, *before* touching the
browser (point `NSFW_MODEL_DIR` at the variant folder):

```bash
NSFW_MODEL_DIR=model/lcnet python scripts/sanity_check.py some_sfw.jpg some_nsfw.jpg
```

It prints `P(nsfw)` for both the FP32 and INT8 graphs so you can confirm they
agree and that neither collapsed.

### Training on Hugging Face

Each `scripts/train_nsfw_hf_<variant>.py` is the same trainer, but it loads data
from the Hugging Face `datasets` library instead of a local folder, and is
written as a self-contained **UV script** so it runs directly on **HF Jobs**
(GPU, no setup). Each emits `nsfw_<variant>.pt` / `labels.json` /
`preprocess.json` into `model/<variant>/`, so steps 2–4 above are unchanged. They
accept three data layouts:

- **Two separate single-class datasets** (`--nsfw-dataset` + `--sfw-dataset`) —
  each is labelled wholesale. This matches the `civitai-top-*-images` datasets,
  which are single-class image sets with no label column.
- **One labelled dataset** (`--hf-dataset`) with a label column / class subfolders.
- **A local imagefolder** (`--data-dir`).

```bash
# Two single-class Hub datasets (the civitai layout), training the lcnet variant:
python scripts/train_nsfw_hf_lcnet.py \
    --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
    --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata

# On HF Jobs (Pro/Team feature; `hf auth login` first). Job storage is EPHEMERAL,
# so --push-repo uploads model/<variant>/ to the Hub when training finishes. The
# uploaded folder contains nsfw_<variant>.pt under the variant-specific name:
hf jobs uv run --flavor a100-large -s HF_TOKEN scripts/train_nsfw_hf_lcnet.py \
    -- --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
       --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata \
       --push-repo you/nsfw-lite-lcnet --epochs 12 --batch-size 256 --num-workers 16

# Then pull that variant's folder down and continue the normal pipeline locally:
hf download you/nsfw-lite-lcnet --repo-type model --local-dir model/lcnet/
python scripts/export_model.py --variant lcnet --calib-data ./calib
npm run embed:lcnet && npm run build
```

Repeat for `mnv4` / `mnv3` / `tinynet` by swapping the script name, the
`--push-repo you/nsfw-lite-<variant>` id, and the `model/<variant>` paths.

**On picking a GPU.** Flavors go `t4-small → l4x1 → a10g-small → a10g-large →
a100-large` (80 GB), with H100/H200/B200 tiers if your plan exposes them. These
backbones are tiny, so past an L4/A10G the GPU is idle and epoch time is bound by
**image decoding + data loading**, not compute. A bigger card mainly lets you
raise `--batch-size`; the real throughput lever is `--num-workers` (decode
parallelism). `a10g-large` is the cost-sensible choice; `a100-large` only if you
want a large batch and the headroom.

**You don't pre-create any repo.** The two input datasets already exist — just
reference them by id. The output `--push-repo` is created for you on first push
(`create_repo(exist_ok=True)`), as long as it's under your namespace and your
`HF_TOKEN` has write scope. Locally (no `--push-repo`) nothing is created; the
model just lands in `./model/<variant>/`.

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

**Local (no-Hub) training.** `scripts/train_nsfw.py` is the torchvision-only
trainer; pass `--backbone <timm-id> --out-dir model/<variant> --data-dir data/`
to target any of the four locally. (It names its checkpoint `nsfw_mnv4.pt`
regardless of backbone, but that's cosmetic — `export_model.py` reads the
backbone from inside the checkpoint, and the embed step only consumes
`model/<variant>/nsfw.int8.onnx` + `labels.json` + `preprocess.json`.)

> **No-code alternative — AutoTrain.** AutoTrain trains image classifiers from a
> zip of per-class folders or a Hub dataset, but it owns the preprocessing
> (ImageNet resize + center-crop) and outputs a Transformers checkpoint, not this
> repo's timm `.pt`. That breaks the deliberate train/serve **resize parity**
> (the browser does a plain square resize, no crop) and won't feed
> `export_model.py` as-is. Prefer the scripts above to keep the lightweight
> timm + INT8-in-browser path intact.

### Label order matters

`ImageFolder` sorts class folders **alphabetically**, so `nsfw/` and `sfw/` give
the order `["nsfw", "sfw"]` — i.e. the model's output index 0 is `nsfw`. The
pipeline reads this order from the checkpoint and bakes it into the package, and
the JS decoder looks up the `nsfw` probability **by name**, so the gate stays
correct regardless of ordering. Don't hardcode an index.

## What ends up in `dist/`

`npm run build` (tsup) emits one self-contained bundle per variant:

```
dist/
  mnv4/   index.js  index.cjs  index.d.ts  worker.js  worker.cjs  …maps
  mnv3/   index.js  index.cjs  index.d.ts  worker.js  worker.cjs  …maps
  tinynet/ …
  lcnet/  …
```

- Each `dist/<variant>/index.*` **embeds only its own model** — there's no
  cross-variant bleed, so importing `…/lcnet` never pulls in the mnv4 weights.
- Each variant gets **its own `worker.js` sibling**, referenced via
  `new Worker(new URL("./worker.js", import.meta.url))`. The worker file does
  **not** embed the model; it receives the bytes via a zero-copy transfer at
  init. Vite / webpack 5 / Rollup detect that construction statically and bundle
  the right per-variant worker.
- The shared TypeScript types live in one small `.d.ts` chunk that every
  variant's `index.d.ts` re-exports; all of it ships under `dist/`.

## Why these choices

**Four extreme-efficiency timm backbones.** All four are among the lightest
ImageNet-pretrained CNNs timm offers, differing mainly by architecture family and
native input resolution: **MobileNetV4-small-050** (~2.2M params / ~0.1 GMACs at
224×224 — roughly a 4× compute cut versus an EfficientNet-b0 head, which is the
whole point of this package), **MobileNetV3-small-050** (~1.6M, 224×224),
**TinyNet-E** (~2.0M, an EfficientNet-family model compound-scaled *down* to a
**106×106** input), and **PP-LCNet-050** (~1.89M, 224×224, designed specifically
for low CPU latency). They're offered as separate bundles so you can A/B them on
your own data and ship whichever wins without touching call sites. `timm`'s
`resolve_model_data_config` hands each trainer the model's own mean/std/input
size, so train- and serve-time preprocessing always match — including the
different native resolutions (notably TinyNet-E's 106×106, not 224) and the
fact that PP-LCNet uses inception normalization rather than the ImageNet default.

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
shipping FP32 if quantization degraded the model — and this verification runs for
**every** backbone, including TinyNet.

**Why not INT4/UINT4.** 4-bit in ONNX Runtime is a *weight-only* scheme that only
touches **MatMul** weights (→ `MatMulNBits`, blockwise) — plus Gather/MoE. There
is **no 4-bit path for Conv**, and onnxruntime-web's 4-bit kernels are the
`MatMulNBits` family (built for transformer inference), not Conv. These backbones
are ~entirely Conv with a single tiny classifier head, so 4-bit would quantize
**nothing** (the head usually exports as `Gemm`, not `MatMul`) and very likely
fail to load in the browser. INT8 stays the right operating point. If you want to
confirm this yourself, `export_model.py --quant int4` runs the real 4-bit
quantizer and prints how many nodes it actually touched (expect 0 on a CNN) plus
a load check; it needs `pip install onnx_ir` on recent onnxruntime, and the
result is **not** expected to run under onnxruntime-web.

**No AvgPool patch needed.** Unlike HF Transformers' EfficientNet (whose oversized
fixed-kernel global pooler traces wrong to ONNX `AveragePool`), timm's heads —
including TinyNet's EfficientNet-family one — use `SelectAdaptivePool2d`, which
traces cleanly to `GlobalAveragePool`. The export still sweeps several
configurations and keeps only the first whose logits match PyTorch within `1e-3`,
so a bad trace can never ship silently regardless of backbone.

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

**Bundled model = bigger tarball.** Because each variant's model is base64-embedded
and the package builds both ESM and CJS, those model bytes appear in *both*
`dist/<variant>/index.js` and `dist/<variant>/index.cjs`. Your app bundle only
pulls in the variant and format it imports. The worker does **not** embed the
model; it receives the bytes via a zero-copy transfer at init. With these tiny
INT8 models each bundle is small, but if tarball size matters more than
self-containment, ship the `.onnx` as an external asset and load it with
`modelBytes` instead.

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
