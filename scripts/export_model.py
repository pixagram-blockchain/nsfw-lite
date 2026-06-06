#!/usr/bin/env python
"""
export_model.py — export a trained timm binary NSFW model (from train_nsfw.py or
the per-variant train_nsfw_hf_<variant>.py trainers) to ONNX in TWO deployment
formats, each VERIFIED against PyTorch, for onnxruntime-web:

  • nsfw.uint8.onnx — static-quantized 8-bit, for the WASM (CPU) execution
    provider. Defaults to U8U8 (uint8 activations + uint8 weights), which is the
    format ONNX Runtime Web's own guidance recommends for the WASM backend; pass
    --wasm-quant u8s8 for per-channel int8 weights + reduce_range (the non-VNNI-
    safe accuracy-leaning alternative).
  • nsfw.fp16.onnx — float16, for the WebGPU execution provider. Converted with
    keep_io_types=True, so the graph's inputs/outputs stay float32 — the JS side
    keeps feeding a normal Float32Array and reading float32 logits, no float16
    plumbing required. (float16 is intentionally NOT used for the CPU/WASM build:
    ORT runs fp16 slowly on CPU.)
  • nsfw.onnx — the float32 reference (self-contained), kept for verification and
    as the source both other formats are derived from.

WHY TWO ARTIFACTS: pick the format per backend at load time, e.g.
    const ep = navigator.gpu ? 'webgpu' : 'wasm';
    const file = ep === 'webgpu' ? 'nsfw.fp16.onnx' : 'nsfw.uint8.onnx';
    session = await ort.InferenceSession.create(file, { executionProviders: [ep] });
NOTE: this changes the emitted filenames (was nsfw.int8.onnx). Update your
embed/build step to copy + reference nsfw.uint8.onnx and nsfw.fp16.onnx.

Backbone-agnostic by design: the checkpoint carries its own backbone name +
labels + preprocessing, so the SAME script exports any timm backbone it was
trained with (e.g. mobilenetv4_conv_small_050, _035, or the full conv_small).
It re-emits labels.json / preprocess.json too.

A LogitsOnly wrapper, an export-strategy sweep that keeps the FIRST config
matching PyTorch within 1e-3, then static quantization with calibration. External
weight data (torch>=2.x may split weights into a <name>.onnx.data sidecar) is
inlined so the shipped .onnx is always self-contained.

INSTALL:
    pip install "timm>=1.0.0" torch onnx onnxruntime onnxconverter-common pillow numpy

RUN:
    # Multi-variant layout. --variant derives the checkpoint AND the out-dir:
    #   --variant <v>  ->  --checkpoint model/<v>/nsfw_<v>.pt   --out-dir model/<v>
    python scripts/export_model.py --variant mnv4 --calib-data data/
    python scripts/export_model.py --variant mnv4 --calib-data data/ --wasm-quant u8s8
    # An explicit --checkpoint or --out-dir always overrides the derived path.

    # Legacy single-model layout (model/nsfw_mnv4.pt -> model/); omit --variant:
    python scripts/export_model.py --calib-data data/
Then:
    npm run embed:<variant> && npm run build      # or: npm run embed-model && npm run build

CALIBRATION & PIXEL ART: static-quant scales are only as good as the calibration
distribution, so --calib-data must be representative of what you serve. For a
pixel-art classifier, point it at actual pixel art (limited palettes / hard edges
produce different activation ranges than natural photos). The calibration resize
mirrors the training/serve transform — same filter (interp) and the same
letterbox geometry — so train, calibrate, and serve stay identical.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import onnx
import torch
import torch.nn as nn
import timm
import onnxruntime as ort
from PIL import Image

# The package ships a single variant, mnv4. --variant only uses this to derive
# the default checkpoint/out-dir paths; the actual architecture is always read
# from the checkpoint, so training a different backbone needs no change here.
VARIANTS = ("mnv4",)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None, choices=list(VARIANTS),
                   help="convenience for the multi-variant layout: derive "
                        "--checkpoint model/<v>/nsfw_<v>.pt and --out-dir model/<v>. "
                        "Explicit --checkpoint/--out-dir override it. Omit for the "
                        "legacy single-model layout (model/nsfw_mnv4.pt -> model/).")
    p.add_argument("--checkpoint", default=None,
                   help="trained .pt to export (default: model/<variant>/nsfw_<variant>.pt "
                        "when --variant is given, else model/nsfw_mnv4.pt)")
    p.add_argument("--out-dir", default=None,
                   help="output dir for nsfw.onnx / nsfw.uint8.onnx / nsfw.fp16.onnx / "
                        "labels.json / preprocess.json (default: model/<variant> with "
                        "--variant, else model)")
    p.add_argument("--calib-data", default=None,
                   help="dir of images (ImageFolder ok) to sample quantization calibration from")
    p.add_argument("--calib-count", type=int, default=400)
    p.add_argument("--wasm-quant", default="u8u8", choices=["u8u8", "u8s8"],
                   help="8-bit scheme for the WASM artifact (default u8u8 = uint8 activations + "
                        "uint8 weights, per-tensor; the broadly-supported WASM-friendly format "
                        "ORT Web recommends). u8s8 = uint8 activations + per-channel int8 weights "
                        "with reduce_range (better accuracy, non-VNNI-safe). Benchmark both — "
                        "for a model this small, plain fp32-on-WASM is also worth comparing.")
    p.add_argument("--skip-fp16", action="store_true",
                   help="don't emit nsfw.fp16.onnx (e.g. if you never serve the WebGPU EP).")
    return p.parse_args()


def softmax(z):
    z = np.asarray(z).reshape(-1)
    e = np.exp(z - z.max())
    return e / e.sum()


def letterbox(img, size, resample, pad_color=(0, 0, 0)):
    """Map an arbitrary-aspect PIL image to a `size`x`size` RGB image by
    LETTERBOXING — preserve the aspect ratio, fit the WHOLE image inside, and pad
    the margins with pad_color. Same geometry as the browser's resizeToSquare
    (src/core.ts), so calibration and serve agree. It's the only fit: cropping
    drops edge content and squashing distorts wide images, both of which cause
    missed detections. Dims use round-half-up and offsets use floor to mirror JS
    Math.round / Math.floor (canvas vs PIL still won't be bit-identical on
    non-integer scales; bake resize into the ONNX graph if you need that)."""
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), tuple(pad_color))
    scale = min(size / w, size / h)
    dw = max(1, int(w * scale + 0.5))
    dh = max(1, int(h * scale + 0.5))
    resized = img.resize((dw, dh), resample)
    ox = (size - dw) // 2  # >=0; matches Math.floor
    oy = (size - dh) // 2
    canvas = Image.new("RGB", (size, size), tuple(pad_color))
    canvas.paste(resized, (ox, oy))
    return canvas


def _rm(path):
    """Remove a model file and any external-data sidecar it may have produced."""
    for f in (path, path + ".data", os.path.splitext(path)[0] + ".data"):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


def inline_save(model_or_path, dst):
    """Save an ONNX model to `dst` fully self-contained (weights inlined). Accepts
    a path (loaded, resolving any sidecar) or an in-memory ModelProto. torch>=2.x
    may externalize weights into <name>.onnx.data; inlining guarantees the shipped
    .onnx never references a missing file."""
    m = onnx.load(model_or_path) if isinstance(model_or_path, str) else model_or_path
    onnx.save_model(m, dst, save_as_external_data=False)


def main():
    args = parse_args()

    # Resolve checkpoint / out-dir. --variant is purely a convenience that derives
    # both for the multi-variant layout (model/<v>/nsfw_<v>.pt -> model/<v>); an
    # explicit --checkpoint or --out-dir always wins. Without --variant we keep the
    # original single-model defaults so legacy train_nsfw.py output exports unchanged.
    if args.variant:
        checkpoint = args.checkpoint or os.path.join("model", args.variant, f"nsfw_{args.variant}.pt")
        out_dir = args.out_dir or os.path.join("model", args.variant)
        print(f"[export] variant={args.variant}  checkpoint={checkpoint}  out-dir={out_dir}")
    else:
        checkpoint = args.checkpoint or os.path.join("model", "nsfw_mnv4.pt")
        out_dir = args.out_dir or "model"

    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = ckpt["backbone"]
    labels = ckpt["labels"]
    size = int(ckpt["size"])
    mean = [float(m) for m in ckpt["mean"]]
    std = [float(s) for s in ckpt["std"]]
    # Resize filter the model was trained with (older checkpoints predate this and
    # default to bilinear). Drives BOTH calibration and the preprocess.json the
    # browser reads, so all three stages stay on the same filter.
    interp = str(ckpt.get("interp", "bilinear"))
    _RES = getattr(Image, "Resampling", Image)  # Pillow >=9.1 moved the enum
    pil_resample = {"bilinear": _RES.BILINEAR, "nearest": _RES.NEAREST}.get(interp, _RES.BILINEAR)
    # The fit is always LETTERBOX (preserve aspect ratio, fit inside, pad). The
    # only knob is the pad colour, threaded from the checkpoint so calibration,
    # preprocess.json, and the browser share it (default black).
    pad_color = tuple(int(c) for c in (ckpt.get("pad_color") or (0, 0, 0)))[:3]
    num_classes = len(labels)
    print(f"[export] backbone={backbone}  classes={labels}  size={size}  interp={interp}  pad_color={pad_color}")
    if num_classes != 2:
        print(f"[export] WARNING: expected 2 classes for the binary model, got {num_classes}")

    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class LogitsOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values)

    wrapped = LogitsOnly(model).eval()
    for mod in wrapped.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.eval()

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        pt_dummy = wrapped(dummy).numpy().reshape(1, -1)

    def onnx_logits(path, x):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return np.asarray(sess.run(None, {name: x.numpy().astype(np.float32)})[0]).reshape(1, -1)

    def exp_legacy(path, opset, fold):
        kw = dict(opset_version=opset, input_names=["pixel_values"], output_names=["logits"],
                  do_constant_folding=fold, training=torch.onnx.TrainingMode.EVAL,
                  dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}})
        with torch.no_grad():
            try:
                torch.onnx.export(wrapped, (dummy,), path, dynamo=False, **kw)
            except TypeError:
                torch.onnx.export(wrapped, (dummy,), path, **kw)

    def exp_dynamo(path):
        with torch.no_grad():
            torch.onnx.export(wrapped, (dummy,), path, dynamo=True)

    strategies = [
        ("legacy, fold=OFF, opset17", lambda p: exp_legacy(p, 17, False)),
        ("legacy, fold=OFF, opset14", lambda p: exp_legacy(p, 14, False)),
        ("legacy, fold=ON,  opset17", lambda p: exp_legacy(p, 17, True)),
        ("legacy, fold=OFF, opset20", lambda p: exp_legacy(p, 20, False)),
        ("dynamo, native opset     ", lambda p: exp_dynamo(p)),
    ]

    fp32_path = os.path.join(out_dir, "nsfw.onnx")
    tmp = os.path.join(out_dir, "_try.onnx")
    chosen = None
    print("\n[export] searching for a faithful export configuration:")
    for name, fn in strategies:
        try:
            _rm(tmp)
            fn(tmp)
            diff = float(np.abs(pt_dummy - onnx_logits(tmp, dummy)).max())
            ok = diff < 1e-3
            print(f"  - {name:28s} max|PT-ONNX|={diff:.6f}  {'PASS' if ok else 'reject'}")
            if ok:
                # Inline as we select: torch>=2.x may have written tmp.onnx.data, so
                # a bare move would orphan the sidecar. onnx.load resolves it; we
                # re-save self-contained straight to the final path.
                inline_save(tmp, fp32_path)
                chosen = name
                break
        except Exception as e:  # noqa: BLE001
            print(f"  - {name:28s} export error: {str(e)[:80]}")
    _rm(tmp)
    if not chosen:
        print("\n[export] FAILED: no configuration matched PyTorch. Do NOT ship.")
        raise SystemExit(2)
    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"[export] FP32 export OK via: {chosen}  ({fp32_mb:.2f} MB, self-contained)")

    # Re-emit labels + preprocess so the package is reproducible from the checkpoint.
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    # The model input is always letterboxed; padColor is the only geometry knob
    # core.ts reads. The legacy cropSize/doCenterCrop fields stay inert.
    preprocess = {
        "size": size, "cropSize": None, "doCenterCrop": False,
        "rescaleFactor": 1.0 / 255.0, "rescaleOffset": False,
        "doNormalize": True, "mean": mean, "std": std, "includeTop": False,
        "interpolation": interp,
        "padColor": list(pad_color),
    }
    with open(os.path.join(out_dir, "preprocess.json"), "w") as f:
        json.dump(preprocess, f, indent=2)

    # ── Shared: shape-inferred source for quantization ───────────────────
    from onnxruntime.quantization import (  # noqa: E402
        quantize_static, CalibrationDataReader, QuantType, QuantFormat,
    )

    def prep_for_quant(src):
        """quantize_static wants shape info. Try ORT's pre-process, then a plain
        onnx shape-inference, then fall back to the raw graph."""
        out = os.path.join(out_dir, "nsfw.prep.onnx")
        try:
            from onnxruntime.quantization import quant_pre_process
            quant_pre_process(src, out)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[export] quant_pre_process unavailable ({str(e)[:50]}); trying shape inference")
        try:
            inline_save(onnx.shape_inference.infer_shapes(onnx.load(src)), out)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[export] shape inference failed ({str(e)[:50]}); using raw FP32")
            return src

    quant_src = prep_for_quant(fp32_path)
    input_name = ort.InferenceSession(quant_src, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    def preprocess_img(path):
        img = Image.open(path).convert("RGB")
        img = letterbox(img, size, pil_resample, pad_color)  # same geometry as serve
        arr = np.asarray(img).astype(np.float32) / 255.0          # HWC, [0,1]
        arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
        arr = np.transpose(arr, (2, 0, 1))[None, ...]             # NCHW
        return np.ascontiguousarray(arr, dtype=np.float32)

    calib_files = []
    if args.calib_data:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            calib_files += glob.glob(os.path.join(args.calib_data, "**", ext), recursive=True)
        import random as _r
        _r.Random(0).shuffle(calib_files)  # spread across subfolders/classes
        calib_files = calib_files[: args.calib_count]

    # ── nsfw.uint8.onnx — quantized build for the WASM (CPU) EP ───────────
    uint8_path = os.path.join(out_dir, "nsfw.uint8.onnx")
    if args.wasm_quant == "u8u8":
        q_kwargs = dict(per_channel=False,
                        weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8)
        scheme = "U8U8 (uint8 act + uint8 weight, per-tensor)"
    else:  # u8s8
        q_kwargs = dict(per_channel=True, reduce_range=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
        scheme = "U8S8 (uint8 act + per-channel int8 weight, reduce_range)"

    if calib_files:
        print(f"\n[export] static-quantizing -> nsfw.uint8.onnx  [{scheme}]  "
              f"with {len(calib_files)} calibration images")

        class Calib(CalibrationDataReader):
            def __init__(self, files):
                self.files = files
                self.i = 0

            def get_next(self):
                while self.i < len(self.files):
                    f = self.files[self.i]
                    self.i += 1
                    try:
                        return {input_name: preprocess_img(f)}
                    except Exception as e:  # noqa: BLE001
                        print(f"[calib] skip {f}: {e}")
                return None

            def rewind(self):
                self.i = 0

        quantize_static(quant_src, uint8_path, calibration_data_reader=Calib(calib_files),
                        quant_format=QuantFormat.QDQ, **q_kwargs)
        # verify the quantized model didn't collapse vs FP32 on a probe
        probe = torch.from_numpy(preprocess_img(calib_files[0]))
        fp = softmax(onnx_logits(fp32_path, probe))
        iq = softmax(onnx_logits(uint8_path, probe))
        agree = int(np.argmax(fp)) == int(np.argmax(iq))
        spread = float(abs(iq.max() - iq.min()))
        print(f"[verify] uint8 spread={spread:.3f}  argmax matches FP32: {agree}  "
              f"p(fp32)={fp.round(3).tolist()}  p(uint8)={iq.round(3).tolist()}")
        if not agree:
            print("[verify] !! uint8 disagrees with FP32 on the probe — quantization likely "
                  "collapsed (bad/unrepresentative calibration?). Shipping FP32 as the WASM "
                  "file so you never ship a broken model; re-run with better --calib-data.")
            inline_save(fp32_path, uint8_path)
        else:
            print("[verify] uint8 OK")
    else:
        print(f"\n[export] no --calib-data given — copying FP32 -> nsfw.uint8.onnx")
        print("[export] (correct but larger/slower; pass --calib-data <dir> for the real 8-bit build)")
        inline_save(fp32_path, uint8_path)
    uint8_mb = os.path.getsize(uint8_path) / (1024 * 1024)

    # ── nsfw.fp16.onnx — float16 build for the WebGPU EP ──────────────────
    fp16_path = os.path.join(out_dir, "nsfw.fp16.onnx")
    fp16_mb = None
    if not args.skip_fp16:
        from onnxconverter_common import float16  # noqa: E402
        # keep_io_types=True: inputs/outputs stay float32 so the JS side feeds a
        # normal Float32Array and reads float32 logits — only the internals are
        # fp16. The weights were inlined above, so the result is self-contained.
        mfp16 = float16.convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
        onnx.save_model(mfp16, fp16_path, save_as_external_data=False)
        # sanity check vs FP32 (CPU EP runs fp16 via casts; just confirm it's faithful)
        d = float(np.abs(onnx_logits(fp32_path, dummy) - onnx_logits(fp16_path, dummy)).max())
        fp_p = softmax(onnx_logits(fp32_path, dummy))
        h_p = softmax(onnx_logits(fp16_path, dummy))
        agree16 = int(np.argmax(fp_p)) == int(np.argmax(h_p))
        fp16_mb = os.path.getsize(fp16_path) / (1024 * 1024)
        print(f"\n[export] fp16 -> nsfw.fp16.onnx ({fp16_mb:.2f} MB)  "
              f"max|fp32-fp16|={d:.4f}  argmax matches FP32: {agree16}")
        if not agree16:
            print("[verify] !! fp16 argmax differs from FP32 on the probe — validate accuracy "
                  "on your val set before serving the WebGPU path (known fp16/WebGPU numerical "
                  "discrepancies exist in some ORT builds).")
    else:
        print("\n[export] --skip-fp16 set; not emitting nsfw.fp16.onnx")

    # cleanup the transient prep file (not the fp32 reference)
    if quant_src != fp32_path:
        _rm(quant_src)

    # ── summary ──────────────────────────────────────────────────────────
    print("\n[export] done. artifacts in", out_dir + ":")
    print(f"           nsfw.onnx        {fp32_mb:5.2f} MB   float32 reference / source")
    print(f"           nsfw.uint8.onnx  {uint8_mb:5.2f} MB   -> load on the 'wasm' (CPU) EP")
    if fp16_mb is not None:
        print(f"           nsfw.fp16.onnx   {fp16_mb:5.2f} MB   -> load on the 'webgpu' EP")
    print("[export] REMINDER: emitted filenames changed (uint8 + fp16, no more nsfw.int8.onnx).")
    print("[export]           Update your embed/build to copy + select these per execution provider.")
    if args.variant:
        print(f"[export] next:  npm run embed:{args.variant} && npm run build")
    else:
        print("[export] next:  npm run embed-model && npm run build")


if __name__ == "__main__":
    main()
