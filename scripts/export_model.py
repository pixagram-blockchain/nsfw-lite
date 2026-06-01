#!/usr/bin/env python
"""
export_model.py — export the trained timm MobileNetV4 binary NSFW model (from
train_nsfw.py) to ONNX + INT8, VERIFIED against PyTorch, for onnxruntime-web.

A LogitsOnly wrapper, an export-strategy sweep that keeps the FIRST config
matching PyTorch within 1e-3, and STATIC INT8 quantization with calibration.
No AvgPool patch is needed — timm MobileNetV4/V3 heads pool cleanly to ONNX
GlobalAveragePool (unlike HF EfficientNet's oversized fixed-kernel pooler).

Quantization scheme: per-channel INT8 weights + UInt8 activations (QDQ). This is
the ORT static-quant default that runs everywhere on the WASM backend and on
WebGPU via per-op CPU fallback for any unsupported QDQ op.

The checkpoint carries its own backbone name + labels + preprocessing, so this
script is self-describing; it re-emits labels.json / preprocess.json too.

INSTALL:
    pip install "timm>=1.0.0" torch onnx onnxruntime pillow numpy

RUN:
    python scripts/export_model.py                      # uses model/nsfw_mnv4.pt
    python scripts/export_model.py --calib-data data/   # sample calib imgs from dataset
Then:
    npm run embed-model && npm run build
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import timm
import onnxruntime as ort
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="model/nsfw_mnv4.pt")
    p.add_argument("--out-dir", default="model")
    p.add_argument("--calib-data", default=None,
                   help="dir of images (ImageFolder ok) to sample INT8 calibration from")
    p.add_argument("--calib-count", type=int, default=400)
    return p.parse_args()


def softmax(z):
    z = np.asarray(z).reshape(-1)
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = ckpt["backbone"]
    labels = ckpt["labels"]
    size = int(ckpt["size"])
    mean = [float(m) for m in ckpt["mean"]]
    std = [float(s) for s in ckpt["std"]]
    num_classes = len(labels)
    print(f"[export] backbone={backbone}  classes={labels}  size={size}")
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

    fp32_path = os.path.join(args.out_dir, "nsfw.onnx")
    tmp = os.path.join(args.out_dir, "_try.onnx")
    chosen = None
    print("\n[export] searching for a faithful export configuration:")
    for name, fn in strategies:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            fn(tmp)
            diff = float(np.abs(pt_dummy - onnx_logits(tmp, dummy)).max())
            ok = diff < 1e-3
            print(f"  - {name:28s} max|PT-ONNX|={diff:.6f}  {'PASS' if ok else 'reject'}")
            if ok:
                shutil.move(tmp, fp32_path)
                chosen = name
                break
        except Exception as e:  # noqa: BLE001
            print(f"  - {name:28s} export error: {str(e)[:80]}")
    if os.path.exists(tmp):
        os.remove(tmp)
    if not chosen:
        print("\n[export] FAILED: no configuration matched PyTorch. Do NOT ship.")
        raise SystemExit(2)
    print(f"[export] FP32 export OK via: {chosen}")

    # Re-emit labels + preprocess so the package is reproducible from the checkpoint.
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    preprocess = {
        "size": size, "cropSize": None, "doCenterCrop": False,
        "rescaleFactor": 1.0 / 255.0, "rescaleOffset": False,
        "doNormalize": True, "mean": mean, "std": std, "includeTop": False,
    }
    with open(os.path.join(args.out_dir, "preprocess.json"), "w") as f:
        json.dump(preprocess, f, indent=2)

    # ── INT8 static quantization (correct for CNNs) ──────────────────────
    from onnxruntime.quantization import (  # noqa: E402
        quantize_static, CalibrationDataReader, QuantType, QuantFormat,
    )

    src = fp32_path
    prepped = os.path.join(args.out_dir, "nsfw.prep.onnx")
    try:
        from onnxruntime.quantization import quant_pre_process
        quant_pre_process(fp32_path, prepped)
        src = prepped
    except Exception as e:  # noqa: BLE001
        print(f"[export] quant_pre_process unavailable ({e}); using raw FP32")

    input_name = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    def preprocess_img(path):
        img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0          # HWC, [0,1]
        arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
        arr = np.transpose(arr, (2, 0, 1))[None, ...]             # NCHW
        return np.ascontiguousarray(arr, dtype=np.float32)

    calib_files = []
    if args.calib_data:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            calib_files += glob.glob(os.path.join(args.calib_data, "**", ext), recursive=True)
        # spread across subfolders/classes rather than taking one class first
        import random as _r
        _r.Random(0).shuffle(calib_files)
        calib_files = calib_files[: args.calib_count]

    int8_path = os.path.join(args.out_dir, "nsfw.int8.onnx")

    if calib_files:
        print(f"\n[export] static-quantizing INT8 with {len(calib_files)} calibration images")

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

        quantize_static(
            src, int8_path,
            calibration_data_reader=Calib(calib_files),
            quant_format=QuantFormat.QDQ, per_channel=True,
            weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        )
        # verify INT8 didn't collapse vs FP32 on a probe (binary: spread = |p0-p1|)
        probe = torch.from_numpy(preprocess_img(calib_files[0]))
        fp = softmax(onnx_logits(fp32_path, probe))
        iq = softmax(onnx_logits(int8_path, probe))
        agree = int(np.argmax(fp)) == int(np.argmax(iq))
        spread = float(abs(iq.max() - iq.min()))
        print(f"[verify] INT8 spread={spread:.3f}  argmax matches FP32: {agree}  "
              f"p(fp32)={fp.round(3).tolist()}  p(int8)={iq.round(3).tolist()}")
        if not agree:
            print("[verify] INT8 disagrees with FP32 on the probe; shipping FP32 as the INT8 file.")
            shutil.copyfile(fp32_path, int8_path)
        else:
            print("[verify] INT8 OK")
    else:
        print(f"\n[export] no --calib-data given — copying FP32 -> {int8_path}")
        print("[export] (correct but larger; pass --calib-data <dir> for the small INT8 build)")
        shutil.copyfile(fp32_path, int8_path)

    try:
        if os.path.exists(prepped):
            os.remove(prepped)
    except OSError:
        pass

    mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"\n[export] done. embedded model will be {mb:.2f} MB")
    print("[export] next:  npm run embed-model && npm run build")


if __name__ == "__main__":
    main()
