#!/usr/bin/env python
"""
sanity_check.py — isolate WHERE binary NSFW detection breaks.

Runs the FP32 export and the INT8 model on the SAME image(s) in Python, using
the EXACT preprocessing emitted in model/preprocess.json (plain resize -> /255
-> normalize). This removes JS preprocessing, the canvas, and the Web Worker
from the equation, so you learn exactly which layer is at fault.

    python scripts/sanity_check.py path/to/safe.jpg path/to/nsfw.jpg ...

How to read it (binary: P(nsfw) is the number that matters):
  - INT8 P(nsfw) flat ~0.5 but FP32 decisive
        -> quantization collapsed the model. Re-quantize STATIC (export_model.py),
           supply more --calib-data.
  - BOTH flat ~0.5
        -> the export/model itself is the problem. Re-train / re-export.
  - BOTH decisive here in Python, but the BROWSER is flat ~0.5
        -> JS-side. Confirm src/assets.generated.ts has real PREPROCESS + LABELS
           (not the empty stub), and that wasmPaths is set so ORT loads. The
           canvas resize is then the place to look for parity drift.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

# Default model dir; override per-variant for debugging, e.g.
#   NSFW_MODEL_DIR=model/lcnet python scripts/sanity_check.py a.jpg b.jpg
OUT_DIR = os.environ.get("NSFW_MODEL_DIR", "model")

imgs = sys.argv[1:]
if not imgs:
    print("usage: python scripts/sanity_check.py <image> [<image> ...]")
    sys.exit(1)

with open(os.path.join(OUT_DIR, "labels.json")) as f:
    labels = [str(x).lower() for x in json.load(f)]
with open(os.path.join(OUT_DIR, "preprocess.json")) as f:
    pp = json.load(f)
print(f"[sanity] labels (id order): {labels}")
nsfw_i = labels.index("nsfw") if "nsfw" in labels else 0

size = int(pp["size"])
mean = np.array(pp["mean"], np.float32)
std = np.array(pp["std"], np.float32)
rescale = float(pp.get("rescaleFactor", 1 / 255))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) * rescale
    if pp.get("doNormalize", True):
        arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr, np.float32)


def load(name):
    full = os.path.join(OUT_DIR, name)
    if not os.path.exists(full):
        return None, full
    return ort.InferenceSession(full, providers=["CPUExecutionProvider"]), full


def run(sess, path):
    x = preprocess(path)
    name = sess.get_inputs()[0].name
    logits = np.asarray(sess.run(None, {name: x})[0]).reshape(-1)
    return logits, softmax(logits)


def show(tag, sess, missing, path):
    if sess is None:
        print(f"  {tag}: (missing {missing})")
        return
    logits, probs = run(sess, path)
    p_nsfw = float(probs[nsfw_i])
    spread = float(abs(probs.max() - probs.min()))
    flag = "  <-- COLLAPSED" if spread < 0.05 else ""
    pretty = ", ".join(f"{labels[i]}={probs[i]:.3f}" for i in range(len(labels)))
    print(f"  {tag}: P(nsfw)={p_nsfw:.3f}  spread={spread:.3f}{flag}")
    print(f"        probs : {pretty}")
    print(f"        logits: [{', '.join(f'{v:+.2f}' for v in logits)}]")


fp32_sess, fp32_path = load("nsfw.onnx")
int8_sess, int8_path = load("nsfw.int8.onnx")

for p in imgs:
    print(f"\n{os.path.basename(p)}")
    show("FP32", fp32_sess, fp32_path, p)
    show("INT8", int8_sess, int8_path, p)

print("\nspread < 0.05 = collapsed/uniform (bad). P(nsfw) near 0 or 1 = confident.")
