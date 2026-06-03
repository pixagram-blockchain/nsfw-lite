#!/usr/bin/env python
"""
sanity_check.py — isolate WHERE binary NSFW detection breaks.

Runs the FP32 export and the uint8 model on the SAME image(s) in Python, using
the EXACT preprocessing emitted in model/preprocess.json — including its
`interpolation` (nearest vs bilinear), so a nearest-trained pixel-art model is
fed nearest-resized images here too. This removes JS preprocessing, the canvas,
and the Web Worker from the equation, so you learn which layer is at fault.

    NSFW_MODEL_DIR=model/mnv4 python scripts/sanity_check.py safe.png nsfw.png ...

Test BOTH distributions if you can: a few clearly-SFW and clearly-NSFW images,
AND (separately) one image that looks like your TRAINING data vs one that looks
like what you SERVE. That split is what separates "model is broken" from
"content-domain gap" (e.g. trained on photos, served on pixel art).

How to read it (binary: P(nsfw) is the number that matters):
  - FP32 decisive (high on nsfw, low on sfw), uint8 flat ~0.5
        -> quantization collapsed. Re-export STATIC with representative
           --calib-data (pixel art, not photos).
  - FP32 decisive here, but the BROWSER never flags
        -> JS-side. Confirm src/variants/mnv4/assets.generated.ts has real
           PREPROCESS + LABELS (not the empty stub) i.e. you ran `npm run
           embed:mnv4` AFTER export; that preprocess.json's interpolation is
           honored in core.ts; that wasmPaths is set; and your threshold isn't
           too high (lower thresholds.nsfw to flag more).
  - FP32 ALSO flat / always-low P(nsfw)
        -> the model itself never learned nsfw. Check the trainer's final
           macro-F1 (≈0.5 = it never learned), whether you trained the _l
           (035, random-init) model, the label order below, and the
           content-domain gap (does FP32 flag a TRAINING-style image but not a
           pixel-art one?).
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

# Default model dir (mnv4 layout); override for debugging, e.g.
#   NSFW_MODEL_DIR=model/mnv4 python scripts/sanity_check.py a.png b.png
OUT_DIR = os.environ.get("NSFW_MODEL_DIR", "model/mnv4")

imgs = sys.argv[1:]
if not imgs:
    print("usage: NSFW_MODEL_DIR=model/mnv4 python scripts/sanity_check.py <image> [<image> ...]")
    sys.exit(1)

with open(os.path.join(OUT_DIR, "labels.json")) as f:
    labels = [str(x).lower() for x in json.load(f)]
with open(os.path.join(OUT_DIR, "preprocess.json")) as f:
    pp = json.load(f)
nsfw_i = labels.index("nsfw") if "nsfw" in labels else 0

size = int(pp["size"])
mean = np.array(pp["mean"], np.float32)
std = np.array(pp["std"], np.float32)
rescale = float(pp.get("rescaleFactor", 1 / 255))

# Honor the resize filter the model was trained + calibrated with. Hardcoding
# bilinear for a nearest-trained model is itself a parity bug that skews results.
_RES = getattr(Image, "Resampling", Image)  # Pillow >=9.1 moved the enum
interp = str(pp.get("interpolation", "bilinear"))
resample = {"nearest": _RES.NEAREST, "bilinear": _RES.BILINEAR}.get(interp, _RES.BILINEAR)

print(f"[sanity] labels (id order): {labels}  (reading P(nsfw) at index {nsfw_i})")
print(f"[sanity] size={size}  interp={interp}  mean={mean.tolist()}  std={std.tolist()}")


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((size, size), resample)
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
uint8_sess, uint8_path = load("nsfw.uint8.onnx")

for p in imgs:
    print(f"\n{os.path.basename(p)}")
    show("FP32 ", fp32_sess, fp32_path, p)
    show("UINT8", uint8_sess, uint8_path, p)

print("\nspread < 0.05 = collapsed/uniform (bad). P(nsfw) near 0 or 1 = confident.")
