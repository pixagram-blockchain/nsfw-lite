#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "timm>=1.0.0",
#   "torch",
#   "torchvision",
#   "datasets>=2.14.0",
#   "pillow",
#   "numpy",
#   "huggingface_hub>=0.24.0",
# ]
# ///
"""
train_nsfw_hf_mnv4_160.py — the LIGHTWEIGHT production trainer: the
mobilenetv4_conv_small_050 backbone (~0.96M params) at 160px, the smallest and
fastest member of the lineup, tuned to ship in the browser.

Same data sources and the SAME parity transforms / outputs as the rest of the
family — model/mnv4/nsfw_mnv4.pt, labels.json, preprocess.json — so
scripts/export_model.py (--variant mnv4) and the embed/build steps work unchanged.

Two things differ from the reference mnv4 trainer:
  1. --img-size defaults to 160 (was native 224). For pixel-art SFW/NSFW that's
     plenty, and at ~0.14x the compute of the full conv_small@256 it's the cheap
     end of the lineup. The size flows into preprocess.json + the checkpoint, so
     train-, export- and serve-time resolution stay identical (parity holds).
  2. OPTIONAL knowledge distillation (--teacher-checkpoint): train this small
     student to imitate a bigger, more accurate teacher (e.g. the full conv_small
     at 256). The student matches the teacher's softened logits ON TOP OF the hard
     labels, recovering much of the teacher's accuracy at a fraction of the cost —
     the principled way to get "big-model accuracy in a small, fast model." OFF
     unless a teacher is given, so by default this is simply the _050@160 trainer.

The rest of the precision recipe is unchanged: Mixup (auto-disabled under
distillation), RandomErasing, EMA, a cosine schedule with early-stop, and a
per-epoch nsfw AUC readout. All train-only / parity-preserving.

WHERE THE DATA COMES FROM (pick one):
  --nsfw-dataset R1 --sfw-dataset R2   two SEPARATE single-class Hub datasets;
                                       all of R1 is labelled nsfw, all of R2 sfw.
  --hf-dataset <user/repo>             ONE Hub dataset with a label column / class
                                       subfolders (private/gated is fine).
  --data-dir <path>                    a LOCAL ImageFolder with two class subdirs.

The classes normalize to {nsfw, sfw} (names map through ALIASES below). In the
two-dataset mode index order is fixed nsfw=0, sfw=1; otherwise it follows the
dataset's ClassLabel names. labels.json is written in THAT order and the JS side
reads the `nsfw` score by name, so the order never actually matters.

RUN IT ON HUGGING FACE JOBS (GPU, no local setup — this file is a UV script):
    # Jobs are a Pro/Team/Enterprise feature. Authenticate first: `hf auth login`.
    # Your two datasets, on a big GPU, pushing the result to your account:
    hf jobs uv run --flavor a100-large --timeout 2h -s HF_TOKEN scripts/train_nsfw_hf_mnv4_160.py \
        -- --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
           --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata \
           --push-repo <you>/nsfw_mnv4_160 --interp nearest \
           --batch-size 160 --num-workers 16 --patience 5
    # (you can also pass a public URL to this file instead of a local path.)

  GPU flavors run small -> large: t4-small, l4x1, a10g-small, a10g-large,
  a100-large (80 GB), and higher tiers (H100/H200/B200) if your plan exposes them.
  BUT this backbone is ~2.2M params / 0.1 GMACs — on anything past an L4/A10G the
  GPU sits idle and throughput is bound by IMAGE DECODE + data loading, not matmul.
  So an a100-large buys headroom for a big --batch-size, not a proportional
  speedup; raise --num-workers (decode parallelism) for the real win. An
  a10g-large is the cost-sensible pick.

  HF Jobs storage is EPHEMERAL — the filesystem is wiped when the job ends. Pass
  --push-repo so the trained model/ folder is uploaded to the Hub; then locally:
      hf download <you>/nsfw_mnv4_160 --repo-type model --local-dir model/mnv4/
      python scripts/export_model.py --variant mnv4 --calib-data <calib-dir>
      npm run embed:mnv4 && npm run build

RUN IT LOCALLY (or in a GPU Space / notebook) the ordinary way:
    pip install "timm>=1.0.0" torch torchvision "datasets>=2.14.0" pillow numpy huggingface_hub
    python scripts/train_nsfw_hf_mnv4.py \
        --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
        --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata
    python scripts/train_nsfw_hf_mnv4.py --data-dir ./data   # local imagefolder

RESPONSIBLE-USE NOTE: if you upload your NSFW images to a Hub dataset, make the
repo PRIVATE or gated, review Hugging Face's current Content Policy, and screen
for illegal material (run perceptual-hash matching against known-bad sets, e.g.
CSAM, BEFORE uploading). You remain legally responsible for data you hold or push.
"""
import argparse
import json
import math
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

# Map common label/folder names onto the two canonical classes. Anything not
# listed is lowercased and used as-is (so `nsfw` and `sfw` need no mapping).
ALIASES = {
    "safe": "sfw", "neutral": "sfw", "clean": "sfw", "normal": "sfw", "ham": "sfw",
    "explicit": "nsfw", "porn": "nsfw", "adult": "nsfw", "unsafe": "nsfw",
}
CANONICAL = {"sfw", "nsfw"}

DEFAULT_BACKBONE = "mobilenetv4_conv_small_050.e3000_r224_in1k"


# torch.amp moved out of torch.cuda.amp; support both so this runs across versions.
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def make_scaler(enabled):
        return _GradScaler("cuda", enabled=enabled)

    def amp_ctx(enabled):
        return _autocast("cuda", enabled=enabled)
except Exception:  # noqa: BLE001
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler  # type: ignore

    def make_scaler(enabled):
        return _GradScaler(enabled=enabled)

    def amp_ctx(enabled):
        return _autocast(enabled=enabled)


class ModelEma:
    """Minimal exponential moving average of model weights (version-proof; no
    timm-version coupling). Float tensors (incl. BN running stats) are averaged;
    integer buffers (e.g. num_batches_tracked) are copied. Initialize it AFTER
    warmup via .start() so the average isn't anchored to the random-head init."""

    def __init__(self, decay: float):
        import copy as _copy
        self._copy = _copy
        self.decay = decay
        self.module = None  # lazily set by start()

    def start(self, model):
        self.module = self._copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if self.module is None:
            return
        ema_sd = self.module.state_dict()
        for k, v in model.state_dict().items():
            ev = ema_sd[k]
            if ev.dtype.is_floating_point:
                ev.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                ev.copy_(v)


def letterbox(img, size, resample, pad_color=(0, 0, 0)):
    """Map an arbitrary-aspect PIL image to a `size`x`size` RGB image by
    LETTERBOXING — preserve the aspect ratio, fit the WHOLE image inside, and pad
    the margins with pad_color. Same geometry as export_model.py's letterbox and
    the browser's resizeToSquare (src/core.ts), so train-val, calibration, and
    serve agree. Dims use round-half-up and offsets use floor (JS Math.round /
    Math.floor)."""
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


class Letterbox:
    """Deterministic letterbox transform (PIL image -> PIL image), the val/eval
    geometry. RandomLetterbox below is its train-time, jittered counterpart."""

    def __init__(self, size, interp, pad_color=(0, 0, 0)):
        self.size = int(size)
        self.pad_color = tuple(int(c) for c in pad_color)[:3]
        _RES = getattr(Image, "Resampling", Image)  # Pillow >=9.1 moved the enum
        self.resample = {"bilinear": _RES.BILINEAR,
                         "nearest": _RES.NEAREST}.get(interp, _RES.BILINEAR)

    def __call__(self, img):
        return letterbox(img.convert("RGB"), self.size, self.resample, self.pad_color)


class RandomLetterbox:
    """Train-time analogue of 'border' serve: fit the WHOLE image inside
    size x size (aspect ratio preserved), with a random shrink (area in `scale`)
    and a random translation, padding the rest with pad_color. At area=1 and
    centred it equals letterbox(...); the randomness puts letterbox
    bars of VARYING size and position into the train distribution, so the model
    meets the bars it will see at serve instead of only at inference. PIL->PIL;
    flip / colour-jitter / erase follow in the Compose."""

    def __init__(self, size, interp, pad_color=(0, 0, 0), scale=(0.8, 1.0)):
        self.size = int(size)
        self.pad = tuple(int(c) for c in pad_color)[:3]
        self.lo, self.hi = float(scale[0]), float(scale[1])
        _RES = getattr(Image, "Resampling", Image)
        self.resample = {"bilinear": _RES.BILINEAR,
                         "nearest": _RES.NEAREST}.get(interp, _RES.BILINEAR)

    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        S = self.size
        if w == 0 or h == 0:
            return Image.new("RGB", (S, S), self.pad)
        bs = min(S / w, S / h)                       # fit whole image inside S x S
        f = math.sqrt(random.uniform(self.lo, self.hi))  # area -> linear scale, <=1
        dw = max(1, min(S, int(w * bs * f + 0.5)))
        dh = max(1, min(S, int(h * bs * f + 0.5)))
        resized = img.resize((dw, dh), self.resample)
        ox = random.randint(0, S - dw)               # random translation (centred at eval)
        oy = random.randint(0, S - dh)
        canvas = Image.new("RGB", (S, S), self.pad)
        canvas.paste(resized, (ox, oy))
        return canvas


def parse_args():
    p = argparse.ArgumentParser()
    # Data source — choose ONE of these three:
    #   (a) --nsfw-dataset + --sfw-dataset : two SEPARATE single-class Hub datasets.
    #       Each is labelled wholesale (all of nsfw-dataset -> nsfw, etc). This is the
    #       layout of the civitai-top-*-images-with-metadata datasets.
    #   (b) --hf-dataset                   : ONE Hub dataset that already carries a
    #       label column / two class subfolders.
    #   (c) --data-dir                     : a LOCAL ImageFolder with two class subdirs.
    p.add_argument("--nsfw-dataset", default=None,
                   help="Hub dataset whose images are ALL nsfw (pair with --sfw-dataset)")
    p.add_argument("--sfw-dataset", default=None,
                   help="Hub dataset whose images are ALL sfw (pair with --nsfw-dataset)")
    p.add_argument("--hf-dataset", default=None,
                   help="single Hub dataset that already has a label column / class subfolders")
    p.add_argument("--data-dir", default=None,
                   help="local ImageFolder root (loaded via the 'imagefolder' builder)")
    p.add_argument("--config-name", default=None, help="dataset config/subset name, if any")
    p.add_argument("--train-split", default="train", help="split to train on (default: train)")
    p.add_argument("--val-split", default=None,
                   help="held-out split name; if omitted, carve --val-frac out of the train split")
    p.add_argument("--image-column", default="image", help="image column name (default: image)")
    p.add_argument("--label-column", default="label",
                   help="label column for --hf-dataset/--data-dir mode (default: label)")

    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="timm model id (default: mobilenetv4_conv_small_050.e3000_r224_in1k, the "
                        "~2.2M-param lightweight centerpiece — kept as-is since it's the proven "
                        "winner). For more capacity: mobilenetv4_conv_small.e2400_r224_in1k "
                        "(~3.8M, full width) or mobilenetv4_conv_medium.e500_r224_in1k (~9.7M).")
    p.add_argument("--img-size", type=int, default=160,
                   help="input resolution (DEFAULT 160 for this lightweight build). "
                        "mnv4_conv_small_050 is fully convolutional, so this is a free lever: "
                        "160px is ~0.14x the compute of the full conv_small at 256 and is "
                        "plenty for pixel-art SFW/NSFW, whose source images rarely carry more "
                        "than that. The chosen size is written into preprocess.json, so the "
                        "browser resizes to the SAME value — train/serve parity holds. Raise it "
                        "(e.g. 224) only if you measure a recall gain that justifies the cost.")
    p.add_argument("--interp", default="bilinear", choices=["bilinear", "nearest"],
                   help="resize filter applied at EVERY stage (train aug, val, and — via the "
                        "checkpoint — calibration + the browser). 'nearest' = pixelated: crisp "
                        "pixel blocks with no blur, the faithful choice for pixel art that is "
                        "UPSCALED to the input size. (It ALIASES when DOWNSCALING large images, "
                        "so prefer it when sources are small sprites at/below the input size.) "
                        "'bilinear' (default) = smooth, matching a canvas with image smoothing on. "
                        "The choice is recorded in the checkpoint so export + serve stay in parity "
                        "automatically; only nearest/bilinear are offered because those are the "
                        "two a browser <canvas> can reproduce exactly.")
    p.add_argument("--pad-color", type=int, nargs=3, default=[0, 0, 0], metavar=("R", "G", "B"),
                   help="letterbox fill colour, 0-255 (default 0 0 0 = black; matches the JS "
                        "padColor default). The model always sees letterboxed input — aspect "
                        "ratio preserved, whole image fit inside, margins padded with this — "
                        "because cropping drops edge content and squashing distorts wide images, "
                        "both of which cause missed detections in moderation. Recorded in the "
                        "checkpoint so export calibration and the browser use the same pad.")

    # ── Optional knowledge distillation ──────────────────────────────────
    # Train the small _050 student to imitate a bigger, more accurate teacher
    # (e.g. the full mobilenetv4_conv_small at 256). The student learns the
    # teacher's SOFTENED logits on top of the hard labels, recovering much of the
    # teacher's accuracy at a fraction of the compute. All OFF unless a teacher is
    # given, so the script is a plain _050@160 trainer by default.
    p.add_argument("--teacher-checkpoint", default=None,
                   help="path to a trained teacher .pt (e.g. model/mnv4_full/nsfw_mnv4.pt, a "
                        "full conv_small trained at 256). When set, distillation is ON.")
    p.add_argument("--teacher-backbone", default=None,
                   help="override the teacher's backbone id (default: read it from the teacher "
                        "checkpoint, which is the normal case).")
    p.add_argument("--kd-alpha", type=float, default=0.7,
                   help="distillation mix in [0,1]: loss = kd_alpha*KD + (1-kd_alpha)*CE_hard "
                        "(default 0.7). Only used when --teacher-checkpoint is given.")
    p.add_argument("--kd-temp", type=float, default=4.0,
                   help="softmax temperature for the KD term (default 4.0; higher = softer "
                        "targets that expose more of the teacher's inter-class confidence).")
    p.add_argument("--epochs", type=int, default=20,
                   help="max epochs (default 20; stronger aug benefits from a longer run. "
                        "Best-by-macro-F1 checkpoint is kept, and --patience early-stops on "
                        "plateau, so this is a ceiling, not a fixed cost).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5,
                   help="cosine floor (default 1e-5). The schedule decays to this instead of 0 so "
                        "the final epochs keep learning a little rather than freezing.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.1, help="used only if --val-split not given")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="model/mnv4")

    # Augmentation / regularization knobs.
    p.add_argument("--rrc-min", type=float, default=0.8,
                   help="lower area bound for the random letterbox shrink during training "
                        "(default 0.8). The train aug fits the whole image inside the square "
                        "(like serve) and randomly shrinks it within [rrc-min, 1] + translates, "
                        "so the model sees bars of varying size/position. Higher = gentler "
                        "(less bar-size jitter), keeping training closer to the deterministic "
                        "eval letterbox.")
    p.add_argument("--random-erase", type=float, default=0.25,
                   help="RandomErasing probability (default 0.25; 0 disables). Occludes a small "
                        "patch — a cheap regularizer that, unlike heavy cropping, leaves the "
                        "subject in frame. Train-only, so serve-time parity is untouched.")
    p.add_argument("--mixup-alpha", type=float, default=0.1,
                   help="Mixup alpha (default 0.1; 0 disables). Blends whole image PAIRS and their "
                        "labels — a strong, parity-safe regularizer that smooths the decision "
                        "boundary and improves precision on borderline images. When active, an "
                        "unweighted soft-target loss is used (fine for a balanced set).")
    p.add_argument("--cutmix-alpha", type=float, default=0.0,
                   help="CutMix alpha (default 0.0 = OFF). Left off on purpose for moderation: "
                        "pasting an sfw patch over an nsfw image's explicit region while lowering "
                        "its label can teach under-flagging of partially-occluded content. Enable "
                        "only if you understand that risk.")
    p.add_argument("--mixup-prob", type=float, default=1.0,
                   help="probability a batch is mixed when mixup/cutmix is active (default 1.0).")
    p.add_argument("--mixup-switch-prob", type=float, default=0.5,
                   help="prob of choosing cutmix over mixup when BOTH alphas > 0 (default 0.5).")
    p.add_argument("--ema", dest="ema", action="store_true", default=True,
                   help="track an exponential moving average of the weights; eval both raw and "
                        "EMA each epoch and keep whichever scores higher (pure upside). ON by "
                        "default for this trainer.")
    p.add_argument("--no-ema", dest="ema", action="store_false",
                   help="disable EMA weight averaging.")
    p.add_argument("--ema-decay", type=float, default=0.998,
                   help="EMA decay (default 0.998). Lower for short runs: the averaging window is "
                        "~1/(1-decay) steps, so 0.998≈500 steps. EMA starts after warmup.")
    p.add_argument("--patience", type=int, default=5,
                   help="early-stop after this many epochs without a macro-F1 improvement "
                        "(default 5; 0 = disabled). Best checkpoint is always kept.")

    # Optional: push the trained model/ folder to the Hub (essential on HF Jobs,
    # whose local storage is wiped when the job finishes).
    p.add_argument("--push-repo", default=None,
                   help="repo id to upload model/ to after training (e.g. you/nsfw-lite-mnv4)")
    p.add_argument("--push-repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--push-private", action="store_true", default=True,
                   help="create the push repo as private (default: True)")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def normalize_labels(names):
    labels = [ALIASES.get(str(c).lower(), str(c).lower()) for c in names]
    if len(names) != 2 or set(labels) != CANONICAL:
        raise SystemExit(
            f"[train] ERROR: expected exactly two classes mapping to {sorted(CANONICAL)}, "
            f"but got {list(names)} -> {labels}.\n"
            f"        Use two classes named nsfw/sfw (aliases: {sorted(ALIASES)})."
        )
    return labels


def _rank_auc(scores, positive):
    """ROC-AUC for the positive class via the Mann–Whitney U statistic — no
    sklearn dependency. `scores` = P(nsfw) per sample, `positive` = boolean mask
    of true-nsfw rows. Ties get arbitrary (not averaged) ranks, which is fine for
    epoch-to-epoch monitoring since float probabilities almost never tie."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, loader, device, num_classes, nsfw_index=None):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    probs_nsfw, true_y = [], []  # for threshold-independent AUC on the nsfw class
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(ce(logits, y)) * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1).cpu().numpy()
        yc = y.cpu().numpy()
        for t, pr in zip(yc, pred):
            cm[t, pr] += 1
        if nsfw_index is not None:
            probs_nsfw.append(torch.softmax(logits, dim=1)[:, nsfw_index].float().cpu().numpy())
            true_y.append(yc)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    acc = tp.sum() / max(1, cm.sum())
    auc = float("nan")
    if nsfw_index is not None and probs_nsfw:
        scores = np.concatenate(probs_nsfw)
        ys = np.concatenate(true_y)
        auc = _rank_auc(scores, ys == nsfw_index)
    return {"loss": loss_sum / max(1, n), "acc": float(acc),
            "prec": prec, "rec": rec, "f1": f1, "macro_f1": float(f1.mean()),
            "auc": auc, "cm": cm}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  backbone={args.backbone}")

    # Fail fast on auth: pushing the model needs a WRITE token in a namespace you
    # own. Verify BOTH up front — on HF Jobs a late failure loses the whole run to
    # ephemeral storage. (HfApi() reads HF_TOKEN from the env or the login file.)
    if args.push_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            who = api.whoami()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                "[train] Hugging Face auth failed for --push-repo (token missing or INVALID).\n"
                "        Check ALL of:\n"
                "          - HF_TOKEN holds your REAL token (not a placeholder like hf_xxxx)\n"
                "            with no quotes/spaces/newlines.\n"
                "          - HF_TOKEN OVERRIDES `hf auth login`. If it's stale, `unset HF_TOKEN`\n"
                "            (also check ~/.bashrc, ~/.zshrc, and any .env) and log in again.\n"
                "          - get or rotate a token at https://huggingface.co/settings/tokens\n"
                "        On HF Jobs, forward with `-s HF_TOKEN` and ensure it's set+valid on the\n"
                "        launch machine. Or drop --push-repo to train without uploading.\n"
                f"        underlying error: {e}"
            )

        name = who.get("name", "?")

        # WRITE-scope check. A read token passes whoami() but can't create/push a
        # repo (the 401 on /api/repos/create). Determine the token's role best-effort;
        # classic tokens expose it, fine-grained tokens may report None (then we don't
        # block — the push try/except below will still give a clean message).
        role = None
        try:
            role = api.get_token_permission()  # -> "read" | "write" | None
        except Exception:  # noqa: BLE001
            try:
                role = who.get("auth", {}).get("accessToken", {}).get("role")
            except Exception:  # noqa: BLE001
                role = None
        if role == "read":
            raise SystemExit(
                "[train] your token authenticated but is READ-ONLY — it cannot create or\n"
                "        push to a repo (that's the 401 on /api/repos/create).\n"
                "        Fix: create a token with the WRITE role (or a fine-grained token with\n"
                "        'Write access to contents/settings of repos you own') at\n"
                "        https://huggingface.co/settings/tokens, then set HF_TOKEN to it.\n"
                "        (Re-using a read token from `hf auth login`? HF_TOKEN overrides it —\n"
                "        `unset HF_TOKEN` or export the new write token.)"
            )

        # Namespace check: you can only push under your own username or an org you
        # belong to. A typo'd or someone-else's namespace also 401s on create.
        owner = args.push_repo.split("/")[0] if "/" in args.push_repo else name
        orgs = [o.get("name") for o in who.get("orgs", [])]
        if owner != name and owner not in orgs:
            raise SystemExit(
                f"[train] --push-repo namespace '{owner}' isn't yours. You are '{name}'"
                + (f" (orgs: {', '.join(o for o in orgs if o)})" if any(orgs) else "")
                + f".\n        Use '{name}/<repo>' or an org you belong to."
            )

        print(f"[train] hub auth OK — '{name}'"
              + (f", token role: {role}" if role else "") + f", pushing to {args.push_repo}")

    from datasets import load_dataset, concatenate_datasets, ClassLabel, Image as HFImage

    img_col = args.image_column
    lbl_col = "label"  # every mode is normalized to a 'label' ClassLabel column

    def _pick_split(d):
        return args.train_split if args.train_split in d else list(d.keys())[0]

    def _single_class(repo, label_value, tag):
        """Load a one-class Hub dataset and tag every row with `label_value`."""
        d = load_dataset(repo, args.config_name)
        s = d[_pick_split(d)]
        if img_col not in s.column_names:
            raise SystemExit(f"[train] '{img_col}' not in {tag} dataset columns "
                             f"{s.column_names}; set --image-column.")
        # Drop any metadata columns so the two sources share an identical schema
        # and concatenate cleanly; we only need the image.
        drop = [c for c in s.column_names if c != img_col]
        if drop:
            s = s.remove_columns(drop)
        # Normalize the image feature so two independently-built datasets match.
        s = s.cast_column(img_col, HFImage())
        s = s.add_column(lbl_col, [label_value] * len(s))
        print(f"[train] {tag}: {len(s)} images")
        return s

    val_external = None
    if args.nsfw_dataset or args.sfw_dataset:
        # (a) two separate single-class datasets — nsfw=0, sfw=1.
        if not (args.nsfw_dataset and args.sfw_dataset):
            raise SystemExit("[train] pass BOTH --nsfw-dataset and --sfw-dataset.")
        print("[train] two single-class Hub datasets (nsfw=0, sfw=1):")
        train_raw = concatenate_datasets([
            _single_class(args.nsfw_dataset, 0, "nsfw"),
            _single_class(args.sfw_dataset, 1, "sfw"),
        ]).cast_column(lbl_col, ClassLabel(names=["nsfw", "sfw"]))
        labels = ["nsfw", "sfw"]
    elif args.hf_dataset or args.data_dir:
        # (b) single Hub dataset with a label column, or (c) local imagefolder.
        is_local = bool(args.data_dir)
        spec = args.data_dir or args.hf_dataset
        print(f"[train] {'local imagefolder' if is_local else 'Hub dataset'}: {spec}")
        d = load_dataset("imagefolder", data_dir=spec) if is_local \
            else load_dataset(spec, args.config_name)
        train_raw = d[_pick_split(d)]
        ext = args.label_column
        if img_col not in train_raw.column_names or ext not in train_raw.column_names:
            raise SystemExit(
                f"[train] need columns ({img_col}, {ext}) in {train_raw.column_names}; set "
                "--image-column/--label-column, or use --nsfw-dataset/--sfw-dataset."
            )
        if not isinstance(train_raw.features[ext], ClassLabel):
            train_raw = train_raw.class_encode_column(ext)
        labels = normalize_labels(train_raw.features[ext].names)  # native (normalized) order
        if ext != lbl_col:
            train_raw = train_raw.rename_column(ext, lbl_col)
        if args.val_split and args.val_split in d:
            val_external = d[args.val_split]
            if ext in val_external.column_names and not isinstance(val_external.features[ext], ClassLabel):
                val_external = val_external.class_encode_column(ext)
            normalize_labels(val_external.features[ext].names)  # validate held-out split
            if ext != lbl_col:
                val_external = val_external.rename_column(ext, lbl_col)
    else:
        raise SystemExit("[train] choose a data source: --nsfw-dataset + --sfw-dataset, "
                         "--hf-dataset, or --data-dir.")

    num_classes = 2
    print(f"[train] labels (index order): {labels}")

    # ── Build the model first so transforms match its native size/mean/std ──
    probe = timm.create_model(args.backbone, pretrained=True, num_classes=num_classes)
    dcfg = timm.data.resolve_model_data_config(probe)
    native_size = int(dcfg["input_size"][-1])
    mean = [float(m) for m in dcfg["mean"]]
    std = [float(s) for s in dcfg["std"]]
    del probe
    # Resolution: default to the backbone's native size (224). mnv4_conv_small is
    # fully convolutional, so --img-size can raise it (e.g. 256, the resolution
    # timm evaluates this model at). The chosen size flows into the transforms AND
    # into preprocess.json / the checkpoint below, so train-, export- and serve-
    # time resolution all stay identical (parity preserved). mean/std are
    # resolution-independent and still come from the backbone's data config.
    size = int(args.img_size) if args.img_size else native_size
    if size != native_size:
        print(f"[train] input size={size} (overriding backbone native {native_size}px)  "
              f"mean={mean}  std={std}")
    else:
        print(f"[train] input size={size}  mean={mean}  std={std}")

    # The resize filter (args.interp) is shared by the train aug + val so the
    # model only ever sees one interpolation; it's saved in the checkpoint and
    # flows to calibration + the browser, so train/serve parity holds.
    pad_color = tuple(int(c) for c in args.pad_color)[:3]

    # Train spatial augmentation = RandomLetterbox: fit the WHOLE image inside the
    # square (preserve aspect ratio), with a random shrink + translation, padded.
    # This matches the deterministic letterbox served at eval (Letterbox below)
    # and, crucially, puts the bars the model sees at serve into the TRAIN
    # distribution. (We letterbox rather than RandomResizedCrop because cropping
    # drops edge content and squashing distorts wide images — both miss detections
    # in moderation.) flip / colour-jitter / erase follow.
    spatial = RandomLetterbox(size, args.interp, pad_color, scale=(args.rrc_min, 1.0))

    train_tf_list = [
        spatial,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
    ]
    # RandomErasing operates on the tensor; place it BEFORE Normalize so the
    # erased patch is black in [0,1] space (a real pixel value) and then gets
    # normalized like everything else. Train-only — val/serve parity untouched.
    if args.random_erase and args.random_erase > 0:
        train_tf_list.append(transforms.RandomErasing(p=args.random_erase))
    train_tf_list.append(transforms.Normalize(mean, std))
    train_tf = transforms.Compose(train_tf_list)

    # Deterministic letterbox = the val/serve geometry, identical to
    # export_model.py's letterbox and the browser's resizeToSquare. The train aug
    # above is its jittered counterpart, so train and serve share one geometry.
    val_tf = transforms.Compose([
        Letterbox(size, args.interp, pad_color),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Train/val splits ─────────────────────────────────────────────────
    if val_external is not None:
        val_raw = val_external
    else:
        split = train_raw.train_test_split(
            test_size=args.val_frac, seed=args.seed, stratify_by_column=lbl_col
        )
        train_raw, val_raw = split["train"], split["test"]
        print(f"[train] split: {len(train_raw)} train / {len(val_raw)} val")

    # Class weights from the RAW (pre-transform) labels — reading the column does
    # not decode images. Inverse-frequency to counter imbalance (10k/10k ≈ 1.0).
    train_targets = np.array(train_raw[lbl_col], dtype=np.int64)
    counts = np.clip(np.bincount(train_targets, minlength=num_classes).astype(np.float64), 1, None)
    weights = counts.sum() / (num_classes * counts)
    print("[train] class counts: " + ", ".join(f"{l}={int(c)}" for l, c in zip(labels, counts)))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    # Attach transforms lazily (run on __getitem__, decoding the image to a tensor).
    def make_apply(tf):
        def _apply(batch):
            batch["pixel_values"] = [tf(im.convert("RGB")) for im in batch[img_col]]
            return batch
        return _apply

    train_ds = train_raw.with_transform(make_apply(train_tf))
    val_ds = val_raw.with_transform(make_apply(val_tf))

    def collate(examples):
        x = torch.stack([e["pixel_values"] for e in examples])
        y = torch.tensor([int(e[lbl_col]) for e in examples], dtype=torch.long)
        return x, y

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin, collate_fn=collate)

    # ── Train ────────────────────────────────────────────────────────────
    model = timm.create_model(args.backbone, pretrained=True, num_classes=num_classes).to(device)

    # Mixup/CutMix (timm). Blends image pairs + labels into soft targets — a
    # strong, train-only regularizer (val/serve parity is untouched). When it's
    # active the loss MUST be a soft-target CE; the per-class weighting is dropped
    # (it doesn't compose with soft targets), which is a no-op on a balanced set.
    # Mixup already smooths the labels, so we hand it --label-smoothing and don't
    # double-smooth in the loss.
    mixup_active = (args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0)
    mixup_fn = None
    if mixup_active:
        # timm's Mixup asserts an even batch size. drop_last=True means every
        # batch is exactly --batch-size, so an odd value would fail every step;
        # catch it now with a clear message rather than mid-training.
        if args.batch_size % 2 != 0:
            raise SystemExit(
                f"[train] --batch-size must be even when mixup/cutmix is active "
                f"(got {args.batch_size}). Use an even size, or disable mixup with "
                f"--mixup-alpha 0 --cutmix-alpha 0."
            )
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
            mode="batch", label_smoothing=args.label_smoothing, num_classes=num_classes,
        )
        criterion = SoftTargetCrossEntropy()
        print(f"[train] mixup on (mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, "
              f"prob={args.mixup_prob}); soft-target loss, class-weighting disabled under mixup")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    # ── Optional teacher for knowledge distillation ──────────────────────
    teacher = None
    to_teacher_input = None
    if args.teacher_checkpoint:
        tck = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        t_backbone = args.teacher_backbone or tck["backbone"]
        teacher = timm.create_model(t_backbone, pretrained=False, num_classes=num_classes)
        teacher.load_state_dict(tck["state_dict"])
        teacher.eval().to(device)
        for tp in teacher.parameters():
            tp.requires_grad_(False)
        t_size = int(tck.get("size", size))
        # The teacher may normalize its input differently from the student (the
        # full conv_small uses ImageNet mean/std; the _050 student uses
        # 0.5/0.5/0.5) AND was trained at its own resolution. The batch is already
        # normalized with the STUDENT'S stats at the STUDENT'S size, so to feed the
        # teacher we undo that, resize to the teacher's resolution, then re-apply
        # the teacher's stats — all cheap, all on-device.
        s_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s_std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        t_mean = torch.tensor([float(x) for x in tck["mean"]], device=device).view(1, 3, 1, 1)
        t_std = torch.tensor([float(x) for x in tck["std"]], device=device).view(1, 3, 1, 1)

        def to_teacher_input(xb):
            x01 = xb * s_std + s_mean  # student-normalized -> [0,1]
            if t_size != size:
                x01 = nn.functional.interpolate(x01, size=(t_size, t_size),
                                                 mode="bilinear", align_corners=False)
            return (x01 - t_mean) / t_std

        # KD already supplies soft targets; stacking mixup on top over-regularizes
        # a tiny student, so disable mixup under distillation and use the weighted
        # hard-label CE for the (1 - kd_alpha) term.
        if mixup_fn is not None:
            print("[train] distillation ON -> disabling mixup (KD already provides soft "
                  "targets; both at once over-regularizes the small student).")
            mixup_fn = None
            criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
        print(f"[train] distillation ON: teacher={t_backbone} @ {t_size}px "
              f"(kd_alpha={args.kd_alpha}, T={args.kd_temp}); the {size}px student batch is "
              f"re-normalized + resized to the teacher before its forward pass.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    # Cosine decays to a floor of min_lr/lr (not 0) so the tail keeps learning.
    lr_floor = min(1.0, max(0.0, args.min_lr / args.lr)) if args.lr > 0 else 0.0

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return lr_floor + (1.0 - lr_floor) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    use_amp = device == "cuda"
    scaler = make_scaler(use_amp)
    nsfw_i = labels.index("nsfw")

    ema = ModelEma(args.ema_decay) if args.ema else None
    if ema is not None:
        print(f"[train] EMA on (decay={args.ema_decay}); starts after warmup "
              f"({warmup_steps} steps), eval keeps the better of raw/EMA")

    def save_ckpt(net, f1):
        torch.save({
            "state_dict": net.state_dict(),
            "backbone": args.backbone,
            "labels": labels,
            "size": size,
            "mean": mean,
            "std": std,
            "interp": args.interp,
            "pad_color": list(pad_color),
            "macro_f1": f1,
        }, best_path)

    best_f1, best_path = -1.0, os.path.join(args.out_dir, "nsfw_mnv4.pt")

    # ---- robustness: write sidecars up front + push best ckpt incrementally ----
    def write_sidecars():
        # labels.json + preprocess.json never change during training, so write them
        # ONCE up front. That way a checkpoint pushed mid-run (e.g. right before a
        # job timeout) lands on the Hub as a COMPLETE, immediately-usable model.
        with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
            json.dump(labels, f)
        # interpolation + padColor make a mid-run-pushed checkpoint serve with
        # the SAME filter and pad it was trained on (export_model.py re-emits
        # identical values). The fit is always letterbox; cropSize/doCenterCrop
        # stay inert.
        preprocess = {
            "size": size, "cropSize": None, "doCenterCrop": False,
            "rescaleFactor": 1.0 / 255.0, "rescaleOffset": False,
            "doNormalize": True, "mean": mean, "std": std, "includeTop": False,
            "interpolation": args.interp,
            "padColor": list(pad_color),
        }
        with open(os.path.join(args.out_dir, "preprocess.json"), "w") as f:
            json.dump(preprocess, f, indent=2)
    write_sidecars()

    # Push out_dir to the Hub. Used (a) every time the best checkpoint improves —
    # so a timed-out/killed job still leaves the best-so-far weights on the Hub
    # instead of losing everything to ephemeral job storage — and (b) as a final
    # sync after training. Failures here are NON-FATAL: the run keeps going.
    push_api = None
    if args.push_repo:
        from huggingface_hub import HfApi
        push_api = HfApi()  # uses HF_TOKEN from the environment / job secret
        try:
            push_api.create_repo(repo_id=args.push_repo, repo_type=args.push_repo_type,
                                 private=args.push_private, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train] !! could not pre-create {args.push_repo} (will retry at push): {e}")

    def push_hub(tag):
        if push_api is None:
            return
        try:
            push_api.upload_folder(folder_path=args.out_dir, repo_id=args.push_repo,
                                  repo_type=args.push_repo_type,
                                  commit_message=f"{tag} (macroF1={best_f1:.4f})")
            print(f"[train] pushed {tag} -> {args.push_repo} (macroF1={best_f1:.4f})")
        except Exception as e:  # noqa: BLE001
            print(f"[train] !! push of '{tag}' failed (continuing; weights saved locally): {e}")
    no_improve = 0
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            # Mix BEFORE autocast so label interpolation stays in fp32. mixup_fn
            # turns y into soft targets [B, num_classes]; without it we keep the
            # hard integer labels for the weighted CE path.
            if mixup_fn is not None:
                x, target = mixup_fn(x, y)
            else:
                target = y
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx(use_amp):
                student_logits = model(x)
                if teacher is not None:
                    # Soft-target distillation (Hinton): match the teacher's
                    # temperature-softened logits, scaled by T^2 to keep the
                    # gradient magnitude comparable to the hard-label term.
                    with torch.no_grad():
                        teacher_logits = teacher(to_teacher_input(x))
                    T = args.kd_temp
                    kd = nn.functional.kl_div(
                        nn.functional.log_softmax(student_logits / T, dim=1),
                        nn.functional.softmax(teacher_logits / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)
                    loss = args.kd_alpha * kd + (1.0 - args.kd_alpha) * criterion(student_logits, target)
                else:
                    loss = criterion(student_logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            gstep += 1
            # Begin EMA exactly at the end of warmup so the average tracks only the
            # post-warmup trajectory (never the random-head init), then update it.
            if ema is not None:
                if ema.module is None and gstep >= warmup_steps:
                    ema.start(model)
                ema.update(model)
            if bi % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  e{epoch} step {bi}/{steps_per_epoch} loss={float(loss):.4f} lr={lr_now:.2e}")

        # Evaluate the raw model and (if active) the EMA model; keep the better.
        candidates = [("raw", model)]
        if ema is not None and ema.module is not None:
            candidates.append(("ema", ema.module))

        improved = False
        for tag, net in candidates:
            m = evaluate(net, val_loader, device, num_classes, nsfw_index=nsfw_i)
            per_f1 = ", ".join(f"{l}={f:.3f}" for l, f in zip(labels, m["f1"]))
            label = f"[val:{tag}]" if len(candidates) > 1 else "[val]"
            print(f"{label} epoch {epoch}: loss={m['loss']:.4f} acc={m['acc']:.4f} "
                  f"macroF1={m['macro_f1']:.4f} | F1[{per_f1}]")
            print(f"      nsfw recall={m['rec'][nsfw_i]:.3f}  nsfw precision={m['prec'][nsfw_i]:.3f}  "
                  f"nsfw AUC={m['auc']:.4f}")
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                save_ckpt(net, best_f1)
                print(f"[train] saved best ({tag}, macroF1={best_f1:.4f}) -> {best_path}")
                improved = True
                push_hub("best checkpoint")

        # Early stopping (best checkpoint already saved above; safe to stop).
        if args.patience > 0:
            no_improve = 0 if improved else no_improve + 1
            if no_improve >= args.patience:
                print(f"[train] early stop: no macro-F1 improvement for {args.patience} "
                      f"epoch(s) (best={best_f1:.4f}).")
                break

    # labels.json + preprocess.json were already written up front by
    # write_sidecars(), so any checkpoint pushed mid-run is already complete.

    print(f"\n[train] done. best macroF1={best_f1:.4f}")

    # ── Optionally push model/ to the Hub (survives ephemeral job storage) ──
    if args.push_repo:
        from huggingface_hub import HfApi
        api = HfApi()  # uses HF_TOKEN from the environment / job secret
        print(f"[train] pushing {args.out_dir}/ -> {args.push_repo} ({args.push_repo_type})")
        try:
            api.create_repo(repo_id=args.push_repo, repo_type=args.push_repo_type,
                            private=args.push_private, exist_ok=True)
            api.upload_folder(folder_path=args.out_dir, repo_id=args.push_repo,
                              repo_type=args.push_repo_type)
            print(f"[train] uploaded. fetch locally with:\n"
                  f"        hf download {args.push_repo} --repo-type {args.push_repo_type} --local-dir {args.out_dir}/")
        except Exception as e:  # noqa: BLE001
            # Don't lose the run: the trained files are already saved in out_dir.
            print("[train] !! PUSH FAILED, but training is done and saved locally in "
                  f"'{args.out_dir}/' ({', '.join(sorted(os.listdir(args.out_dir)))}).")
            print(f"[train]    error: {e}")
            print("[train]    Likely a WRITE-scope or namespace issue on the token. With a")
            print("[train]    valid write token you can upload the saved files WITHOUT retraining:")
            print(f"[train]        hf upload {args.push_repo} {args.out_dir}/ --repo-type {args.push_repo_type}")
            print("[train]    (On an ephemeral HF Job this directory is wiped when the job ends —")
            print("[train]     fix the token and re-run, or run locally where ./model/ persists.)")
            raise SystemExit(1)

    print("[train] next:  python scripts/export_model.py --variant mnv4 --calib-data <calib-dir>")
    print("[train] then:  npm run embed-model && npm run build")


if __name__ == "__main__":
    main()
