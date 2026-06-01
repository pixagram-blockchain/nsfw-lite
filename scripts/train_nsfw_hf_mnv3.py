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
train_nsfw_hf_mnv3.py — same trainer as train_nsfw.py, but sourcing data from the
Hugging Face `datasets` library instead of a local torchvision ImageFolder.

It is identical in every way that matters to the rest of the pipeline: same
MobileNetV3-small-050 backbone, the same RandomResizedCrop (train) /
plain-square-resize (val) parity transforms, the same class-weighting, loop,
and the SAME outputs — model/nsfw_mnv3.pt, model/labels.json, model/preprocess.json
— so scripts/export_model.py and the embed/build steps work unchanged.

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
    hf jobs uv run --flavor a100-large -s HF_TOKEN scripts/train_nsfw_hf_mnv3.py \
        -- --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
           --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata \
           --push-repo <you>/nsfw-lite-mnv3 --epochs 12 \
           --batch-size 256 --num-workers 16
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
      hf download <you>/nsfw-lite-mnv3 --repo-type model --local-dir model/
      python scripts/export_model.py --calib-data <calib-dir>
      npm run embed-model && npm run build

RUN IT LOCALLY (or in a GPU Space / notebook) the ordinary way:
    pip install "timm>=1.0.0" torch torchvision "datasets>=2.14.0" pillow numpy huggingface_hub
    python scripts/train_nsfw_hf_mnv3.py \
        --nsfw-dataset wallstoneai/civitai-top-nsfw-images-with-metadata \
        --sfw-dataset  wallstoneai/civitai-top-sfw-images-with-metadata
    python scripts/train_nsfw_hf_mnv3.py --data-dir ./data   # local imagefolder

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

DEFAULT_BACKBONE = "mobilenetv3_small_050.lamb_in1k"


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
                   help="timm model id (default: mobilenetv3_small_050.lamb_in1k)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.1, help="used only if --val-split not given")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="model/mnv3")

    # Augmentation / regularization knobs.
    p.add_argument("--rrc-min", type=float, default=0.8,
                   help="RandomResizedCrop lower scale bound (default 0.8). Higher = gentler "
                        "cropping; for NSFW, avoids cropping the explicit region out of frame "
                        "and stays closer to the full-frame resize used at serve time.")
    p.add_argument("--ema", action="store_true", default=False,
                   help="track an exponential moving average of the weights; eval both raw and "
                        "EMA each epoch and keep whichever scores higher (pure upside).")
    p.add_argument("--ema-decay", type=float, default=0.998,
                   help="EMA decay (default 0.998). Lower for short runs: the averaging window is "
                        "~1/(1-decay) steps, so 0.998≈500 steps. EMA starts after warmup.")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after this many epochs without a macro-F1 improvement "
                        "(0 = disabled, train all --epochs). Best checkpoint is always kept.")

    # Optional: push the trained model/ folder to the Hub (essential on HF Jobs,
    # whose local storage is wiped when the job finishes).
    p.add_argument("--push-repo", default=None,
                   help="repo id to upload model/ to after training (e.g. you/nsfw-lite-mnv3)")
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


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(ce(logits, y)) * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1).cpu().numpy()
        for t, pr in zip(y.cpu().numpy(), pred):
            cm[t, pr] += 1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    acc = tp.sum() / max(1, cm.sum())
    return {"loss": loss_sum / max(1, n), "acc": float(acc),
            "prec": prec, "rec": rec, "f1": f1, "macro_f1": float(f1.mean()), "cm": cm}


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
    size = int(dcfg["input_size"][-1])
    mean = [float(m) for m in dcfg["mean"]]
    std = [float(s) for s in dcfg["std"]]
    del probe
    print(f"[train] input size={size}  mean={mean}  std={std}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(args.rrc_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),  # plain resize == the browser's canvas resize
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
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

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
            "macro_f1": f1,
        }, best_path)

    best_f1, best_path = -1.0, os.path.join(args.out_dir, "nsfw_mnv3.pt")
    no_improve = 0
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx(use_amp):
                loss = criterion(model(x), y)
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
            m = evaluate(net, val_loader, device, num_classes)
            per_f1 = ", ".join(f"{l}={f:.3f}" for l, f in zip(labels, m["f1"]))
            label = f"[val:{tag}]" if len(candidates) > 1 else "[val]"
            print(f"{label} epoch {epoch}: loss={m['loss']:.4f} acc={m['acc']:.4f} "
                  f"macroF1={m['macro_f1']:.4f} | F1[{per_f1}]")
            print(f"      nsfw recall={m['rec'][nsfw_i]:.3f}  nsfw precision={m['prec'][nsfw_i]:.3f}")
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                save_ckpt(net, best_f1)
                print(f"[train] saved best ({tag}, macroF1={best_f1:.4f}) -> {best_path}")
                improved = True

        # Early stopping (best checkpoint already saved above; safe to stop).
        if args.patience > 0:
            no_improve = 0 if improved else no_improve + 1
            if no_improve >= args.patience:
                print(f"[train] early stop: no macro-F1 improvement for {args.patience} "
                      f"epoch(s) (best={best_f1:.4f}).")
                break

    # Emit labels + preprocess for the export/JS side (same as train_nsfw.py).
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    preprocess = {
        "size": size, "cropSize": None, "doCenterCrop": False,
        "rescaleFactor": 1.0 / 255.0, "rescaleOffset": False,
        "doNormalize": True, "mean": mean, "std": std, "includeTop": False,
    }
    with open(os.path.join(args.out_dir, "preprocess.json"), "w") as f:
        json.dump(preprocess, f, indent=2)

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

    print("[train] next:  python scripts/export_model.py --checkpoint model/mnv3/nsfw_mnv3.pt --calib-data <calib-dir> --out-dir model/mnv3")
    print("[train] then:  npm run embed-model && npm run build")


if __name__ == "__main__":
    main()
