"""Train model_x: spatial-heatmap ball localizer (WhyyTorch).

Why this can work (vs flatten -> 5 regression):
  - Keeps a spatial map through the whole net
  - Predicts an 8x8 (or 16x16) heatmap, not one global (cx, cy)
  - Soft Gaussian target at the ball center — dense, local supervision

Usage:
  python train_model_x.py

Edit the CONFIG block below to change training params.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "00-foundation"))

from Autograd import WhyyTorch as wt

DATA_DIR = ROOT / "data" / "images"
LABELS_PATH = ROOT / "data" / "labels.json"
CKPT_PATH = ROOT / "model_x_weights.npz"
VIS_PATH = ROOT / "model_x_val_preview.png"

# ---------------------------------------------------------------------------
# CONFIG — edit these, then run: python train_model_x.py
# ---------------------------------------------------------------------------
EPOCHS = 6
LR = 5e-3
IMG_SIZE = 128         # must be divisible by 8 (64 -> 8x8 heatmap, 128 -> 16x16)
SEED = 41
TRAIN_FRAC = 0.8
TRAIN_LIMIT = None     # e.g. 100 to smoke-test; None = all train images
VAL_LIMIT = None       # e.g. 40 during epochs; None = full val each epoch
LOG_EVERY = 100        # print running loss every N steps; 0 = quiet
CONF_THRESH = 0.35     # peak heatmap >= this => "ball present"
HIT_THRESH = 0.20      # |dx|+|dy| below this counts as localization hit

# Heatmap target / loss — stops the "predict all zeros" cheat
SIGMA_CELLS = 1.75     # wider Gaussian blob (was ~1.0)
POS_WEIGHT = 50.0      # weight on ball cells vs background in loss
HEAD_BIAS_INIT = -1.0  # last-layer bias; sigmoid(-1)≈0.27 so not stuck at 0

# ---------------------------------------------------------------------------
# Layers (padding + vectorized im2col / pool)
# ---------------------------------------------------------------------------


def _as_wt(x, requires_grad=False):
    if isinstance(x, wt):
        return x
    return wt(x, requires_grad=requires_grad)


def sigmoid(x):
    """Stable sigmoid with autograd."""
    x = _as_wt(x)
    z = np.clip(x.data, -20.0, 20.0)
    s = 1.0 / (1.0 + np.exp(-z))
    out = wt(
        s.astype(np.float32),
        requires_grad=x.requires_grad,
        _op="sigmoid",
        children=(x,),
    )

    def backward():
        x._accumulate_grad(out.grad * s * (1.0 - s))

    out._backward = backward
    return out


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.kernels = wt(
            np.random.randn(out_channels, kernel_size, kernel_size, in_channels).astype(np.float32)
            * scale
        )
        self.biases = wt(np.zeros(out_channels, dtype=np.float32))

    def forward(self, x):
        x = _as_wt(x)
        xd = x.data
        pad = self.padding
        if pad:
            xd_pad = np.pad(xd, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        else:
            xd_pad = xd

        kd = self.kernels.data
        bd = self.biases.data
        ks = self.kernel_size
        h_out = xd_pad.shape[0] - ks + 1
        w_out = xd_pad.shape[1] - ks + 1

        # im2col via as_strided
        shape = (h_out, w_out, ks, ks, xd_pad.shape[2])
        strides = (
            xd_pad.strides[0],
            xd_pad.strides[1],
            xd_pad.strides[0],
            xd_pad.strides[1],
            xd_pad.strides[2],
        )
        patches = as_strided(xd_pad, shape=shape, strides=strides).reshape(h_out * w_out, -1).copy()
        kernels_flat = kd.reshape(self.out_channels, -1)
        out_2d = patches @ kernels_flat.T + bd
        out_data = out_2d.reshape(h_out, w_out, self.out_channels)

        self._patches = patches
        self._kernels_flat = kernels_flat
        self._h_out = h_out
        self._w_out = w_out
        self._xd_pad_shape = xd_pad.shape

        out = wt(
            out_data,
            requires_grad=x.requires_grad or self.kernels.requires_grad or self.biases.requires_grad,
            _op="conv2d",
            children=(x, self.kernels, self.biases),
        )

        def backward():
            g = out.grad
            g2d = g.reshape(h_out * w_out, self.out_channels)
            grad_kernels = (g2d.T @ self._patches).reshape(kd.shape)
            grad_biases = g2d.sum(axis=0)
            grad_patches = g2d @ self._kernels_flat
            grad_pad = np.zeros(self._xd_pad_shape, dtype=np.float32)
            for i in range(h_out):
                for j in range(w_out):
                    grad_pad[i : i + ks, j : j + ks] += grad_patches[i * w_out + j].reshape(
                        ks, ks, xd_pad.shape[2]
                    )
            grad_input = grad_pad[pad:-pad, pad:-pad] if pad else grad_pad
            x._accumulate_grad(grad_input)
            self.kernels._accumulate_grad(grad_kernels)
            self.biases._accumulate_grad(grad_biases)

        out._backward = backward
        return out

    def parameters(self):
        return [self.kernels, self.biases]


class MaxPool2d:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def forward(self, x):
        x = _as_wt(x)
        xd = x.data
        ks = self.kernel_size
        out_h = xd.shape[0] // ks
        out_w = xd.shape[1] // ks
        c = xd.shape[2]
        cropped = xd[: out_h * ks, : out_w * ks, :]
        windows = cropped.reshape(out_h, ks, out_w, ks, c).transpose(0, 2, 1, 3, 4)
        flat = windows.reshape(out_h, out_w, ks * ks, c)
        idx = flat.argmax(axis=2)
        out_data = flat.max(axis=2)

        out = wt(
            out_data.astype(np.float32),
            requires_grad=x.requires_grad,
            _op="maxpool2d",
            children=(x,),
        )

        def backward():
            grad_input = np.zeros_like(xd, dtype=np.float32)
            g = out.grad
            for i in range(out_h):
                for j in range(out_w):
                    for ch in range(c):
                        flat_idx = int(idx[i, j, ch])
                        wi, wj = divmod(flat_idx, ks)
                        grad_input[i * ks + wi, j * ks + wj, ch] += g[i, j, ch]
            x._accumulate_grad(grad_input)

        out._backward = backward
        return out

    def parameters(self):
        return []


class ReLU:
    def forward(self, x):
        return _as_wt(x).relu()

    def parameters(self):
        return []


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        x = _as_wt(x, requires_grad=False)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def step(self, lr):
        for p in self.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad


# ---------------------------------------------------------------------------
# model_x — spatial heatmap (NO giant flatten -> 5)
# ---------------------------------------------------------------------------


def build_model_x(img_size: int = 64) -> Sequential:
    """
    64x64 -> pool -> 32 -> pool -> 16 -> pool -> 8x8x32 -> 1x1 -> 8x8 logits
    128x128 similarly ends at 16x16.
    """
    assert img_size % 8 == 0, "img_size must be divisible by 8"
    head = Conv2d(32, 1, 1, padding=0)
    # Bias toward mild activation so the net must learn peaks, not collapse to 0
    head.biases.data[:] = np.float32(HEAD_BIAS_INIT)
    model = Sequential(
        Conv2d(3, 8, 3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(8, 16, 3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(16, 32, 3, padding=1),
        ReLU(),
        MaxPool2d(2),
        head,  # spatial head: heatmap logits
    )
    return model


def grid_size(img_size: int) -> int:
    return img_size // 8


def make_heatmap_target(cx, cy, has_ball: bool, grid: int, sigma_cells: float = SIGMA_CELLS) -> np.ndarray:
    """Soft Gaussian blob on the grid (peak=1). Zeros if no ball."""
    if not has_ball:
        return np.zeros((grid, grid), dtype=np.float32)
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)
    x = (xx + 0.5) / grid
    y = (yy + 0.5) / grid
    sigma = sigma_cells / grid
    d2 = (x - float(cx)) ** 2 + (y - float(cy)) ** 2
    g = np.exp(-d2 / (2.0 * sigma * sigma))
    g = (g / (g.max() + 1e-8)).astype(np.float32)
    return g


def heatmap_loss(logits, target: np.ndarray):
    """Weighted MSE on sigmoid(heatmap).

    Background cells weight=1; ball blob cells weight up to 1+POS_WEIGHT.
    Without this, predicting all ~0 minimizes loss because the target is sparse.
    """
    pred = sigmoid(logits)
    if pred.data.ndim == 3:
        pred_2d = pred.reshape(pred.data.shape[0], pred.data.shape[1])
    else:
        pred_2d = pred

    t = wt(target.astype(np.float32), requires_grad=False)
    w = wt((1.0 + POS_WEIGHT * target).astype(np.float32), requires_grad=False)
    err2 = (pred_2d - t) ** 2
    loss = (err2 * w).sum() / w.sum()
    return loss, pred_2d


def decode_heatmap(prob: np.ndarray):
    """Argmax cell -> normalized (cx, cy), confidence = peak."""
    if prob.ndim == 3:
        prob = prob[..., 0]
    grid = prob.shape[0]
    idx = int(np.argmax(prob))
    gy, gx = divmod(idx, grid)
    cx = (gx + 0.5) / grid
    cy = (gy + 0.5) / grid
    return cx, cy, float(prob[gy, gx])


# ---------------------------------------------------------------------------
# Data ingestion (source-grouped split — no aug leakage)
# ---------------------------------------------------------------------------


def load_dataset(img_size: int):
    labels = json.loads(LABELS_PATH.read_text())
    X, Y_heat, meta = [], [], []
    grid = grid_size(img_size)

    for item in labels:
        path = DATA_DIR / item["image"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0

        has_ball = bool(item.get("has_ball"))
        cx = item.get("cx") or 0.0
        cy = item.get("cy") or 0.0
        heat = make_heatmap_target(cx, cy, has_ball, grid)

        group = item.get("source", item["image"])
        X.append(img)
        Y_heat.append(heat)
        meta.append(
            {
                "image": item["image"],
                "group": group,
                "has_ball": has_ball,
                "cx": float(cx) if has_ball else None,
                "cy": float(cy) if has_ball else None,
            }
        )

    return np.array(X), np.array(Y_heat), meta, grid


def source_grouped_split(X, Y, meta, seed=42, train_frac=0.8):
    groups = np.array([m["group"] for m in meta])
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_train = int(train_frac * len(perm))
    train_g = set(perm[:n_train])
    test_g = set(perm[n_train:])
    assert train_g.isdisjoint(test_g)

    train_idx = [i for i, g in enumerate(groups) if g in train_g]
    test_idx = [i for i, g in enumerate(groups) if g in test_g]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    def take(idxs):
        return X[idxs], Y[idxs], [meta[i] for i in idxs]

    return take(train_idx), take(test_idx), len(train_g), len(test_g)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def eval_split(model, X, Y, meta):
    loss_sum = 0.0
    loc_err = []
    hits = 0
    ball_n = 0
    conf_correct = 0

    for i in range(len(X)):
        logits = model(X[i])
        loss, pred = heatmap_loss(logits, Y[i])
        loss_sum += float(loss.data)
        prob = pred.data
        px, py, pconf = decode_heatmap(prob)
        m = meta[i]
        pred_ball = pconf >= CONF_THRESH
        if pred_ball == m["has_ball"]:
            conf_correct += 1
        if m["has_ball"]:
            ball_n += 1
            err = abs(px - m["cx"]) + abs(py - m["cy"])
            loc_err.append(err)
            if err < HIT_THRESH:
                hits += 1

    return {
        "loss": loss_sum / max(len(X), 1),
        "presence_acc": conf_correct / max(len(X), 1),
        "loc_l1": float(np.mean(loc_err)) if loc_err else float("nan"),
        "hit_rate": hits / max(ball_n, 1),
        "n": len(X),
        "n_ball": ball_n,
    }


def save_model(model, path=CKPT_PATH):
    arrays = {}
    for i, p in enumerate(model.parameters()):
        arrays[f"p{i}"] = p.data
    np.savez(path, **arrays)
    print(f"saved {path}")


def train():
    np.random.seed(SEED)
    print("Loading data...")
    X, Y, meta, grid = load_dataset(IMG_SIZE)
    (x_train, y_train, m_train), (x_val, y_val, m_val), n_tr_g, n_va_g = source_grouped_split(
        X, Y, meta, seed=SEED, train_frac=TRAIN_FRAC
    )
    print(
        f"images={len(X)}  train={len(x_train)} ({n_tr_g} sources)  "
        f"val={len(x_val)} ({n_va_g} sources)  img={IMG_SIZE}  grid={grid}x{grid}"
    )

    model_x = build_model_x(IMG_SIZE)
    logits = model_x(x_train[0])
    print(f"model_x out shape: {logits.shape}  (expect {(grid, grid, 1)})")
    assert logits.shape[:2] == (grid, grid)

    n_train = len(x_train) if TRAIN_LIMIT is None else min(TRAIN_LIMIT, len(x_train))
    print(
        f"Training model_x on {n_train}/{len(x_train)} train images, "
        f"lr={LR}, epochs={EPOCHS}, pos_weight={POS_WEIGHT}, sigma={SIGMA_CELLS}"
    )

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        order = np.random.permutation(n_train)
        total = 0.0
        for step, i in enumerate(order):
            model_x.zero_grad()
            logits = model_x(x_train[i])
            loss, _ = heatmap_loss(logits, y_train[i])
            loss.backward()
            model_x.step(LR)
            total += float(loss.data)
            if LOG_EVERY and (step + 1) % LOG_EVERY == 0:
                print(f"  epoch {epoch} step {step+1}/{n_train}  running_loss={total/(step+1):.4f}")

        train_loss = total / n_train
        val_n = len(x_val) if VAL_LIMIT is None else min(VAL_LIMIT, len(x_val))
        metrics = eval_split(model_x, x_val[:val_n], y_val[:val_n], m_val[:val_n])
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_loss={metrics['loss']:.4f}  "
            f"presence_acc={metrics['presence_acc']:.3f}  "
            f"hit@{HIT_THRESH:.2f}={metrics['hit_rate']:.3f}  "
            f"loc_L1={metrics['loc_l1']:.3f}"
        )

    print("\nFinal validation (full val set):")
    final = eval_split(model_x, x_val, y_val, m_val)
    print(
        f"  val_loss={final['loss']:.4f}  presence_acc={final['presence_acc']:.3f}  "
        f"hit@{HIT_THRESH:.2f}={final['hit_rate']:.3f}  loc_L1={final['loc_l1']:.3f}  "
        f"(n={final['n']}, balls={final['n_ball']})"
    )

    print("\nSample predictions (val):")
    for i in range(min(5, len(x_val))):
        logits = model_x(x_val[i])
        _, pred = heatmap_loss(logits, y_val[i])
        px, py, pconf = decode_heatmap(pred.data)
        m = m_val[i]
        gt = f"({m['cx']:.2f},{m['cy']:.2f})" if m["has_ball"] else "no-ball"
        print(
            f"  {m['image']}: pred=({px:.2f},{py:.2f}) conf={pconf:.2f}  gt={gt}  "
            f"group={m['group']}"
        )

    save_model(model_x)
    visualize_val_predictions(model_x, x_val, y_val, m_val, n=5)
    print(f"Done in {time.perf_counter() - t0:.1f}s")
    return model_x, final


def visualize_val_predictions(model, x_val, y_val, m_val, n=5):
    """Show n val images: image + GT (lime) + pred (red) + heatmap overlay."""
    n = min(n, len(x_val))
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    h, w = x_val[0].shape[:2]
    for row in range(n):
        img = x_val[row]
        target = y_val[row]
        m = m_val[row]
        logits = model(img)
        _, pred = heatmap_loss(logits, target)
        prob = pred.data
        if prob.ndim == 3:
            prob = prob[..., 0]
        px, py, pconf = decode_heatmap(prob)

        # Left: image with markers
        ax0 = axes[row, 0]
        ax0.imshow(np.clip(img, 0, 1))
        if m["has_ball"]:
            ax0.scatter(
                m["cx"] * w, m["cy"] * h,
                s=120, facecolors="none", edgecolors="lime", linewidths=2, label="GT",
            )
        ax0.scatter(
            px * w, py * h,
            s=80, c="red", marker="x", linewidths=2, label="pred",
        )
        ax0.set_title(
            f"{m['image']}  conf={pconf:.2f}\n"
            f"GT={'ball' if m['has_ball'] else 'no-ball'}  "
            f"pred=({px:.2f},{py:.2f})"
        )
        ax0.axis("off")
        ax0.legend(loc="upper right", fontsize=8)

        # Right: predicted heatmap
        ax1 = axes[row, 1]
        im = ax1.imshow(prob, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax1.set_title("pred heatmap")
        ax1.set_xticks(range(prob.shape[1]))
        ax1.set_yticks(range(prob.shape[0]))
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle("model_x validation preview (lime=GT, red=pred)", fontsize=12)
    fig.tight_layout()
    fig.savefig(VIS_PATH, dpi=140, bbox_inches="tight")
    print(f"saved validation preview -> {VIS_PATH}")
    plt.show()


if __name__ == "__main__":
    train()
