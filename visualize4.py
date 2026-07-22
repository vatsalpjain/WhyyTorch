"""WaveNet-style architecture visualization for Bigram_4.ipynb.

Renders three LinkedIn-friendly panels (no generated name samples):
  1. Pyramid merge diagram — 8 -> 4 -> 2 -> 1 context compression.
  2. Stage pipeline — operations, shapes, and parameter counts.
  3. Training loss — train vs validation cross-entropy over steps.

Run:
    uv run visualize4.py
    uv run visualize4.py --steps 500   # quick preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from Autograd import WhyyTorch as wt, cross_entropy_loss


# ---------------------------------------------------------------------------
# Model + data (mirrors Bigram_4.ipynb)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 8
N_EMBD = 48
N_HIDDEN = 400
VOCAB_SIZE = 27


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = wt(np.random.randn(fan_in, fan_out) / fan_in**0.5)
        self.bias = wt(np.random.randn(fan_out)) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = wt(np.ones(dim))
        self.beta = wt(np.zeros(dim))
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var = np.ones(dim, dtype=np.float32)

    def __call__(self, x):
        if self.training:
            if len(x.shape) == 3:
                xmean = x.mean((0, 1), keepdims=True)
                xvar = x.var((0, 1), keepdims=True)
            else:
                xmean = x.mean(0, keepdims=True)
                xvar = x.var(0, keepdims=True)
        else:
            if len(x.shape) == 3:
                xmean = wt(self.running_mean.reshape(1, 1, -1), requires_grad=False)
                xvar = wt(self.running_var.reshape(1, 1, -1), requires_grad=False)
            else:
                xmean = wt(self.running_mean.reshape(1, -1), requires_grad=False)
                xvar = wt(self.running_var.reshape(1, -1), requires_grad=False)
        xhat = (x - xmean) / (xvar + self.eps).sqrt()
        self.out = self.gamma * xhat + self.beta
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.data.reshape(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.data.reshape(-1)
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = x.tanh()
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = wt(np.random.randn(num_embeddings, embedding_dim))

    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        self.out = x.reshape(x.shape[0], x.shape[1] // self.n, self.n * x.shape[2])
        if self.out.shape[1] == 1:
            self.out = self.out.reshape(self.out.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def wavenet_param_count():
    """Analytic parameter count matching Bigram_4.ipynb architecture."""
    return (
        VOCAB_SIZE * N_EMBD
        + (2 * N_EMBD) * N_HIDDEN
        + (2 * N_HIDDEN) * N_HIDDEN * 2
        + N_HIDDEN * VOCAB_SIZE
        + 3 * 2 * N_HIDDEN
    )


def build_model():
    return Sequential([
        Embedding(VOCAB_SIZE, N_EMBD),
        FlattenConsecutive(2),
        Linear(2 * N_EMBD, N_HIDDEN, bias=False),
        BatchNorm1(N_HIDDEN),
        Tanh(),
        FlattenConsecutive(2),
        Linear(2 * N_HIDDEN, N_HIDDEN, bias=False),
        BatchNorm1(N_HIDDEN),
        Tanh(),
        FlattenConsecutive(2),
        Linear(2 * N_HIDDEN, N_HIDDEN, bias=False),
        BatchNorm1(N_HIDDEN),
        Tanh(),
        Linear(N_HIDDEN, VOCAB_SIZE),
    ])


def count_params(model):
    return sum(p.data.size for p in model.parameters())


def load_bigram_data(path="bigram.txt", seed=42):
    words = open(path).read().splitlines()
    stoi = {".": 0, **{chr(ord("a") + i): i + 1 for i in range(26)}}
    np.random.seed(seed)
    shuffled = list(words)
    np.random.shuffle(shuffled)
    n1 = int(0.8 * len(shuffled))
    n2 = int(0.9 * len(shuffled))

    def build_split(word_list):
        xs, ys = [], []
        for w in word_list:
            context = [0] * BLOCK_SIZE
            for ch in w + ".":
                ix = stoi[ch]
                xs.append(context)
                ys.append(ix)
                context = context[1:] + [ix]
        return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)

    return (
        build_split(shuffled[:n1]),
        build_split(shuffled[n1:n2]),
    )


def set_bn_training(model, training=True):
    for layer in model.layers:
        if isinstance(layer, BatchNorm1):
            layer.training = training


def average_loss(model, x, y, batch_size=256):
    set_bn_training(model, False)
    losses = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        losses.append(float(cross_entropy_loss(model(xb), yb).data))
    set_bn_training(model, True)
    return float(np.mean(losses))


def train_wavenet(model, xtr, ytr, xdev, ydev, max_steps=1000, batch_size=32, eval_every=10):
    parameters = model.parameters()
    step_losses = []
    eval_steps = []
    train_losses = []
    val_losses = []

    for step in range(max_steps):
        ix = np.random.randint(0, xtr.shape[0], size=batch_size)
        xb, yb = xtr[ix], ytr[ix]
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)

        for p in parameters:
            p.zero_grad()
        loss.backward()

        lr = 0.1 if step < 500 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        step_losses.append(float(loss.data))

        if step % eval_every == 0:
            tr = average_loss(model, xtr[:5000], ytr[:5000])
            va = average_loss(model, xdev, ydev)
            eval_steps.append(step)
            train_losses.append(tr)
            val_losses.append(va)
            print(f"step {step:4d} | train {tr:.4f} | val {va:.4f}")

    return step_losses, eval_steps, train_losses, val_losses


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COLORS = {
    "input": "#0f766e",
    "embed": "#0891b2",
    "stage1": "#2563eb",
    "stage2": "#4f46e5",
    "stage3": "#7c3aed",
    "output": "#b91c1c",
    "arrow": "#64748b",
    "label": "#1e293b",
    "muted": "#475569",
}


def _rounded_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9.5, alpha=0.92):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor="white",
        linewidth=1.4,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2 + (0.012 if sublabel else 0),
        label,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold", color="white", zorder=4,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.028,
            sublabel,
            ha="center", va="center",
            fontsize=7.5, color="#e2e8f0", zorder=4,
        )


def _merge_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=10,
        color=COLORS["arrow"],
        lw=1.2,
        alpha=0.65,
        zorder=2,
    ))


def draw_wavenet_pyramid(ax):
    """Pyramid diagram: 8 embedded chars merge to a single context vector."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Bottom row: 8 character slots (example context, not sampled names)
    n_slots = 8
    box_w, box_h = 0.09, 0.07
    gap = 0.012
    total_w = n_slots * box_w + (n_slots - 1) * gap
    x0 = (1 - total_w) / 2
    y_input = 0.08
    char_labels = ["·"] * 8  # generic padding/context slots (no name samples)
    input_xs = []

    for i in range(n_slots):
        x = x0 + i * (box_w + gap)
        input_xs.append(x + box_w / 2)
        _rounded_box(
            ax, x, y_input, box_w, box_h, COLORS["embed"],
            char_labels[i], f"24-d", fontsize=10,
        )

    ax.text(0.03, y_input + box_h / 2, "Embedding\n(27 x 24)",
            ha="left", va="center", fontsize=8.5, color=COLORS["muted"], fontweight="bold")

    # Stage rows: 4, 2, 1
    stages = [
        (4, COLORS["stage1"], "Stage 1", "RF = 2 chars", 0.28),
        (2, COLORS["stage2"], "Stage 2", "RF = 4 chars", 0.50),
        (1, COLORS["stage3"], "Stage 3", "RF = 8 chars", 0.72),
    ]

    prev_centers = input_xs
    prev_y = y_input + box_h

    for n, color, title, rf_label, y in stages:
        stage_w = n * 0.14 + max(0, n - 1) * 0.02
        sx0 = (1 - stage_w) / 2
        centers = []
        for i in range(n):
            x = sx0 + i * 0.16
            w, h = 0.14, 0.08
            _rounded_box(ax, x, y, w, h, color, title if n == 1 or i == 0 else "", "200-d", fontsize=8.5)
            cx, cy = x + w / 2, y + h / 2
            centers.append(cx)
            # Arrows from previous row into this box
            span = len(prev_centers) // n
            group = prev_centers[i * span:(i + 1) * span]
            for px in group:
                _merge_arrow(ax, px, prev_y + 0.01, cx, y - 0.01)

        ax.text(
            0.97, y + 0.04, rf_label,
            ha="right", va="center", fontsize=8.5, color=COLORS["label"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", lw=0.6),
        )
        prev_centers = centers
        prev_y = y + 0.08

    # Output logits
    y_out = 0.90
    out_w = 0.22
    _rounded_box(ax, 0.5 - out_w / 2, y_out, out_w, 0.06, COLORS["output"], "Logits", "27 classes", fontsize=9)
    _merge_arrow(ax, prev_centers[0], prev_y + 0.01, 0.5, y_out - 0.01)

    # Side annotation for one stage block
    ax.annotate(
        "Flatten(2) + Linear\n+ BatchNorm + Tanh",
        xy=(0.18, 0.50), xytext=(0.01, 0.42),
        fontsize=8.0, color=COLORS["label"],
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", lw=0.6),
    )

    total = wavenet_param_count()
    ax.set_title(
        f"WaveNet pyramid: merge neighbors, grow receptive field  (block={BLOCK_SIZE}, {total:,} params)",
        fontsize=12.5, fontweight="bold", pad=12,
    )


def draw_wavenet_pipeline(ax):
    """Horizontal pipeline with tensor shapes and operation labels."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("Input", f"{BLOCK_SIZE} token IDs", "(B, 8)", COLORS["input"]),
        ("Embed", f"27 x {N_EMBD}", "(B, 8, 24)", COLORS["embed"]),
        ("Stage 1", "Flatten(2)\nLinear 48->200\nBN + Tanh", "(B, 4, 200)", COLORS["stage1"]),
        ("Stage 2", "Flatten(2)\nLinear 400->200\nBN + Tanh", "(B, 2, 200)", COLORS["stage2"]),
        ("Stage 3", "Flatten(2)\nLinear 400->200\nBN + Tanh", "(B, 200)", COLORS["stage3"]),
        ("Output", "Linear 200->27\nsoftmax", "(B, 27)", COLORS["output"]),
    ]

    n = len(stages)
    xs = np.linspace(0.07, 0.93, n)
    box_w, box_h = 0.11, 0.42

    for i, (title, detail, shape, color) in enumerate(stages):
        x = xs[i] - box_w / 2
        y = 0.30
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=color, edgecolor="white", linewidth=1.4, alpha=0.92, zorder=3,
        )
        ax.add_patch(box)
        ax.text(xs[i], y + box_h * 0.78, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)
        ax.text(xs[i], y + box_h * 0.48, detail, ha="center", va="center",
                fontsize=7.2, color="#e2e8f0", zorder=4, linespacing=1.35)
        ax.text(xs[i], y + box_h * 0.14, shape, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.15), ec="none"))

        if i < n - 1:
            ax.annotate(
                "", xy=(xs[i + 1] - box_w / 2 - 0.005, y + box_h / 2),
                xytext=(x + box_w + 0.005, y + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["arrow"], lw=1.5),
            )

    # Receptive field bar
    rf_y = 0.82
    rf_labels = ["2", "4", "8"]
    rf_xs = [xs[2], xs[3], xs[4]]
    for x, rf in zip(rf_xs, rf_labels):
        ax.text(x, rf_y, f"sees {rf} chars", ha="center", va="bottom",
                fontsize=8.5, color=COLORS["label"], fontweight="bold")
    ax.plot([rf_xs[0], rf_xs[-1]], [rf_y - 0.02, rf_y - 0.02],
            color=COLORS["arrow"], lw=1.5, alpha=0.5)
    ax.text(0.5, rf_y + 0.06, "Receptive field doubles each stage",
            ha="center", va="bottom", fontsize=9.5, color=COLORS["muted"], fontstyle="italic")

    # Width compression bar
    widths = [8, 4, 2, 1]
    width_xs = [xs[1], xs[2], xs[3], xs[4]]
    bar_y = 0.12
    for x, w in zip(width_xs, widths):
        bar_h = 0.04 + 0.025 * w
        ax.bar(x, bar_h, width=0.08, bottom=bar_y, color=COLORS["stage1"], alpha=0.25 + 0.15 * (w / 8))
        ax.text(x, bar_y + bar_h + 0.015, str(w), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.5, bar_y - 0.04, "Sequence width shrinks: 8 -> 4 -> 2 -> 1",
            ha="center", va="top", fontsize=9.0, color=COLORS["muted"])

    ax.set_title("WaveNet pipeline: operations and tensor shapes", fontsize=12.5, fontweight="bold", pad=10)


def plot_training_curves(ax, eval_steps, train_losses, val_losses, step_losses):
    """Train/val eval cross-entropy over training steps."""
    del step_losses  # kept for API symmetry with training loop
    ax.plot(eval_steps, train_losses, color="#2563eb", lw=2.4, marker="o", ms=5,
            label="train (5k subset)")
    ax.plot(eval_steps, val_losses, color="#b91c1c", lw=2.4, marker="s", ms=5,
            label="validation")

    ax.set_title("Training: cross-entropy loss", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("eval loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    best_i = int(np.argmin(val_losses))
    ax.scatter(
        [eval_steps[best_i]], [val_losses[best_i]],
        s=90, c="#16a34a", zorder=5, edgecolors="white", linewidths=1.2,
    )
    ax.annotate(
        f"best val {val_losses[best_i]:.3f}",
        (eval_steps[best_i], val_losses[best_i]),
        textcoords="offset points", xytext=(10, -14),
        fontsize=8.5, color="#16a34a", fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WaveNet visualization for LinkedIn post")
    parser.add_argument("--steps", type=int, default=2000, help="training steps (use 500 for quick preview)")
    parser.add_argument("--eval-every", type=int, default=50, help="eval interval")
    parser.add_argument("--out", type=str, default="wavenet_visualization.png", help="output image path")
    parser.add_argument("--no-train", action="store_true", help="skip training, use synthetic loss curves")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")

    if args.no_train:
        eval_steps = list(range(0, args.steps, args.eval_every))
        train_losses = [2.25 - 0.08 * (s / args.steps) for s in eval_steps]
        val_losses = [2.22 - 0.07 * (s / args.steps) for s in eval_steps]
        step_losses = list(np.linspace(0.5, -0.3, args.steps))
    else:
        print("Loading data...")
        (xtr, ytr), (xdev, ydev) = load_bigram_data()
        print(f"  train: {xtr.shape}, dev: {xdev.shape}")

        print(f"Training WaveNet ({args.steps} steps)...")
        np.random.seed(42)
        model = build_model()
        print(f"  parameters: {count_params(model):,}")
        step_losses, eval_steps, train_losses, val_losses = train_wavenet(
            model, xtr, ytr, xdev, ydev,
            max_steps=args.steps, eval_every=args.eval_every,
        )

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.2))
    draw_wavenet_pyramid(axes[0])
    draw_wavenet_pipeline(axes[1])
    plot_training_curves(axes[2], eval_steps, train_losses, val_losses, step_losses)

    fig.suptitle(
        "WaveNet for character prediction: hierarchical context merging (Karpathy makemore)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=2.5)

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization to: {out_path.resolve()}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
