"""Train the Bigram_4 WaveNet model and export LinkedIn-ready matplotlib visuals.

End-to-end: load data -> train -> save four 1200x1200 PNGs into 03-wavenet/linkedin_assets/.
Uses only numpy, matplotlib, and Autograd (WhyyTorch).

Run:
    uv run python make_linkedin_visuals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "00-foundation"))

from Autograd import WhyyTorch as wt, cross_entropy_loss


# ---------------------------------------------------------------------------
# Hyperparameters (Karpathy makemore WaveNet — capacity + enough steps matter)
# ---------------------------------------------------------------------------
# Train set ~182k rows; batch 32 => ~5700 steps/epoch. Use several epochs, not a few hundred steps.
# n_hidden=200 is intentional: stages 2/3 flatten 2*hidden -> hidden; too small = bottleneck.

BLOCK_SIZE = 8
N_EMBD = 24
N_HIDDEN = 200
VOCAB_SIZE = 27

MAX_STEPS = 15_000
BATCH_SIZE = 32
EVAL_EVERY = 100
LR_HIGH = 0.1
LR_LOW = 0.01
SEED = 42
DATA_PATH = ROOT / "02-bigram" / "bigram.txt"


# ---------------------------------------------------------------------------
# Visualization style constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = HERE / "linkedin_assets"
CHECKPOINT_PATH = OUTPUT_DIR / "wavenet_checkpoint.npz"
HISTORY_PATH = OUTPUT_DIR / "training_history.npz"
FIG_INCHES = 6.0
DPI = 200
FONT = "DejaVu Sans"

PALETTE = {
    "primary": "#0d9488",
    "secondary": "#7c3aed",
    "accent": "#e07a5f",
    "input": "#94a3b8",
    "output": "#64748b",
    "stage1": "#0d9488",
    "stage2": "#7c3aed",
    "stage3": "#e07a5f",
    "line": "#cbd5e1",
    "text": "#1e293b",
    "muted": "#64748b",
    "train": "#2563eb",
    "val": "#e07a5f",
    "card_bg": "#f8fafc",
    "card_border": "#e2e8f0",
    "highlight_bg": "#f0fdfa",
    "highlight_border": "#99f6e4",
    "badge_bg": "#f0fdf4",
    "badge_border": "#bbf7d0",
    "badge_text": "#166534",
}


# ---------------------------------------------------------------------------
# Model (mirrors Bigram_4.ipynb)
# ---------------------------------------------------------------------------

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


def build_model(vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_hidden=N_HIDDEN):
    return Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2),
        Linear(2 * n_embd, n_hidden, bias=False),
        BatchNorm1(n_hidden),
        Tanh(),
        FlattenConsecutive(2),
        Linear(2 * n_hidden, n_hidden, bias=False),
        BatchNorm1(n_hidden),
        Tanh(),
        FlattenConsecutive(2),
        Linear(2 * n_hidden, n_hidden, bias=False),
        BatchNorm1(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size),
    ])


def count_params(model):
    return sum(p.data.size for p in model.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_vocab():
    stoi = {".": 0, **{chr(ord("a") + i): i + 1 for i in range(26)}}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def build_split(word_list, stoi, block_size=BLOCK_SIZE):
    xs, ys = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def load_data(path=ROOT / "02-bigram" / "bigram.txt", seed=SEED):
    words = open(path).read().splitlines()
    stoi, itos = load_vocab()
    np.random.seed(seed)
    shuffled = list(words)
    np.random.shuffle(shuffled)
    n1 = int(0.8 * len(shuffled))
    n2 = int(0.9 * len(shuffled))
    return (
        *build_split(shuffled[:n1], stoi),
        *build_split(shuffled[n1:n2], stoi),
        stoi,
        itos,
    )


# ---------------------------------------------------------------------------
# Training / eval / sampling
# ---------------------------------------------------------------------------

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


def train_model(model, xtr, ytr, xdev, ydev, max_steps=MAX_STEPS,
                batch_size=BATCH_SIZE, eval_every=EVAL_EVERY,
                lr_high=LR_HIGH, lr_low=LR_LOW):
    """Train with SGD; return eval_history and per-step log10 batch losses."""
    parameters = model.parameters()
    eval_history = []
    lossi = []

    for step in range(max_steps):
        ix = np.random.randint(0, xtr.shape[0], size=batch_size)
        xb, yb = xtr[ix], ytr[ix]
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)

        for p in parameters:
            p.zero_grad()
        loss.backward()

        lr = lr_high if step < max_steps // 2 else lr_low
        for p in parameters:
            p.data += -lr * p.grad

        lossi.append(float(np.log10(loss.data)))

        if step % eval_every == 0:
            tr = average_loss(model, xtr[:5000], ytr[:5000])
            va = average_loss(model, xdev, ydev)
            eval_history.append((step, tr, va))
            print(f"step {step:4d} | train {tr:.4f} | val {va:.4f}")

    return eval_history, lossi


# ---------------------------------------------------------------------------
# Checkpoint persistence (weights + eval history)
# ---------------------------------------------------------------------------

def save_eval_history(eval_history, path=HISTORY_PATH):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = np.array([e[0] for e in eval_history], dtype=np.int64)
    train = np.array([e[1] for e in eval_history], dtype=np.float64)
    val = np.array([e[2] for e in eval_history], dtype=np.float64)
    np.savez(path, steps=steps, train=train, val=val)
    return path


def load_eval_history(path=HISTORY_PATH):
    data = np.load(path)
    return list(zip(data["steps"].tolist(), data["train"].tolist(), data["val"].tolist()))


def save_checkpoint(model, eval_history, path=CHECKPOINT_PATH):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {}
    for i, p in enumerate(model.parameters()):
        payload[f"p{i}"] = np.array(p.data, copy=True)
    bn_i = 0
    for layer in model.layers:
        if isinstance(layer, BatchNorm1):
            payload[f"bn{bn_i}_mean"] = layer.running_mean.copy()
            payload[f"bn{bn_i}_var"] = layer.running_var.copy()
            bn_i += 1
    steps = np.array([e[0] for e in eval_history], dtype=np.int64)
    tr = np.array([e[1] for e in eval_history], dtype=np.float64)
    va = np.array([e[2] for e in eval_history], dtype=np.float64)
    np.savez(path, steps=steps, train=tr, val=va, **payload)
    save_eval_history(eval_history)
    print(f"saved checkpoint {path}")
    return path


def load_checkpoint(path=CHECKPOINT_PATH):
    data = np.load(path)
    np.random.seed(SEED)
    model = build_model()
    for i, p in enumerate(model.parameters()):
        p.data = data[f"p{i}"]
    bn_i = 0
    for layer in model.layers:
        if isinstance(layer, BatchNorm1):
            layer.running_mean = data[f"bn{bn_i}_mean"]
            layer.running_var = data[f"bn{bn_i}_var"]
            bn_i += 1
    eval_history = list(zip(data["steps"].tolist(), data["train"].tolist(), data["val"].tolist()))
    return model, eval_history


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _new_figure():
    fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig


def _save(fig, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, facecolor="white", edgecolor="none",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return path


def _canvas_ax(fig, margins=(0.06, 0.06, 0.88, 0.88)):
    ax = fig.add_axes(margins)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def _draw_header(ax, title, subtitle=None, y_title=0.94, y_subtitle=0.905, title_fs=17):
    ax.text(0.5, y_title, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=PALETTE["text"], family=FONT)
    if subtitle:
        ax.text(0.5, y_subtitle, subtitle, ha="center", va="center",
                fontsize=10, color=PALETTE["muted"], family=FONT)


def _rounded_box(ax, cx, cy, w, h, facecolor, label, sublabel="",
                 label_fs=9.5, sublabel_fs=7.5, text_color="white",
                 sublabel_color="#f1f5f9", edgecolor="white", linewidth=1.2):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=3,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(cx, cy + 0.011, label, ha="center", va="center",
                fontsize=label_fs, fontweight="bold", color=text_color,
                family=FONT, zorder=4)
        ax.text(cx, cy - 0.013, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color=sublabel_color, family=FONT, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=label_fs, fontweight="bold", color=text_color,
                family=FONT, zorder=4)
    return patch


def _card_box(ax, x, y, w, h, facecolor=None, edgecolor=None, linewidth=1.0):
    facecolor = facecolor or PALETTE["card_bg"]
    edgecolor = edgecolor or PALETTE["card_border"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _arrow_down(ax, x, y_top, y_bottom, color=None, lw=1.4):
    color = color or PALETTE["line"]
    ax.add_patch(FancyArrowPatch(
        (x, y_top), (x, y_bottom),
        arrowstyle="-|>", mutation_scale=10,
        color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=1,
    ))


def _arrow_right(ax, x_left, x_right, y, color=None, lw=1.4):
    color = color or PALETTE["line"]
    ax.add_patch(FancyArrowPatch(
        (x_left, y), (x_right, y),
        arrowstyle="-|>", mutation_scale=10,
        color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=1,
    ))


def _connect(ax, x1, y1, x2, y2, color=None, lw=0.9):
    color = color or PALETTE["line"]
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-",
        color=color, lw=lw, alpha=0.85, zorder=1,
    ))


def _nn_node(ax, cx, cy, r, color, label="", fs=6.5):
    circle = plt.Circle((cx, cy), r, facecolor=color, edgecolor="white",
                        linewidth=1.0, zorder=3)
    ax.add_patch(circle)
    if label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", family=FONT, zorder=4)


def _draw_bigram_nn(ax, cx, y_top, y_bottom):
    """Bigram as a 27×27 lookup: one input → weight matrix → 27 outputs."""
    r_in, r_out = 0.028, 0.014
    y_in, y_mat, y_out = y_top - 0.04, (y_top + y_bottom) / 2 + 0.02, y_bottom + 0.06

    _nn_node(ax, cx, y_in, r_in, PALETTE["input"], "c", fs=8)
    ax.text(cx, y_in + 0.055, "Previous\nCharacter", ha="center", va="bottom",
            fontsize=7.5, color=PALETTE["text"], family=FONT, fontweight="bold")

    mat_w, mat_h = 0.22, 0.16
    _rounded_box(ax, cx, y_mat, mat_w, mat_h, PALETTE["accent"],
                 "27 × 27", "Count Matrix", label_fs=9, sublabel_fs=7)
    _arrow_down(ax, cx, y_in - r_in, y_mat + mat_h / 2)

    out_xs = np.linspace(cx - 0.11, cx + 0.11, 7)
    for i, ox in enumerate(out_xs):
        lbl = "a" if i == 3 else ""
        _nn_node(ax, ox, y_out, r_out, PALETTE["output"], lbl, fs=6)
    ax.plot([out_xs[0] - r_out, out_xs[-1] + r_out], [y_out, y_out],
            color=PALETTE["muted"], lw=0.8, zorder=2)
    ax.text(out_xs[-1] + 0.03, y_out, "…27", ha="left", va="center",
            fontsize=7, color=PALETTE["muted"], family=FONT)
    _arrow_down(ax, cx, y_mat - mat_h / 2, y_out + r_out + 0.01)

    ax.text(cx, y_out - 0.055, "Next Character", ha="center", va="top",
            fontsize=7.5, color=PALETTE["text"], family=FONT, fontweight="bold")


def _draw_wavenet_nn(ax, cx, y_top, y_bottom):
    """WaveNet tree: 8 inputs merge through 3 stages to one prediction."""
    box_h = 0.038
    leaf_y = y_bottom + 0.08
    leaf_xs = np.linspace(cx - 0.20, cx + 0.20, BLOCK_SIZE)
    leaf_r = 0.016

    for i, lx in enumerate(leaf_xs):
        _nn_node(ax, lx, leaf_y, leaf_r, PALETTE["input"], f"c{i + 1}", fs=5.5)

    stages = [
        (4, (y_bottom + 0.22), PALETTE["stage1"], 0.10),
        (2, (y_bottom + 0.38), PALETTE["stage2"], 0.14),
        (1, (y_bottom + 0.54), PALETTE["stage3"], 0.28),
    ]
    prev_centers = list(leaf_xs)
    prev_y = leaf_y + leaf_r

    for n_nodes, y_stage, color, stage_w in stages:
        if n_nodes == 1:
            centers = [cx]
        else:
            span = 0.40
            x0 = cx - span / 2
            centers = [x0 + span * (i + 0.5) / n_nodes for i in range(n_nodes)]

        group_size = len(prev_centers) // n_nodes
        for i, scx in enumerate(centers):
            _rounded_box(ax, scx, y_stage, stage_w, box_h, color,
                         f"×{n_nodes}", f"{N_HIDDEN}-d", label_fs=7.5, sublabel_fs=6)
            group = prev_centers[i * group_size:(i + 1) * group_size]
            for px in group:
                _connect(ax, px, prev_y, scx, y_stage - box_h / 2)

        prev_centers = centers
        prev_y = y_stage + box_h / 2

    y_pred = y_top - 0.06
    _rounded_box(ax, cx, y_pred, 0.18, box_h, PALETTE["accent"], "Prediction", "27 Classes",
                 label_fs=8)
    _connect(ax, prev_centers[0], prev_y, cx, y_pred - box_h / 2)


def _metric_card(ax, x, y, w, h, label, value):
    _card_box(ax, x, y, w, h)
    ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center",
            fontsize=8.5, color=PALETTE["muted"], family=FONT)
    ax.text(x + w / 2, y + h * 0.32, value, ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["text"], family=FONT)


def _style_axes(ax, grid=False):
    ax.set_facecolor("white")
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["card_border"])
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, color=PALETTE["line"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Building PyTorch from Scratch
# ---------------------------------------------------------------------------

def plot_building_pytorch(filename: str = "01_building_pytorch.png") -> Path:
    fig = _new_figure()
    ax = _canvas_ax(fig, margins=(0.04, 0.08, 0.92, 0.84))
    _draw_header(ax, "Building a WaveNet-Inspired Neural Network",
                 y_title=0.94, title_fs=15)

    cy = 0.48
    s1, s2, s3 = PALETTE["stage1"], PALETTE["stage2"], PALETTE["stage3"]

    def single_box(cx, label, sublabel, color, w=0.10, h=0.10):
        _rounded_box(ax, cx, cy, w, h, color, label, sublabel,
                     label_fs=8.5, sublabel_fs=7)

    def stage_column(cx, color, stage_num, w=0.13):
        layers = ["Flatten(2)", "Linear", "BatchNorm", "Tanh"]
        layer_h, gap = 0.048, 0.010
        total_h = 4 * layer_h + 3 * gap
        y_top = cy + total_h / 2
        patch = FancyBboxPatch(
            (cx - w / 2 - 0.008, y_top - total_h - 0.018),
            w + 0.016, total_h + 0.036,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color, edgecolor=color, alpha=0.12, linewidth=0, zorder=1,
        )
        ax.add_patch(patch)
        ax.text(cx, y_top + 0.028, f"Stage {stage_num}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color, family=FONT)
        y = y_top - layer_h / 2
        prev_bottom = None
        for layer in layers:
            _rounded_box(ax, cx, y, w, layer_h, color, layer, "",
                         label_fs=7.5, text_color="white")
            if prev_bottom is not None:
                _arrow_down(ax, cx, prev_bottom, y + layer_h / 2, color=PALETTE["line"], lw=1.0)
            prev_bottom = y - layer_h / 2
            y -= layer_h + gap
        return cx, w

    # Left → right pipeline
    cols = [
        (0.07, "Input", "Characters", PALETTE["input"], "single"),
        (0.20, "Embedding", f"{VOCAB_SIZE}→{N_EMBD}", s1, "single"),
        (0.37, None, None, s1, "stage1"),
        (0.57, None, None, s2, "stage2"),
        (0.77, None, None, s3, "stage3"),
        (0.91, "Logits", f"{VOCAB_SIZE} cls", PALETTE["output"], "single"),
    ]

    prev_right = None
    for item in cols:
        cx = item[0]
        if item[4] == "single":
            single_box(cx, item[1], item[2], item[3])
            box_right = cx + 0.05
        else:
            _, w = stage_column(cx, item[3], int(item[4][-1]))
            box_right = cx + w / 2
        if prev_right is not None:
            _arrow_right(ax, prev_right + 0.01, cx - (0.065 if item[4] != "single" else 0.05), cy)
        prev_right = box_right

    path = _save(fig, filename)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Bigram vs WaveNet
# ---------------------------------------------------------------------------

def plot_bigram_vs_wavenet(filename: str = "02_bigram_vs_wavenet.png") -> Path:
    fig = _new_figure()
    ax = _canvas_ax(fig)
    _draw_header(ax, "From Bigram to WaveNet")

    left_cx, right_cx = 0.27, 0.73

    ax.text(left_cx, 0.84, "BIGRAM", ha="center", va="center",
            fontsize=11, fontweight="bold", color=PALETTE["text"], family=FONT)
    ax.text(right_cx, 0.84, "WHYYTORCH WAVENET", ha="center", va="center",
            fontsize=11, fontweight="bold", color=PALETTE["text"], family=FONT)

    _draw_bigram_nn(ax, left_cx, y_top=0.76, y_bottom=0.30)
    _draw_wavenet_nn(ax, right_cx, y_top=0.76, y_bottom=0.22)

    ax.text(left_cx, 0.20, "Uses only one\nprevious character",
            ha="center", va="center", fontsize=8, color=PALETTE["muted"], family=FONT)
    ax.text(right_cx, 0.20, "Learns richer representations\nfrom a larger context",
            ha="center", va="center", fontsize=8, color=PALETTE["muted"], family=FONT)

    ax.plot([0.5, 0.5], [0.14, 0.88], color=PALETTE["line"], lw=1.0, zorder=0)

    _card_box(ax, 0.08, 0.05, 0.84, 0.08,
              facecolor=PALETTE["highlight_bg"], edgecolor=PALETTE["highlight_border"],
              linewidth=1.2)
    ax.text(0.5, 0.08, "WaveNet learns hierarchical representations.",
            ha="center", va="center", fontsize=10, color=PALETTE["primary"],
            family=FONT, fontweight="bold")

    path = _save(fig, filename)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Training results dashboard
# ---------------------------------------------------------------------------

def plot_training_results(eval_history, n_params, max_steps=MAX_STEPS,
                          filename: str = "03_training_results.png") -> Path:
    steps = [e[0] for e in eval_history]
    train = [e[1] for e in eval_history]
    val = [e[2] for e in eval_history]
    final_train = train[-1]
    final_val = val[-1]

    fig = _new_figure()
    ax = _canvas_ax(fig, margins=(0.06, 0.06, 0.88, 0.88))
    _draw_header(ax, "Training Results", y_title=0.955, y_subtitle=0.92)

    # Metric cards
    card_w, card_h, card_gap = 0.19, 0.10, 0.025
    card_y = 0.80
    x0 = 0.5 - (4 * card_w + 3 * card_gap) / 2
    metrics = [
        ("Parameters", f"{n_params:,}"),
        ("Training Steps", f"{max_steps:,}"),
        ("Train Loss", f"{final_train:.4f}"),
        ("Validation Loss", f"{final_val:.4f}"),
    ]
    for i, (label, value) in enumerate(metrics):
        _metric_card(ax, x0 + i * (card_w + card_gap), card_y, card_w, card_h,
                     label, value)

    # Loss curve
    ax_plot = fig.add_axes([0.12, 0.16, 0.76, 0.54])
    _style_axes(ax_plot, grid=True)
    ax_plot.plot(steps, train, color=PALETTE["train"], lw=2.2, label="Training Loss")
    ax_plot.plot(steps, val, color=PALETTE["val"], lw=2.2, label="Validation Loss")

    ax_plot.set_xlabel("Training Step", fontsize=9, color=PALETTE["text"], family=FONT)
    ax_plot.set_ylabel("Cross-Entropy Loss", fontsize=9, color=PALETTE["text"], family=FONT)
    ax_plot.legend(frameon=True, framealpha=0.95, edgecolor=PALETTE["card_border"],
                   fontsize=8, loc="upper right")

    ymin = min(min(train), min(val))
    ymax = max(max(train), max(val))
    margin = max((ymax - ymin) * 0.18, 0.05)
    ax_plot.set_ylim(ymin - margin, ymax + margin)

    ax.text(0.5, 0.115,
            f"Final Validation Loss:  {final_val:.4f}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=PALETTE["val"], family=FONT)

    path = _save(fig, filename)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Receptive field growth
# ---------------------------------------------------------------------------

def plot_receptive_field_growth(filename: str = "04_receptive_field_growth.png") -> Path:
    fig = _new_figure()
    ax = _canvas_ax(fig)
    _draw_header(ax, "How WaveNet Gradually Expands Context",
                 'Example: predicting the next letter in "emma"',
                 y_title=0.96, y_subtitle=0.925)

    cx = 0.5
    # 8-char context: padding + "emma" (dot = start token)
    chars = ["·", "·", "·", "·", "e", "m", "m", "a"]

    def join_chars(indices):
        return " ".join(chars[i] for i in indices)

    def draw_char_row(y, labels, box_w, color, group_gap=0.014):
        n = len(labels)
        total_gap = group_gap * max(n - 1, 0)
        span = min(0.82, n * box_w + total_gap)
        x_start = cx - span / 2
        for i, label in enumerate(labels):
            x = x_start + i * (box_w + group_gap) + box_w / 2
            fs = 9.0 if len(label) <= 2 else 7.5
            _rounded_box(ax, x, y, box_w, 0.038, color, label, "",
                         label_fs=fs, text_color="white")
        return y

    row_ys = [0.80, 0.64, 0.48, 0.32, 0.16]
    prev_bottom = None

    ax.text(cx, row_ys[0] + 0.038, "Input Context", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=PALETTE["text"], family=FONT)
    draw_char_row(row_ys[0], chars, 0.072, PALETTE["input"])
    prev_bottom = row_ys[0] - 0.019

    stages = [
        ([join_chars([0, 1]), join_chars([2, 3]), join_chars([4, 5]), join_chars([6, 7])],
         2, 0.12, PALETTE["stage1"]),
        ([join_chars([0, 1, 2, 3]), join_chars([4, 5, 6, 7])],
         4, 0.22, PALETTE["stage2"]),
        ([join_chars(range(8))],
         8, 0.66, PALETTE["stage3"]),
    ]

    for stage_idx, (groups, rf, box_w, color) in enumerate(stages):
        y = row_ys[stage_idx + 1]
        _arrow_down(ax, cx, prev_bottom, y + 0.022)
        ax.text(cx, y + 0.048, f"Stage {stage_idx + 1}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=PALETTE["text"], family=FONT)
        draw_char_row(y, groups, box_w, color, group_gap=0.016)
        ax.text(cx, y - 0.042, f"Receptive Field = {rf}", ha="center", va="center",
                fontsize=8.5, color=PALETTE["muted"], family=FONT, fontstyle="italic")
        prev_bottom = y - 0.052

    y_pred = row_ys[4]
    _arrow_down(ax, cx, prev_bottom, y_pred + 0.022)
    _rounded_box(ax, cx, y_pred, 0.30, 0.044, PALETTE["accent"],
                 "Prediction", "next →  ·")

    path = _save(fig, filename)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all(eval_history, n_params, max_steps=MAX_STEPS):
    save_eval_history(eval_history)
    paths = [
        plot_building_pytorch(),
        plot_bigram_vs_wavenet(),
        plot_training_results(eval_history, n_params, max_steps=max_steps),
        plot_receptive_field_growth(),
    ]
    print(f"\nAll figures saved to {OUTPUT_DIR.resolve()}/")
    return paths


def run_pipeline(data_path=DATA_PATH, max_steps=MAX_STEPS, batch_size=BATCH_SIZE,
                 eval_every=EVAL_EVERY, seed=SEED, lr_high=LR_HIGH, lr_low=LR_LOW,
                 retrain=False):
    """Full train -> visualize pipeline. Skips training when checkpoint exists."""
    n_params = count_params(build_model())

    if CHECKPOINT_PATH.exists() and not retrain:
        print("Loading saved checkpoint (skipping training)...")
        model, eval_history = load_checkpoint()
        n_params = count_params(model)
    elif HISTORY_PATH.exists() and not retrain:
        print("Loading saved training history (skipping training)...")
        eval_history = load_eval_history()
        model = None
    else:
        print("Loading data...")
        xtr, ytr, xdev, ydev, stoi, itos = load_data(data_path, seed=seed)
        print(f"  train: {xtr.shape}, dev: {xdev.shape}")

        np.random.seed(seed)
        model = build_model()
        n_params = count_params(model)
        print(f"  parameters: {n_params:,}")

        print(f"\nTraining ({max_steps} steps)...")
        eval_history, lossi = train_model(
            model, xtr, ytr, xdev, ydev,
            max_steps=max_steps, batch_size=batch_size, eval_every=eval_every,
            lr_high=lr_high, lr_low=lr_low,
        )

        final_train = average_loss(model, xtr[:5000], ytr[:5000])
        final_val = average_loss(model, xdev, ydev)
        print(f"\nFinal eval | train {final_train:.4f} | val {final_val:.4f}")

        save_checkpoint(model, eval_history)

    print("\nGenerating LinkedIn assets...")
    paths = generate_all(eval_history, n_params, max_steps=max_steps)
    return {
        "model": model,
        "eval_history": eval_history,
        "paths": paths,
        "n_params": n_params,
    }


if __name__ == "__main__":
    # --- edit these ---
    data_path = DATA_PATH
    max_steps = MAX_STEPS
    batch_size = BATCH_SIZE
    eval_every = EVAL_EVERY
    seed = SEED
    lr_high = LR_HIGH
    lr_low = LR_LOW
    retrain = False  # set True to train from scratch and overwrite checkpoint

    run_pipeline(
        data_path=data_path,
        max_steps=max_steps,
        batch_size=batch_size,
        eval_every=eval_every,
        seed=seed,
        lr_high=lr_high,
        lr_low=lr_low,
        retrain=retrain,
    )
