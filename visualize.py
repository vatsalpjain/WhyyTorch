"""Visualization helpers for the tiny WhyyTorch MLP.

This script mirrors the tiny-batch training run and renders:
1. A labeled MLP architecture diagram.
2. A training loss curve for 100 epochs.

Run:
	uv run .\visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Autograd import WhyyTorch, MLP, cross_entropy_loss


def make_tiny_batch(seed=7):
	"""Create a reproducible tiny batch, matching the training demo setup."""
	rng = np.random.default_rng(seed)
	x = rng.normal(0.0, 1.0, size=(8, 3)).astype(np.float32)
	y = rng.normal(0.0, 1.0, size=(8, 1)).astype(np.float32)
	return x, y


def train_model(x_np, y_np, layer_sizes, epochs=1000, lr=0.001, log_every=10):
	"""Train an MLP with the custom autograd engine and return diagnostics."""
	x = WhyyTorch(x_np, requires_grad=False)
	y = WhyyTorch(y_np, requires_grad=False)

	model = MLP(x_np.shape[1], layer_sizes)
	losses = []
	log_epochs = []
	log_losses = []

	for epoch in range(epochs):
		pred = model(x)
		loss = ((pred - y) ** 2).mean()
		losses.append(float(loss.data))

		if epoch % log_every == 0:
			print(f"Epoch {epoch + 1}/{epochs}")
			print(f"Loss: {float(loss.data):.7f}")
			log_epochs.append(epoch + 1)
			log_losses.append(float(loss.data))

		loss.backward()
		model.step(lr)
		model.zero_grad()

	return model, np.asarray(losses, dtype=np.float32), np.asarray(log_epochs), np.asarray(log_losses)


def draw_mlp(ax, layer_sizes):
	"""Draw a simple node-link diagram for an MLP architecture."""
	n_layers = len(layer_sizes)
	x_positions = np.linspace(0.08, 0.92, n_layers)

	# Compute y positions per layer so each layer is vertically centered.
	layer_nodes = []
	for size in layer_sizes:
		y = np.linspace(0.08, 0.72, size) if size > 1 else np.array([0.4])
		layer_nodes.append(y)

	# Draw edges first so nodes remain on top.
	for i in range(n_layers - 1):
		for y1 in layer_nodes[i]:
			for y2 in layer_nodes[i + 1]:
				ax.plot(
					[x_positions[i], x_positions[i + 1]],
					[y1, y2],
					color="#94a3b8",
					lw=0.7,
					alpha=0.45,
				)

	# Draw nodes and layer labels.
	colors = ["#0f766e"] + ["#2563eb"] * (n_layers - 2) + ["#b91c1c"]
	labels = ["Input"] + [f"Hidden {i}" for i in range(1, n_layers - 1)] + ["Output"]
	activation_labels = ["-"] + ["ReLU"] * max(0, n_layers - 2) + ["Linear"]

	for i, ys in enumerate(layer_nodes):
		ax.scatter(
			np.full_like(ys, x_positions[i], dtype=np.float32),
			ys,
			s=250,
			c=colors[i],
			edgecolors="white",
			linewidths=1.2,
			zorder=5,
		)
		ax.text(
			x_positions[i],
			1.02,
			f"Layer {i}: {labels[i]}\n{len(ys)} neurons | act={activation_labels[i]}",
			ha="center",
			va="top",
			fontsize=9.5,
			fontweight="bold",
			clip_on=False,
		)

	ax.set_title("MLP Architecture", fontsize=13, fontweight="bold", pad=18)
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1.08)
	ax.axis("off")


def plot_loss_curve(ax, losses, log_epochs, log_losses):
	"""Plot full loss curve with highlighted logged checkpoints."""
	epochs = np.arange(1, len(losses) + 1)
	ax.plot(epochs, losses, color="#1d4ed8", lw=2.0, label="MSE loss (all epochs)")
	ax.scatter(log_epochs, log_losses, color="#b91c1c", s=30, zorder=4, label="Printed every 100 epochs")
	ax.set_title("Training Loss Curve", fontsize=13, fontweight="bold")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.grid(alpha=0.3)
	ax.legend()

def main():
	"""Train the tiny MLP run and render architecture + loss visuals."""
	x_np, y_np = make_tiny_batch(seed=7)

	architecture = [x_np.shape[1], 8, 8, 1]
	_, losses, log_epochs, log_losses = train_model(
		x_np,
		y_np,
		layer_sizes=architecture[1:],
		epochs=1000,
		lr=0.001,
		log_every=100,
	)

	plt.style.use("seaborn-v0_8-whitegrid")
	fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))

	draw_mlp(axes[0], architecture)
	plot_loss_curve(axes[1], losses, log_epochs, log_losses)

	fig.suptitle("WhyyTorch Tiny Run: MLP Architecture + Loss Curve", fontsize=15, fontweight="bold")
	fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
	if "agg" in plt.get_backend().lower():
		out_path = Path("mlp_visualization.png")
		fig.savefig(out_path, dpi=160, bbox_inches="tight")
		print(f"Saved visualization to: {out_path.resolve()}")
	else:
		plt.show()


# =============================================================================
# BigramMLP visualizations
# =============================================================================
# Below: helpers to retrain the BigramMLP from BigramMLP.ipynb (block_size=3,
# 10-D embeddings, 200-unit hidden layer) and produce three plots intended for
# a blog / LinkedIn post:
#   1. Learned embedding space, projected to 2-D with PCA.
#   2. Loss vs learning rate (Karpathy-style in-loop exponential sweep).
#   3. Simple node-link MLP architecture diagram.
# All three are rendered side-by-side by main_bigram().


def load_bigram_words(path="bigram.txt"):
	"""Load words and build the char vocabulary used by BigramMLP.ipynb."""
	words = open(path).read().splitlines()
	stoi = {".": 0, **{chr(ord("a") + i): i + 1 for i in range(26)}}
	itos = {i: c for c, i in stoi.items()}
	return words, stoi, itos


def build_bigram_splits(words, stoi, block_size=3, seed=42):
	"""Word-level shuffled 80/10/10 split returning int64 (X, Y) per split."""
	rng = np.random.default_rng(seed)
	shuffled = list(words)
	rng.shuffle(shuffled)
	n = len(shuffled)
	n1 = int(0.8 * n)
	n2 = int(0.9 * n)

	def build(word_list):
		Xs, Ys = [], []
		for w in word_list:
			ctx = [0] * block_size
			for ch in w + ".":
				ix = stoi[ch]
				Xs.append(ctx)
				Ys.append(ix)
				ctx = ctx[1:] + [ix]
		return np.array(Xs, dtype=np.int64), np.array(Ys, dtype=np.int64)

	return build(shuffled[:n1]), build(shuffled[n1:n2]), build(shuffled[n2:])


def init_bigram_params(emb_dim=10, hidden=200, vocab=27, block_size=3, seed=42):
	"""Initialize the BigramMLP parameters identical to BigramMLP.ipynb cell 15."""
	np.random.seed(seed)
	C = WhyyTorch(np.random.randn(vocab, emb_dim))
	W1 = WhyyTorch(np.random.randn(block_size * emb_dim, hidden))
	b1 = WhyyTorch(np.random.randn(hidden))
	W2 = WhyyTorch(np.random.randn(hidden, vocab))
	b2 = WhyyTorch(np.random.randn(vocab))
	return C, W1, b1, W2, b2


def bigram_forward(C, W1, b1, W2, b2, Xb):
	"""Forward pass mirroring BigramMLP.ipynb cell 17."""
	emb = C[Xb].reshape(Xb.shape[0], -1)
	h = WhyyTorch.tanh(emb @ W1 + b1)
	return h @ W2 + b2


def train_bigram(C, W1, b1, W2, b2, Xtr, Ytr, lr=0.1, epochs=10000,
				 batch_size=32, log_every=2000, seed=0):
	"""Standard mini-batch SGD training loop. Returns the per-step loss array."""
	parameters = [C, W1, b1, W2, b2]
	rng = np.random.default_rng(seed)
	losses = np.zeros(epochs, dtype=np.float32)
	for epoch in range(epochs):
		idx = rng.integers(0, Xtr.shape[0], size=(batch_size,))
		Xb, Yb = Xtr[idx], Ytr[idx]
		logits = bigram_forward(C, W1, b1, W2, b2, Xb)
		loss = cross_entropy_loss(logits, Yb)
		for p in parameters:
			p.zero_grad()
		loss.backward()
		for p in parameters:
			p.data -= lr * p.grad
		losses[epoch] = float(loss.data)
		if log_every and epoch % log_every == 0:
			print(f"  epoch {epoch:5d} | loss {float(loss.data):.4f}")
	return losses


def lr_sweep(Xtr, Ytr, lr_log_min=-5.0, lr_log_max=0.0, steps=1000,
			 batch_size=32, seed=42):
	"""Karpathy-style in-loop sweep: ONE training run, lr increases exponentially.

	Returns (lrs, losses) as float arrays of length `steps`.
	"""
	C, W1, b1, W2, b2 = init_bigram_params(seed=seed)
	parameters = [C, W1, b1, W2, b2]
	lr_exps = np.linspace(lr_log_min, lr_log_max, steps)
	lrs = (10.0 ** lr_exps).astype(np.float32)
	rng = np.random.default_rng(seed)

	losses = np.zeros(steps, dtype=np.float32)
	for i in range(steps):
		idx = rng.integers(0, Xtr.shape[0], size=(batch_size,))
		Xb, Yb = Xtr[idx], Ytr[idx]
		logits = bigram_forward(C, W1, b1, W2, b2, Xb)
		loss = cross_entropy_loss(logits, Yb)
		for p in parameters:
			p.zero_grad()
		loss.backward()
		lr = float(lrs[i])
		for p in parameters:
			p.data -= lr * p.grad
		losses[i] = float(loss.data)
	# Clamp any divergent losses so the plot stays readable.
	losses = np.clip(losses, 0.0, 30.0)
	return lrs, losses


def pca_2d(M):
	"""Project rows of M to 2-D via SVD-based PCA. Pure NumPy."""
	Mc = M - M.mean(axis=0, keepdims=True)
	_, _, Vt = np.linalg.svd(Mc, full_matrices=False)
	return Mc @ Vt[:2].T


def plot_embedding_pca(ax, C_data, itos):
	"""Scatter PCA-projected embeddings with letter labels."""
	proj = pca_2d(C_data)
	ax.scatter(
		proj[:, 0], proj[:, 1],
		s=260, c="#0f766e", edgecolors="white", linewidths=1.2, zorder=3,
	)
	for i in range(proj.shape[0]):
		ax.annotate(
			itos[i], (proj[i, 0], proj[i, 1]),
			ha="center", va="center", fontsize=10, fontweight="bold",
			color="white", zorder=4,
		)
	ax.set_title("Learned character embeddings (PCA -> 2D)", fontsize=13, fontweight="bold")
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.grid(alpha=0.3)


def plot_lr_sweep(ax, lrs, losses, smooth_window=25):
	"""Plot loss vs learning rate (log x-axis), with a moving-average smoothing."""
	if smooth_window > 1 and smooth_window < len(losses):
		kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
		smoothed = np.convolve(losses, kernel, mode="valid")
		lrs_s = lrs[smooth_window - 1:]
	else:
		smoothed = losses
		lrs_s = lrs

	ax.plot(lrs_s, smoothed, color="#1d4ed8", lw=1.8, label="batch loss (smoothed)")
	best_i = int(np.argmin(smoothed))
	ax.scatter(
		[lrs_s[best_i]], [smoothed[best_i]],
		c="#b91c1c", s=70, zorder=5,
		label=f"min @ lr={lrs_s[best_i]:.4f}",
	)
	ax.set_xscale("log")
	ax.set_title("Loss vs Learning Rate (in-loop sweep)", fontsize=13, fontweight="bold")
	ax.set_xlabel("learning rate (log scale)")
	ax.set_ylabel("training batch loss")
	ax.grid(alpha=0.3, which="both")
	ax.legend()


def draw_bigram_mlp(ax, max_nodes_drawn=10):
	"""Node-link diagram showing the full BigramMLP pipeline including softmax."""
	# Stages: input chars -> embedded -> hidden(tanh) -> logits -> softmax probs
	layer_sizes = [3, 30, 200, 27, 27]
	layer_titles = ["Input", "Embedded", "Hidden", "Logits", "Probs"]
	layer_subtitles = [
		"3 char ids",
		"3 x 10 = 30",
		"200 (tanh)",
		"27",
		"27 (softmax)",
	]
	# Operation between layer i and i+1, with parameter count.
	edge_labels = [
		"embed C\n270 params",
		"Linear + tanh\n6,200 params",
		"Linear\n5,427 params",
		"softmax\n0 params",
	]

	n_layers = len(layer_sizes)
	x_positions = np.linspace(0.06, 0.94, n_layers)

	layer_y = []
	for size in layer_sizes:
		drawn = min(size, max_nodes_drawn)
		if drawn > 1:
			ys = np.linspace(0.18, 0.72, drawn)
		else:
			ys = np.array([0.45])
		layer_y.append(ys)

	# Edges between drawn nodes only.
	for i in range(n_layers - 1):
		for y1 in layer_y[i]:
			for y2 in layer_y[i + 1]:
				ax.plot(
					[x_positions[i], x_positions[i + 1]], [y1, y2],
					color="#94a3b8", lw=0.4, alpha=0.35,
				)

	# Edge labels (operation + param count) above each gap.
	for i in range(n_layers - 1):
		x_mid = (x_positions[i] + x_positions[i + 1]) / 2.0
		ax.text(
			x_mid, 0.88, edge_labels[i],
			ha="center", va="center", fontsize=8.0, color="#1e293b",
			bbox=dict(
				boxstyle="round,pad=0.25",
				fc="white", ec="#cbd5e1", lw=0.6, alpha=0.9,
			),
		)

	# Nodes + per-layer labels.
	colors = ["#0f766e", "#0891b2", "#2563eb", "#b91c1c", "#7c3aed"]
	for i, ys in enumerate(layer_y):
		ax.scatter(
			np.full_like(ys, x_positions[i], dtype=np.float32), ys,
			s=180, c=colors[i], edgecolors="white", linewidths=1.2, zorder=5,
		)
		if layer_sizes[i] > max_nodes_drawn:
			ax.text(
				x_positions[i], 0.78, f"... ({layer_sizes[i]} total)",
				ha="center", va="bottom", fontsize=7.5, color="#475569",
			)
		ax.text(
			x_positions[i], 0.12, layer_titles[i],
			ha="center", va="top", fontsize=10.0, fontweight="bold",
		)
		ax.text(
			x_positions[i], 0.06, layer_subtitles[i],
			ha="center", va="top", fontsize=8.5, color="#475569",
		)

	total_params = 27 * 10 + (30 * 200 + 200) + (200 * 27 + 27)  # 11,897
	ax.set_title(
		f"BigramMLP architecture  ·  {total_params:,} total parameters",
		fontsize=12.5, fontweight="bold", pad=14,
	)
	ax.set_xlim(0, 1)
	ax.set_ylim(0.0, 0.98)
	ax.axis("off")


def main_bigram():
	"""Run BigramMLP training + LR sweep and render the three LinkedIn plots."""
	print("Loading bigram dataset...")
	words, stoi, itos = load_bigram_words("bigram.txt")
	(Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = build_bigram_splits(words, stoi)
	print(f"  train: {Xtr.shape}, dev: {Xdev.shape}, test: {Xte.shape}")

	print("Running learning-rate sweep (1000 steps, lr 1e-5 -> 1.0)...")
	lrs, sweep_losses = lr_sweep(Xtr, Ytr, steps=1000)

	print("Training final model for embedding plot (lr=0.1, 10000 epochs)...")
	C, W1, b1, W2, b2 = init_bigram_params(seed=42)
	train_bigram(C, W1, b1, W2, b2, Xtr, Ytr, lr=0.1, epochs=10000, log_every=2000)

	plt.style.use("seaborn-v0_8-whitegrid")
	fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0))
	plot_embedding_pca(axes[0], C.data, itos)
	plot_lr_sweep(axes[1], lrs, sweep_losses)
	draw_bigram_mlp(axes[2])

	fig.suptitle(
		"BigramMLP: Embedding space, Loss vs LR, Architecture",
		fontsize=16, fontweight="bold",
	)
	fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=2.5)

	if "agg" in plt.get_backend().lower():
		out = Path("bigram_mlp_visualization.png")
		fig.savefig(out, dpi=160, bbox_inches="tight")
		print(f"Saved to: {out.resolve()}")
	else:
		plt.show()


if __name__ == "__main__":
	main_bigram()
