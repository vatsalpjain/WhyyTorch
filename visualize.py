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

from Autograd import WhyyTorch, MLP


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


if __name__ == "__main__":
	main()
