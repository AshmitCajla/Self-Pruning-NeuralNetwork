# Sparse Gate Pruning on CIFAR-10

This project investigates **learnable weight pruning** via soft gate scores on a Multi-Layer Perceptron trained on CIFAR-10. Rather than hard-cutting weights post-training, gates are learned *jointly* with the network using an L1 sparsity penalty, letting the model decide which connections matter.


## What's Going On

Most neural networks are over-parameterized, they carry thousands of weights that contribute almost nothing to the final prediction. This project makes those redundant weights *visible* and *removable* during training itself.

Each linear layer in the network is wrapped with a `PrunableLinear` module that learns a **gate score** per weight. A sigmoid activation turns these scores into soft on/off switches:

```
output = Linear(x, weight × sigmoid(gate_scores)) + bias
```

During training, an **L1 regularization term** penalizes active gates, nudging the network to zero out unnecessary connections. The result: a sparse model that's measurably leaner while staying competitive on accuracy.


## Architecture

```
Input (3×32×32 = 3072)
        ↓
  PrunableLinear(3072 → 512)  ← 1,572,864 learnable gates
        ↓ ReLU
  PrunableLinear(512 → 256)   ←   131,072 learnable gates
        ↓ ReLU
  PrunableLinear(256 → 10)    ←     2,560 learnable gates
        ↓
  Class Logits (10 classes)
```

A `BaselineMLP` (same shape, no gates) is trained in parallel as a reference point.


## Configuration

| Parameter | Value |
|---|---|
| Batch Size | 128 |
| Epochs | 15 (with early stopping) |
| Learning Rate | 1e-3 (Cosine Annealed → 1e-6) |
| Sparsity Lambdas | `[1e-6, 1e-5, 1e-4]` |
| Sparsity Threshold | 0.01 (gate < this → "pruned") |
| Early Stop Patience | 3 epochs |
| Optimizer | Adam |
| Seed | 42 |

---

## Experiment Design

Three prunable models are trained, one per lambda value, against a single baseline:

```
Baseline → no gates, no penalty, full weights
λ=1e-6   → very mild pressure to prune
λ=1e-5   → moderate pruning pressure
λ=1e-4   → aggressive pruning pressure
```

This sweeps the **accuracy-sparsity tradeoff curve** — the core question being: *how much sparsity can we gain before accuracy noticeably degrades?*

---

## Outputs & Plots

All plots are saved to `results/`:

| File | Description |
|---|---|
| `loss_<λ>.png` | Train vs. validation loss curves |
| `acc_<λ>.png` | Train vs. validation accuracy curves |
| `sp_<λ>.png` | Sparsity % over training epochs |
| `lr_<λ>.png` | Learning rate schedule (cosine decay) |
| `tradeoff.png` | Sparsity vs. final accuracy across all λ values |
| `gates.png` | Gate value distribution of the best model |

---

## Running It

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run the notebook
jupyter notebook tredence.ipynb
```

CIFAR-10 data will be auto-downloaded to `./data/` on first run.

---

## Key Concepts

**Soft Gating** — Instead of hard binary masks, sigmoid gates allow smooth gradients to flow through during backprop. The network learns pruning rather than having it imposed.

**L1 Penalty on Gates** — Adding `λ × Σ|gate|` to the loss creates pressure toward zero. Higher λ = more aggressive pruning.

**Cosine LR Annealing** — Gradually reducing the learning rate helps the model settle into sparse solutions rather than oscillating.

**Early Stopping** — Monitors validation loss with patience=3 to prevent overfitting and wasted compute, restoring the best checkpoint at the end.

---

## Insights to Look For

- Does a small λ give "free" sparsity with no accuracy cost?
- At what λ does the accuracy cliff appear?
- Does the gate distribution show a clear bimodal pattern (near 0 or near 1)?
- How does sparsity evolve epoch-by-epoch — does it grow steadily or jump?

---

## Project Structure

```
.
├── tredence.ipynb      # Main experiment notebook
├── results/            # Auto-generated plots
├    ├── loss_*.png
├    ├── acc_*.png
├    ├── sp_*.png
├    ├── lr_*.png
├    ├── tradeoff.png
├    └── gates.png
├── Data
```

---
