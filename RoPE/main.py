import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# Setup: shared content embeddings and shared projection weights
# ----------------------------------------------------------------------------
T = 12          # sequence length for the main heatmaps
d_model = 64    # model / head dimension (must be even for RoPE pairing)

X = torch.randn(T, d_model)                 # token content embeddings (identical for both methods)
W_q = torch.randn(d_model, d_model) * 0.1
W_k = torch.randn(d_model, d_model) * 0.1

scale = d_model ** 0.5


def sinusoidal_pe(positions, dim, base=10000.0):
    """Standard Transformer sinusoidal positional encoding, evaluated at arbitrary positions."""
    positions = positions.float().unsqueeze(1)                     # (T, 1)
    i = torch.arange(dim // 2).float()                             # (dim/2,)
    div = base ** (2 * i / dim)                                    # (dim/2,)
    angles = positions / div                                       # (T, dim/2)
    pe = torch.zeros(len(positions), dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def rope_angles(positions, dim, base=10000.0):
    i = torch.arange(dim // 2).float()
    freqs = 1.0 / (base ** (2 * i / dim))                          # (dim/2,)
    return positions.float().unsqueeze(1) * freqs.unsqueeze(0)     # (T, dim/2)


def apply_rope(x, positions, base=10000.0):
    """Rotate adjacent pairs (x0,x1), (x2,x3), ... by an angle proportional to position."""
    T_, dim = x.shape
    angles = rope_angles(positions, dim, base)                     # (T, dim/2)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)         # (T, dim)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)         # (T, dim)

    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    x_rotated_pairs = torch.stack([-x2, x1], dim=-1).reshape(T_, dim)

    return x * cos + x_rotated_pairs * sin


def attention_scores_absolute(X, positions, W_q, W_k):
    pe = sinusoidal_pe(positions, X.shape[-1])
    X_pe = X + pe
    Q = X_pe @ W_q
    K = X_pe @ W_k
    return (Q @ K.T) / scale


def attention_scores_rope(X, positions, W_q, W_k):
    Q = X @ W_q
    K = X @ W_k
    Q = apply_rope(Q, positions)
    K = apply_rope(K, positions)
    return (Q @ K.T) / scale


def softmax_rows(scores):
    return torch.softmax(scores, dim=-1)


# ----------------------------------------------------------------------------
# Experiment 1: plain attention heatmaps, positions 0..T-1
# ----------------------------------------------------------------------------
positions_base = torch.arange(T)

scores_abs = attention_scores_absolute(X, positions_base, W_q, W_k)
scores_rope = attention_scores_rope(X, positions_base, W_q, W_k)

attn_abs = softmax_rows(scores_abs)
attn_rope = softmax_rows(scores_rope)

# ----------------------------------------------------------------------------
# Experiment 2: shift test — same content window, relabeled positions
# ----------------------------------------------------------------------------
shift = 20
positions_shifted = torch.arange(shift, shift + T)

scores_abs_shifted = attention_scores_absolute(X, positions_shifted, W_q, W_k)
scores_rope_shifted = attention_scores_rope(X, positions_shifted, W_q, W_k)

attn_abs_shifted = softmax_rows(scores_abs_shifted)
attn_rope_shifted = softmax_rows(scores_rope_shifted)

# how much does the attention *pattern* change under a pure position shift?
abs_diff = (attn_abs - attn_abs_shifted).abs().mean().item()
rope_diff = (attn_rope - attn_rope_shifted).abs().mean().item()

# ----------------------------------------------------------------------------
# Experiment 3: theoretical RoPE score vs. relative distance
# ----------------------------------------------------------------------------
n_trials = 200
d_small = 64
rel_distances = torch.arange(0, 64)
avg_scores = torch.zeros(len(rel_distances))

for trial in range(n_trials):
    q = torch.randn(d_small)
    k = torch.randn(d_small)
    for idx, rel in enumerate(rel_distances):
        q_rot = apply_rope(q.unsqueeze(0), torch.tensor([0]))[0]
        k_rot = apply_rope(k.unsqueeze(0), torch.tensor([rel.item()]))[0]
        avg_scores[idx] += (q_rot @ k_rot) / d_small**0.5
avg_scores /= n_trials

# ============================================================================
# PLOTTING
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(19, 11))

vmin = min(attn_abs.min().item(), attn_rope.min().item())
vmax = max(attn_abs.max().item(), attn_rope.max().item())

im0 = axes[0, 0].imshow(attn_abs.detach().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Absolute (sinusoidal) PE\nattention weights, positions 0-11")
axes[0, 0].set_xlabel("key position")
axes[0, 0].set_ylabel("query position")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(attn_rope.detach().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 1].set_title("RoPE\nattention weights, positions 0-11")
axes[0, 1].set_xlabel("key position")
axes[0, 1].set_ylabel("query position")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

axes[0, 2].plot(rel_distances.numpy(), avg_scores.numpy(), color="#2a6f97", linewidth=2)
axes[0, 2].axhline(0, color="gray", linewidth=0.8)
axes[0, 2].set_title("RoPE: avg. raw score vs. relative distance\n(averaged over 200 random q,k pairs)")
axes[0, 2].set_xlabel("relative distance |m - n|")
axes[0, 2].set_ylabel("average q·k score")

vmin2 = min(attn_abs_shifted.min().item(), attn_rope_shifted.min().item())
vmax2 = max(attn_abs_shifted.max().item(), attn_rope_shifted.max().item())

im2 = axes[1, 0].imshow(attn_abs_shifted.detach().numpy(), cmap="viridis", vmin=vmin2, vmax=vmax2)
axes[1, 0].set_title(f"Absolute PE — SAME window,\nrelabeled as positions {shift}-{shift+T-1}")
axes[1, 0].set_xlabel("key position (index in window)")
axes[1, 0].set_ylabel("query position (index in window)")
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

im3 = axes[1, 1].imshow(attn_rope_shifted.detach().numpy(), cmap="viridis", vmin=vmin2, vmax=vmax2)
axes[1, 1].set_title(f"RoPE — SAME window,\nrelabeled as positions {shift}-{shift+T-1}")
axes[1, 1].set_xlabel("key position (index in window)")
axes[1, 1].set_ylabel("query position (index in window)")
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

axes[1, 2].bar(
    ["Absolute PE", "RoPE"],
    [abs_diff, rope_diff],
    color=["#c1121f", "#2a6f97"],
)
axes[1, 2].set_title("Mean |attention change|\nwhen the whole window is shifted by +20 positions")
axes[1, 2].set_ylabel("mean absolute difference in attention weight")
for i, v in enumerate([abs_diff, rope_diff]):
    axes[1, 2].text(i, v, f"{v:.4f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("rope_vs_absolute_pe.png", dpi=150)
print("Saved figure.")
print(f"Mean |attention change| under shift  —  Absolute PE: {abs_diff:.5f}   RoPE: {rope_diff:.5f}")