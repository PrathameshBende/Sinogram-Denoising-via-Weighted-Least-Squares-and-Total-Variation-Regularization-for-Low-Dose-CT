import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0. Load Data
# ─────────────────────────────────────────────────────────────
sino_noisy = np.load('results/p2_results/sino_noisy.npy')
raw_counts = np.load('results/p2_results/raw_counts.npy')
sino_clean = np.load('results/p2_results/sino_clean.npy')

# ─────────────────────────────────────────────────────────────
# 1. Compute Weight Matrix
# ─────────────────────────────────────────────────────────────
def compute_weights(raw_counts, sigma_e=5.0):
    """
    Var(log(y)) ≈ (y + sigma_e²) / y²
    W_raw  = 1 / Var       (true physical weights)
    W_norm = normalized    (for stable optimization)
    """
    y = np.clip(raw_counts, 1.0, None)

    var_log = (y + sigma_e**2) / (y**2)

    W_raw = 1.0 / var_log
    W_norm = W_raw / (W_raw.max() + 1e-8)

    return W_raw, W_norm

W_raw, W_norm = compute_weights(raw_counts, sigma_e=5.0)

# ─────────────────────────────────────────────────────────────
# 2. Validation (use W_norm for interpretation)
# ─────────────────────────────────────────────────────────────
print("✅ Phase 3: Weight Matrix Computed")

print(f"   W_raw range:  [{W_raw.min():.4f}, {W_raw.max():.4f}]")
print(f"   W_norm range: [{W_norm.min():.4f}, {W_norm.max():.4f}]")
print(f"   Mean W_norm:  {W_norm.mean():.4f}")

high_att_mask = sino_clean > 2.0
low_att_mask = sino_clean < 0.5

mean_w_high = W_norm[high_att_mask].mean()
mean_w_low = W_norm[low_att_mask].mean()

print(f"   Mean Weight (High Attenuation): {mean_w_high:.4f}")
print(f"   Mean Weight (Low Attenuation):  {mean_w_low:.4f}")

assert mean_w_low > mean_w_high, "❌ Weight logic failed"

# ─────────────────────────────────────────────────────────────
# 3. Save (IMPORTANT: save both)
# ─────────────────────────────────────────────────────────────
np.save('results/p3_results/W_raw.npy', W_raw)
np.save('results/p3_results/W_norm.npy', W_norm)

# ─────────────────────────────────────────────────────────────
# 4. Visualization (use W_norm)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(sino_clean, cmap='gray', aspect='auto')
axes[0].set_title('Clean Sinogram')
axes[0].axis('off')

axes[1].imshow(raw_counts, cmap='gray', aspect='auto')
axes[1].set_title('Raw Counts')
axes[1].axis('off')

axes[2].imshow(W_norm, cmap='gray', aspect='auto')
axes[2].set_title('Weight Matrix (Normalized)\nBright = Trust')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('results/p3_results/phase3_weights.png', dpi=150)
plt.show()

print("💾 Saved W_raw.npy and W_norm.npy")