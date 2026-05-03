import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import os

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0. Load Data
# ─────────────────────────────────────────────────────────────
sino_noisy = np.load('results/p2_results/sino_noisy.npy')
W_norm_loaded = np.load('results/W.npy')
raw_counts = np.load('results/p2_results/raw_counts.npy')
sino_clean = np.load('results/p2_results/sino_clean.npy')

# ─────────────────────────────────────────────────────────────
# 1. Compute Raw Weights (For Discrepancy Principle)
# ─────────────────────────────────────────────────────────────
y_raw = np.clip(raw_counts, 1.0, None)
sigma_e = 5.0
var_log = (y_raw + sigma_e**2) / (y_raw**2)
W_raw = 1.0 / var_log  # True inverse variances (unscaled)

# Prepare Normalized Weights for Optimization
W_norm = (W_norm_loaded - W_norm_loaded.min()) / (W_norm_loaded.max() - W_norm_loaded.min() + 1e-8)

# ─────────────────────────────────────────────────────────────
# 2. WLS-TV with Simple Discrepancy Principle
# ─────────────────────────────────────────────────────────────
def wls_tv_discrepancy(sino_noisy, W_norm, W_raw, lam=0.02, alpha=0.5, 
                       max_iters=100, chi2_target=1.0):
    """
    Solves WLS-TV and stops immediately when Chi^2 >= target.
    This implements the Discrepancy Principle: stop when residual matches noise.
    """
    print(f"🔧 Starting Discrepancy-Based WLS-TV...")
    print(f"   Target Chi²: {chi2_target} (Stop when Chi² >= Target)")
    
    x = sino_noisy.copy()
    rmse_history = []
    chi2_history = []
    
    for i in range(max_iters):
        x_old = x.copy()
        
        # --- Optimization Step ---
        grad = W_norm * (x - sino_noisy)
        x_grad = x - alpha * grad
        tau = alpha * lam
        x = denoise_tv_chambolle(x_grad, weight=tau, max_num_iter=20, eps=1e-4)
        x = np.clip(x, 0.0, sino_noisy.max())
        
        # --- Discrepancy Check ---
        residual = W_raw * (x - sino_noisy)**2
        chi2 = np.mean(residual)
        chi2_history.append(chi2)
        
        # --- Performance Monitoring (For Report Only) ---
        rmse_oracle = np.sqrt(np.mean((x - sino_clean)**2))
        rmse_history.append(rmse_oracle)
        
        print(f"   Iter {i+1:2d}: RMSE={rmse_oracle:.4f} | Chi²={chi2:.3f} | δ={np.linalg.norm(x-x_old)/(np.linalg.norm(x_old)+1e-8):.5f}")
        
        # --- Stopping Condition ---
        # Stop when Chi^2 reaches or crosses the target (1.0)
        if chi2 >= chi2_target:
            print(f"   🛑 Discrepancy Principle Triggered: Chi²={chi2:.3f} >= {chi2_target}")
            break
            
    return x, W_norm, rmse_history, chi2_history

# Run optimization
sino_denoised, W_norm, rmse_history, chi2_history = wls_tv_discrepancy(
    sino_noisy, W_norm, W_raw, lam=0.02, alpha=0.5, 
    max_iters=100, chi2_target=1.0
)

# ─────────────────────────────────────────────────────────────
# 3. Validation & Visualization
# ─────────────────────────────────────────────────────────────
rmse_final = np.sqrt(np.mean((sino_denoised - sino_clean)**2))
rmse_noisy = np.sqrt(np.mean((sino_noisy - sino_clean)**2))

print(f"\n✅ Phase 4 Complete")
print(f"   Sinogram RMSE: {rmse_final:.4f} (noisy baseline: {rmse_noisy:.4f})")
print(f"   ✅ Improvement: {((rmse_noisy - rmse_final) / rmse_noisy * 100):.1f}%")

# Save
np.save('results/sino_denoised.npy', sino_denoised)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(sino_denoised, cmap='gray', aspect='auto')
axes[0].set_title('Denoised Sinogram')
axes[0].axis('off')

axes[1].plot(rmse_history, 'b-', label='Oracle RMSE')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Convergence')
axes[1].grid(True, alpha=0.3)

axes[2].plot(chi2_history, 'r-', label='Chi² Statistic')
axes[2].axhline(y=1.0, color='k', linestyle='--', label='Target (1.0)')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Chi²')
axes[2].set_title('Discrepancy Principle (Simple)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/phase4_validation.png', dpi=150)
plt.show()

print("💾 Saved results and plots.")