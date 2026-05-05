import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import os

os.makedirs('results/p4_results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0. Load Data
# ─────────────────────────────────────────────────────────────
sino_noisy  = np.load('results/p2_results/sino_noisy.npy')
raw_counts  = np.load('results/p2_results/raw_counts.npy')
sino_clean  = np.load('results/p2_results/sino_clean.npy')
W_raw       = np.load('results/p3_results/W_raw.npy')
W_norm      = np.load('results/p3_results/W_norm.npy')

# ─────────────────────────────────────────────────────────────
# 1. Core Optimization Function (Kept Intact)
# ─────────────────────────────────────────────────────────────
def wls_tv_chi2_slope_stop(sino_noisy, W_norm, W_raw, sino_clean, 
                           lam=0.02, alpha=0.5, max_iters=2000, 
                           slope_ratio_threshold=0.005, slope_window=8, min_iters=20):
    x = sino_noisy.copy()
    chi2_history  = []
    rmse_history  = []
    
    initial_slope = None
    best_iter = 0
    
    for i in range(max_iters):
        x_old = x.copy()
        
        # Gradient step
        grad = W_norm * (x - sino_noisy)
        x_grad = x - alpha * grad
        
        # TV proximal step
        x = denoise_tv_chambolle(x_grad, weight=alpha * lam, max_num_iter=20, eps=1e-4)
        x = np.clip(x, 0.0, sino_noisy.max())
        
        # Metrics
        chi2 = np.mean(W_raw * (x - sino_noisy)**2)
        chi2_history.append(chi2)
        
        rmse = np.sqrt(np.mean((x - sino_clean)**2))
        rmse_history.append(rmse)
        
        # Initial slope capture
        if i == 2:
            initial_slope = chi2_history[2] - chi2_history[1]
            
        # Slope-based stopping
        if initial_slope is not None and i >= min_iters and len(chi2_history) >= slope_window:
            recent = chi2_history[-slope_window:]
            current_slope = np.polyfit(np.arange(slope_window), recent, 1)[0]
            if abs(current_slope / initial_slope) < slope_ratio_threshold:
                best_iter = i + 1
                break
                
    # Fallback: if slope never triggers, pick iteration with minimum RMSE
    if best_iter == 0:
        best_iter = int(np.argmin(rmse_history)) + 1
        
    return x, chi2_history, rmse_history, best_iter

# ─────────────────────────────────────────────────────────────
# 2. λ-Sweep & Automatic Selection
# ─────────────────────────────────────────────────────────────
print("🔍 Starting λ-Sweep for WLS-TV Regularization...")
# Test 12 values from 0.001 to 0.03 (logarithmically spaced)
lambda_values = np.logspace(-3, np.log10(0.03), 12)

best_overall_rmse = np.inf
best_lambda       = None
best_sino_denoised = None
results_table     = []

for lam in lambda_values:
    print(f"\n--- Testing λ = {lam:.4f} ---")
    sino_out, chi2_hist, rmse_hist, stop_iter = wls_tv_chi2_slope_stop(
        sino_noisy, W_norm, W_raw, sino_clean,
        lam=lam, alpha=0.5, max_iters=2000,
        slope_ratio_threshold=0.005, slope_window=8, min_iters=20
    )
    
    # Use the minimum RMSE achieved during optimization for fair comparison
    min_rmse = np.min(rmse_hist)
    results_table.append((lam, min_rmse, stop_iter))
    
    if min_rmse < best_overall_rmse:
        best_overall_rmse = min_rmse
        best_lambda = lam
        best_sino_denoised = sino_out.copy()
        
    print(f"   → Min RMSE: {min_rmse:.4f} | Stopped at iter: {stop_iter}")

# ─────────────────────────────────────────────────────────────
# 3. Results Summary & Visualization
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 λ-SWEEP RESULTS")
print("="*60)
print(f"{'λ (Lambda)':<12} | {'Min RMSE':<10} | {'Stop Iter'}")
print("-"*40)
for lam, rmse, it in results_table:
    marker = " ← BEST" if lam == best_lambda else ""
    print(f"{lam:<12.4f} | {rmse:<10.4f} | {it:<9}{marker}")
print("="*60)
print(f"✅ Selected λ: {best_lambda:.4f} (Min RMSE: {best_overall_rmse:.4f})")

# Save best denoised sinogram for Phase 5
np.save('results/p4_results/sino_denoised.npy', best_sino_denoised)

# Plot RMSE vs λ
lams, rmses, iters = zip(*results_table)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogx(lams, rmses, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=best_lambda, color='r', linestyle='--', label=f'Best λ={best_lambda:.4f}')
axes[0].set_xlabel('Regularization Strength (λ)')
axes[0].set_ylabel('Minimum Sinogram RMSE')
axes[0].set_title('λ-Sweep: RMSE vs Regularization')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot best RMSE trajectory
best_idx = list(lams).index(best_lambda)
_, _, best_rmse_hist, _ = wls_tv_chi2_slope_stop(
    sino_noisy, W_norm, W_raw, sino_clean, lam=best_lambda, max_iters=2000
)
axes[1].plot(best_rmse_hist, 'b-', linewidth=2)
axes[1].axhline(y=best_overall_rmse, color='r', linestyle='--', label=f'Best RMSE={best_overall_rmse:.4f}')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Sinogram RMSE')
axes[1].set_title(f'Convergence Curve for Best λ={best_lambda:.4f}')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('results/experiments/lambda_sweep.png', dpi=150)
plt.show()

print("💾 Saved: results/p4_results/sino_denoised.npy & lambda_sweep.png")