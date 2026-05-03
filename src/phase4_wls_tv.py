import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import os

os.makedirs('results', exist_ok=True)

sino_noisy  = np.load('results/p2_results/sino_noisy.npy')
raw_counts  = np.load('results/p2_results/raw_counts.npy')
sino_clean  = np.load('results/p2_results/sino_clean.npy') 
W_raw  = np.load('results/p3_results/W_raw.npy')     
W_norm = np.load('results/p3_results/W_norm.npy')    

# Chi²-Slope Stopping

def wls_tv_chi2_slope_stop(
        sino_noisy, W_norm, W_raw, sino_clean,
        lam=0.02, alpha=0.5,
        max_iters=200,
        slope_ratio_threshold=0.005,   # stop when slope < 2% of initial slope
        slope_window=8,               # window to estimate current slope
        min_iters=20):                # never stop before this many iters

    print("🔧 WLS-TV (Chi²-Slope Stopping)")
    print(f"   slope_ratio_threshold = {slope_ratio_threshold}")
    print(f"   slope_window          = {slope_window}")
    print(f"   min_iters             = {min_iters}\n")

    x = sino_noisy.copy()
    chi2_history  = []
    rmse_history  = []
    delta_history = []

    x_best   = x.copy()
    best_iter = 0
    initial_slope = None

    for i in range(max_iters):
        x_old = x.copy()

        # ── Gradient step ──
        grad   = W_norm * (x - sino_noisy)
        x_grad = x - alpha * grad

        # ── TV proximal step ──
        x = denoise_tv_chambolle(x_grad, weight=alpha * lam,
                                 max_num_iter=20, eps=1e-4)
        x = np.clip(x, 0.0, sino_noisy.max())

        # ── Chi² (mean form — scale doesn't matter, only slope does) ──
        chi2  = np.mean(W_raw * (x - sino_noisy)**2)
        chi2_history.append(chi2)

        # ── RMSE diagnostic ──
        rmse  = np.sqrt(np.mean((x - sino_clean)**2))
        rmse_history.append(rmse)

        delta = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-8)
        delta_history.append(delta)

        print(f"Iter {i+1:3d}: Chi²={chi2:.5f} | RMSE={rmse:.4f} | δ={delta:.6f}")

        # ── Capture initial slope (between iter 2 and 3, after transient) ──
        if i == 2:
            initial_slope = chi2_history[2] - chi2_history[1]
            print(f"    Initial Chi² slope: {initial_slope:.6f}")

        # ── Slope-based stopping ──
        if initial_slope is not None and i >= min_iters and len(chi2_history) >= slope_window:
            recent = chi2_history[-slope_window:]
            # linear fit slope over window (robust to noise)
            xs = np.arange(slope_window)
            current_slope = np.polyfit(xs, recent, 1)[0]
            slope_ratio   = current_slope / initial_slope

            print(f"            slope_ratio={slope_ratio:.4f}", end="")

            if slope_ratio < slope_ratio_threshold:
                print(f"   Chi² slope flattened (ratio={slope_ratio:.4f} < {slope_ratio_threshold})")
                x_best    = x.copy()
                best_iter = i + 1
                break
            else:
                print()

        # ── Track best RMSE iter for the plot marker  ──
        if rmse <= min(rmse_history):
            x_best    = x.copy()
            best_iter = i + 1

    if best_iter == 0:
        best_iter = len(chi2_history)
        x_best = x.copy()
        print(f"     Max iters reached without triggering slope stop.")

    print(f"\n Stopped at iteration: {best_iter}")
    print(f"   Chi² at stop : {chi2_history[best_iter-1]:.5f}")
    print(f"   RMSE at stop : {rmse_history[best_iter-1]:.4f}")
    print(f"   Best RMSE    : {min(rmse_history):.4f}  (iter {np.argmin(rmse_history)+1})")

    return x_best, chi2_history, rmse_history, delta_history, best_iter


# ─────────────────────────────────────────────────────────────
# 3. Run
# ─────────────────────────────────────────────────────────────
sino_denoised, chi2_history, rmse_history, delta_history, best_iter = \
    wls_tv_chi2_slope_stop(
        sino_noisy, W_norm, W_raw, sino_clean,
        lam=0.02,
        alpha=0.5,
        max_iters=200,
        slope_ratio_threshold=0.005,
        slope_window=8,
        min_iters=20
    )

np.save('results/p4_results/sino_denoised.npy', sino_denoised)

rmse_noisy = np.sqrt(np.mean((sino_noisy - sino_clean)**2))
print(f"\nPhase 4 Complete")
print(f"   Noisy RMSE baseline : {rmse_noisy:.4f}")
print(f"   Denoised RMSE       : {rmse_history[best_iter-1]:.4f}")
print(f"   Improvement         : {((rmse_noisy - rmse_history[best_iter-1]) / rmse_noisy * 100):.1f}%")

# ─────────────────────────────────────────────────────────────
# 4. Visualization
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

axes[0].imshow(sino_noisy,    cmap='gray', aspect='auto')
axes[0].set_title('Noisy Sinogram')
axes[0].axis('off')

axes[1].imshow(sino_denoised, cmap='gray', aspect='auto')
axes[1].set_title(f'Denoised (stopped iter {best_iter})')
axes[1].axis('off')

# Chi² with stop marker
axes[2].plot(chi2_history, 'r-', linewidth=2, label='Chi²')
axes[2].axvline(x=best_iter-1, color='k', linestyle='--', label=f'Stop (iter {best_iter})')
axes[2].set_title('Chi² (Slope-Based Stop)')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Chi²')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# RMSE with stop and best markers
best_rmse_iter = int(np.argmin(rmse_history))
axes[3].plot(rmse_history, 'b-', linewidth=2, label='RMSE')
axes[3].axvline(x=best_iter-1,    color='k',  linestyle='--', label=f'Stop (iter {best_iter})')
axes[3].axvline(x=best_rmse_iter, color='g',  linestyle=':',  label=f'Best RMSE (iter {best_rmse_iter+1})')
axes[3].set_title('RMSE (Diagnostic)')
axes[3].set_xlabel('Iteration')
axes[3].set_ylabel('RMSE')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p4_results/phase4_validation.png', dpi=150)
plt.show()

print("Saved: results/sino_denoised.npy, results/phase4_validation.png")