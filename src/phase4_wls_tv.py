import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import iradon
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import os

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0. Load Data & Setup Evaluation Geometry
# ─────────────────────────────────────────────────────────────
sino_noisy  = np.load('results/p2_results/sino_noisy.npy')
raw_counts  = np.load('results/p2_results/raw_counts.npy')
sino_clean  = np.load('results/p2_results/sino_clean.npy')
x_true      = np.load('results/p2_results/x_true.npy')
W_raw       = np.load('results/p3_results/W_raw.npy')
W_norm      = np.load('results/p3_results/W_norm.npy')

# Reconstruction geometry & circular ROI mask (matches Phase 5)
angles = np.linspace(0, 180, 360, endpoint=False)
y_grid, x_grid = np.ogrid[:512, :512]
mask = (x_grid - 255.5)**2 + (y_grid - 255.5)**2 <= 255**2

# ─────────────────────────────────────────────────────────────
# 1. Core Optimization Function
# ─────────────────────────────────────────────────────────────
def wls_tv_chi2_slope_stop(
        sino_noisy, W_norm, W_raw, sino_clean, x_true, mask, angles,
        lam=0.02, alpha=0.5,
        max_iters=2000,
        slope_ratio_threshold=0.0035,
        slope_window=8,
        min_iters=20):

    print("🔧 WLS-TV (Chi²-Slope Stopping + Image Diagnostics)")
    print(f"   slope_ratio_threshold = {slope_ratio_threshold}")
    print(f"   slope_window          = {slope_window}")
    print(f"   min_iters             = {min_iters}\n")

    x = sino_noisy.copy()
    chi2_history       = []
    rmse_sino_history  = []
    rmse_img_history   = []
    ssim_img_history   = []
    delta_history      = []

    x_best   = x.copy()
    best_iter = 0
    initial_slope = None

    for i in range(max_iters):
        x_old = x.copy()

        # ── Forward Step: Gradient descent on weighted data fidelity ──
        grad   = W_norm * (x - sino_noisy)
        x_grad = x - alpha * grad

        # ── Backward Step: TV proximal mapping ──
        x = denoise_tv_chambolle(x_grad, weight=alpha * lam,
                                 max_num_iter=20, eps=1e-4)
        x = np.clip(x, 0.0, sino_noisy.max())

        # ── Chi² Statistic (Physics-based stopping) ──
        chi2  = np.mean(W_raw * (x - sino_noisy)**2)
        chi2_history.append(chi2)

        # ── Sinogram RMSE ──
        rmse_sino  = np.sqrt(np.mean((x - sino_clean)**2))
        rmse_sino_history.append(rmse_sino)

        # ── Image-Domain Diagnostics (Reconstruct for sanity check) ──
        x_recon = iradon(x, theta=angles, filter_name='ramp', circle=True)
        rmse_img = np.sqrt(mean_squared_error(x_true[mask], x_recon[mask]))
        ssim_img = ssim(x_true[mask], x_recon[mask], data_range=x_true.max())
        
        rmse_img_history.append(rmse_img)
        ssim_img_history.append(ssim_img)

        # ── Update Step Magnitude ──
        delta = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-8)
        delta_history.append(delta)

        # ── Console Output ──
        print(f"Iter {i+1:3d}: Chi²={chi2:.5f} | Sino-RMSE={rmse_sino:.4f} | "
              f"Img-RMSE={rmse_img:.4f} | Img-SSIM={ssim_img:.4f} | δ={delta:.6f}")

        # ── Capture initial Chi² slope ──
        if i == 2:
            initial_slope = chi2_history[2] - chi2_history[1]
            print(f"    Initial Chi² slope: {initial_slope:.6f}")

        # ── Slope-based stopping ──
        if initial_slope is not None and i >= min_iters and len(chi2_history) >= slope_window:
            recent = chi2_history[-slope_window:]
            xs = np.arange(slope_window)
            current_slope = np.polyfit(xs, recent, 1)[0]
            slope_ratio   = current_slope / initial_slope

            print(f"            slope_ratio={slope_ratio:.4f}", end="")

            if slope_ratio < slope_ratio_threshold:
                print(f"   ✅ Chi² slope flattened (ratio={slope_ratio:.4f} < {slope_ratio_threshold})")
                x_best    = x.copy()
                best_iter = i + 1
                break
            else:
                print()

        # ── Track best Image RMSE iterate ──
        if rmse_img <= min(rmse_img_history):
            x_best    = x.copy()
            best_iter = i + 1

    if best_iter == 0:
        best_iter = len(chi2_history)
        x_best = x.copy()
        print(f"     ⚠️ Max iters reached without triggering slope stop.")

    print(f"\n🏁 Stopped at iteration: {best_iter}")
    print(f"   Chi² at stop      : {chi2_history[best_iter-1]:.5f}")
    print(f"   Best Image RMSE   : {min(rmse_img_history):.4f}  (iter {np.argmin(rmse_img_history)+1})")
    print(f"   Best Image SSIM   : {max(ssim_img_history):.4f} (iter {np.argmax(ssim_img_history)+1})")

    return x_best, chi2_history, rmse_sino_history, rmse_img_history, ssim_img_history, delta_history, best_iter


# ─────────────────────────────────────────────────────────────
# 2. Run Optimization
# ─────────────────────────────────────────────────────────────
sino_denoised, chi2_history, rmse_sino_history, rmse_img_history, ssim_img_history, delta_history, best_iter = \
    wls_tv_chi2_slope_stop(
        sino_noisy, W_norm, W_raw, sino_clean, x_true, mask, angles,
        lam=0.02,
        alpha=0.5,
        max_iters=2000,
        slope_ratio_threshold=0.0035,
        slope_window=8,
        min_iters=20
    )

np.save('results/p4_results/sino_denoised.npy', sino_denoised)

rmse_sino_noisy = np.sqrt(np.mean((sino_noisy - sino_clean)**2))
print(f"\n✅ Phase 4 Complete")
print(f"   Noisy Sinogram RMSE : {rmse_sino_noisy:.4f}")
print(f"   Denoised Sinogram RMSE: {rmse_sino_history[best_iter-1]:.4f}")
print(f"   Improvement         : {((rmse_sino_noisy - rmse_sino_history[best_iter-1]) / rmse_sino_noisy * 100):.1f}%")

# ─────────────────────────────────────────────────────────────
# 3. Visualization (2x3 Layout)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Row 1: Sinograms & Chi2
axes[0,0].imshow(sino_noisy, cmap='gray', aspect='auto')
axes[0,0].set_title('Noisy Sinogram'); axes[0,0].axis('off')

axes[0,1].imshow(sino_denoised, cmap='gray', aspect='auto')
axes[0,1].set_title(f'Denoised (stopped iter {best_iter})'); axes[0,1].axis('off')

axes[0,2].plot(chi2_history, 'r-', linewidth=2, label='Chi²')
axes[0,2].axvline(x=best_iter-1, color='k', linestyle='--', label=f'Stop (iter {best_iter})')
axes[0,2].set_title('Chi² (Slope-Based Stop)'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

# Row 2: Metric Curves
axes[1,0].plot(rmse_sino_history, 'b-', linewidth=2, label='Sinogram RMSE')
axes[1,0].axvline(x=best_iter-1, color='k', linestyle='--', label=f'Stop (iter {best_iter})')
axes[1,0].set_title('Sinogram RMSE vs Iteration'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(rmse_img_history, 'g-', linewidth=2, label='Image RMSE')
axes[1,1].axvline(x=best_iter-1, color='k', linestyle='--', label=f'Stop (iter {best_iter})')
axes[1,1].set_title('Image RMSE (Reconstructed) vs Iteration'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

axes[1,2].plot(ssim_img_history, 'm-', linewidth=2, label='Image SSIM')
axes[1,2].axvline(x=best_iter-1, color='k', linestyle='--', label=f'Stop (iter {best_iter})')
axes[1,2].set_title('Image SSIM vs Iteration'); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p4_results/phase4_validation.png', dpi=150)
plt.show()

print("💾 Saved: results/p4_results/sino_denoised.npy & phase4_validation.png")