# experiment_dose_sweep.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from skimage.restoration import denoise_tv_chambolle
import os

os.makedirs('results/experiments', exist_ok=True)

# 0. Setup Ground Truth
x_true_raw = resize(shepp_logan_phantom(), (512, 512), anti_aliasing=True, mode='constant')
angles = np.linspace(0, 180, 360, endpoint=False)
temp_sino = radon(x_true_raw, theta=angles, circle=True)
scaling_factor = 5.0 / np.max(temp_sino)
x_true = x_true_raw * scaling_factor
sino_clean = radon(x_true, theta=angles, circle=True)

# 1. Sweep Parameters
dose_levels = [500, 1000, 5000, 10000, 50000]  # 1% to 50% dose
lam = 0.02  # Fixed lambda for fair comparison
alpha = 0.5
results = {'dose': [], 'rmse_noisy': [], 'rmse_denoised': [], 'ssim_noisy': [], 'ssim_denoised': []}

# 2. Run Sweep
for I0 in dose_levels:
    print(f"\n--- Dose Level: I0={I0} ---")
    
    # Phase 2: Simulate
    rng = np.random.default_rng(42)
    expected = I0 * np.exp(-sino_clean)
    counts = np.clip(rng.poisson(expected).astype(float) + rng.normal(0, 5.0, size=expected.shape), 1.0, None)
    sino_noisy = -np.log(counts / I0)
    
    # Phase 3: Weights
    var_log = (counts + 5.0**2) / (counts**2)
    W_raw = 1.0 / var_log
    W_norm = W_raw / (W_raw.max() + 1e-8)
    
    # Phase 4: Optimize (Slope-based stopping)
    x = sino_noisy.copy()
    chi2_hist, rmse_hist = [], []
    initial_slope, best_iter = None, 0
    
    for i in range(200):
        x_old = x.copy()
        grad = W_norm * (x - sino_noisy)
        x = denoise_tv_chambolle(x - alpha * grad, weight=alpha * lam, max_num_iter=20, eps=1e-4)
        x = np.clip(x, 0.0, sino_noisy.max())
        
        chi2 = np.mean(W_raw * (x - sino_noisy)**2)
        rmse = np.sqrt(np.mean((x - sino_clean)**2))
        chi2_hist.append(chi2)
        rmse_hist.append(rmse)
        
        if i == 2: initial_slope = chi2_hist[2] - chi2_hist[1]
        if initial_slope and i >= 20:
            slope = np.polyfit(np.arange(8), chi2_hist[-8:], 1)[0]
            if abs(slope / initial_slope) < 0.005:
                best_iter = i + 1
                break
    
    # Reconstruct & Metrics
    mask = (np.ogrid[:512, :512][0] - 255.5)**2 + (np.ogrid[:512, :512][1] - 255.5)**2 <= 255**2
    x_noisy = iradon(sino_noisy, theta=angles, filter_name='ramp', circle=True)
    x_denoised = iradon(x, theta=angles, filter_name='ramp', circle=True)
    
    results['dose'].append(I0)
    results['rmse_noisy'].append(np.sqrt(mean_squared_error(x_true[mask], x_noisy[mask])))
    results['rmse_denoised'].append(np.sqrt(mean_squared_error(x_true[mask], x_denoised[mask])))
    results['ssim_noisy'].append(ssim(x_true, x_noisy, data_range=x_true.max()))
    results['ssim_denoised'].append(ssim(x_true, x_denoised, data_range=x_true.max()))
    
    print(f"   Improvement: RMSE ↓{((results['rmse_noisy'][-1]-results['rmse_denoised'][-1])/results['rmse_noisy'][-1]*100):.1f}%")

# 3. Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogx(results['dose'], results['rmse_noisy'], 'bo-', label='Noisy FBP')
axes[0].semilogx(results['dose'], results['rmse_denoised'], 'rs-', label='WLS-TV [Ours]')
axes[0].set_xlabel('Incident Photons (I0)'); axes[0].set_ylabel('Image RMSE')
axes[0].set_title('Dose Robustness: RMSE vs I0'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].semilogx(results['dose'], results['ssim_noisy'], 'bo-', label='Noisy FBP')
axes[1].semilogx(results['dose'], results['ssim_denoised'], 'rs-', label='WLS-TV [Ours]')
axes[1].set_xlabel('Incident Photons (I0)'); axes[1].set_ylabel('Image SSIM')
axes[1].set_title('Dose Robustness: SSIM vs I0'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/experiments/dose_sweep.png', dpi=150)
plt.show()