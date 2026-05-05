# experiment_sparse_view.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from skimage.restoration import denoise_tv_chambolle
import os

os.makedirs('results/experiments', exist_ok=True)

# Setup
x_true_raw = resize(shepp_logan_phantom(), (512, 512), anti_aliasing=True, mode='constant')
temp_sino = radon(x_true_raw, theta=np.linspace(0, 180, 360), circle=True)
scaling_factor = 5.0 / np.max(temp_sino)
x_true = x_true_raw * scaling_factor

views = [360, 180, 90, 60, 30]  # Dense to ultra-sparse
lam = 0.055
alpha = 0.5
results = {'views': [], 'rmse_fbp': [], 'rmse_wls': [], 'ssim_fbp': [], 'ssim_wls': []}

for n_views in views:
    print(f"\n--- Views: {n_views} ---")
    angles = np.linspace(0, 180, n_views, endpoint=False)
    sino_clean = radon(x_true, theta=angles, circle=True)
    
    # Simulate noise
    rng = np.random.default_rng(42)
    expected = 1e3 * np.exp(-sino_clean)
    counts = np.clip(rng.poisson(expected).astype(float) + rng.normal(0, 5.0, size=expected.shape), 1.0, None)
    sino_noisy = -np.log(counts / 1e3)
    
    # Weights & Optimization
    var_log = (counts + 5.0**2) / (counts**2)
    W_norm = (1.0 / var_log) / ((1.0 / var_log).max() + 1e-8)
    
    x = sino_noisy.copy()
    for i in range(100):  # Fixed iters for sweep speed
        grad = W_norm * (x - sino_noisy)
        x = denoise_tv_chambolle(x - alpha * grad, weight=alpha * lam, max_num_iter=10, eps=1e-4)
        x = np.clip(x, 0.0, sino_noisy.max())
        
    # Reconstruct
    mask = (np.ogrid[:512, :512][0] - 255.5)**2 + (np.ogrid[:512, :512][1] - 255.5)**2 <= 255**2
    x_fbp = iradon(sino_noisy, theta=angles, filter_name='ramp', circle=True)
    x_wls = iradon(x, theta=angles, filter_name='ramp', circle=True)
    
    results['views'].append(n_views)
    results['rmse_fbp'].append(np.sqrt(mean_squared_error(x_true[mask], x_fbp[mask])))
    results['rmse_wls'].append(np.sqrt(mean_squared_error(x_true[mask], x_wls[mask])))
    results['ssim_fbp'].append(ssim(x_true, x_fbp, data_range=x_true.max()))
    results['ssim_wls'].append(ssim(x_true, x_wls, data_range=x_true.max()))
    
    print(f"   FBP SSIM: {results['ssim_fbp'][-1]:.4f} | WLS-TV SSIM: {results['ssim_wls'][-1]:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(results['views'], results['rmse_fbp'], 'bo-', label='FBP')
axes[0].plot(results['views'], results['rmse_wls'], 'rs-', label='WLS-TV')
axes[0].set_xlabel('Number of Projections'); axes[0].set_ylabel('Image RMSE')
axes[0].set_title('Sparse-View Robustness: RMSE'); axes[0].invert_xaxis(); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(results['views'], results['ssim_fbp'], 'bo-', label='FBP')
axes[1].plot(results['views'], results['ssim_wls'], 'rs-', label='WLS-TV')
axes[1].set_xlabel('Number of Projections'); axes[1].set_ylabel('Image SSIM')
axes[1].set_title('Sparse-View Robustness: SSIM'); axes[1].invert_xaxis(); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/experiments/sparse_view.png', dpi=150)
plt.show()