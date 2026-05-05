# experiment_filter_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import os

os.makedirs('results/experiments', exist_ok=True)

# Load data
x_true = np.load('results/p2_results/x_true.npy')
sino_noisy = np.load('results/p2_results/sino_noisy.npy')
sino_denoised = np.load('results/p4_results/sino_denoised.npy')
angles = np.linspace(0, 180, 360, endpoint=False)
mask = (np.ogrid[:512, :512][0] - 255.5)**2 + (np.ogrid[:512, :512][1] - 255.5)**2 <= 255**2

filters = ['ramp', 'hann', 'hamming', 'cosine']
results = {'filter': [], 'rmse_noisy': [], 'rmse_denoised': []}

for filt in filters:
    print(f"--- Filter: {filt} ---")
    x_noisy = iradon(sino_noisy, theta=angles, filter_name=filt, circle=True)
    x_denoised = iradon(sino_denoised, theta=angles, filter_name=filt, circle=True)
    
    results['filter'].append(filt)
    results['rmse_noisy'].append(np.sqrt(mean_squared_error(x_true[mask], x_noisy[mask])))
    results['rmse_denoised'].append(np.sqrt(mean_squared_error(x_true[mask], x_denoised[mask])))

# Plot
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(filters))
width = 0.35
plt.bar(x_pos - width/2, results['rmse_noisy'], width, label='Noisy FBP', color='lightcoral')
plt.bar(x_pos + width/2, results['rmse_denoised'], width, label='WLS-TV + FBP', color='skyblue')
plt.xticks(x_pos, filters)
plt.ylabel('Image RMSE')
plt.title('Filter Comparison: Raw vs Denoised')
plt.legend(); plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/experiments/filter_comparison.png', dpi=150)
plt.show()