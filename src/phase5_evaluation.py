import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.metrics import structural_similarity as ssim, mean_squared_error

x_true = np.load('results/p2_results/x_true.npy')
sino_clean = np.load('results/p2_results/sino_clean.npy')
sino_noisy = np.load('results/p2_results/sino_noisy.npy')
sino_denoised = np.load('results/p4_results/sino_denoised.npy')

angles = np.linspace(0, 180, 360, endpoint=False)

print("Reconstructing Images...")

x_oracle = iradon(sino_clean, theta=angles, filter_name='ramp', circle=True)
x_noisy = iradon(sino_noisy, theta=angles, filter_name='ramp', circle=True)
x_ours = iradon(sino_denoised, theta=angles, filter_name='ramp', circle=True)


# Compute Metrics
def evaluate(name, x_est):
    y_grid, x_grid = np.ogrid[:512, :512]
    mask = (x_grid - 255.5)**2 + (y_grid - 255.5)**2 <= 255**2
    
    rmse = np.sqrt(mean_squared_error(x_true[mask], x_est[mask]))
    ssim_val = ssim(x_true[mask], x_est[mask], data_range=x_true.max())
    
    print(f"{name:<35} | RMSE: {rmse:.4f} | SSIM: {ssim_val:.4f}")
    return rmse, ssim_val


print("FINAL EVALUATION RESULTS")

evaluate("Oracle FBP", x_oracle)
evaluate("Raw Noisy FBP", x_noisy)
evaluate("WLS-TV [Ours] ", x_ours)


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0,0].imshow(x_oracle, cmap='gray')
axes[0,0].set_title('Oracle (Clean FBP)'); axes[0,0].axis('off')

axes[0,1].imshow(x_noisy, cmap='gray')
axes[0,1].set_title('Raw Noisy FBP'); axes[0,1].axis('off')

axes[0,2].imshow(x_ours, cmap='gray')
axes[0,2].set_title('WLS-TV [Ours]'); axes[0,2].axis('off')

# Error maps
error_noisy = np.abs(x_true - x_noisy)
error_ours = np.abs(x_true - x_ours)

im1 = axes[1,0].imshow(error_noisy, cmap='hot', vmin=0, vmax=0.1)
axes[1,0].set_title('Error: Noisy FBP'); axes[1,0].axis('off')
plt.colorbar(im1, ax=axes[1,0], fraction=0.046)

im2 = axes[1,1].imshow(error_ours, cmap='hot', vmin=0, vmax=0.1)
axes[1,1].set_title('Error: WLS-TV [Ours]'); axes[1,1].axis('off')
plt.colorbar(im2, ax=axes[1,1], fraction=0.046)

axes[1,2].plot(x_true[256, :], 'k-', label='Ground Truth', linewidth=2)
axes[1,2].plot(x_noisy[256, :], 'b--', label='Noisy', alpha=0.5)
axes[1,2].plot(x_ours[256, :], 'r-', label='WLS-TV [Ours]', linewidth=1.5)
axes[1,2].set_title('Image Profile (Row 256)')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p5_results/final_results.png', dpi=150)
plt.show()

print("\nSaved final_results.png")