import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os

os.makedirs('results', exist_ok=True)


# Generate Ground Truth with PHYSICAL SCALING
x_true_raw = resize(shepp_logan_phantom(), (512, 512), anti_aliasing=True, mode='constant')
angles = np.linspace(0, 180, 360, endpoint=False)


temp_sino = radon(x_true_raw, theta=angles, circle=True)
target_max_line_integral = 5.0
scaling_factor = target_max_line_integral / np.max(temp_sino)
x_true = x_true_raw * scaling_factor

sino_clean = radon(x_true, theta=angles, circle=True)

print(f"Physical Scaling Applied:")
print(f"   Scaling factor: {scaling_factor:.4f}")
print(f"   I0: 5e4 photons | rho_e: {5.0}")

# Low-Dose Physics Simulation
def simulate_low_dose(sino_clean, I0=5e4, sigma_e=5.0, seed=42):
    """
    Simulates photon-counting detector measurements.
    Model: y ~ Poisson(I0 * exp(-l)) + N(0, sigma_e^2)
    """
    rng = np.random.default_rng(seed)
    
    # Beer-Lambert law: expected photon transmission
    expected_counts = I0 * np.exp(-sino_clean)
    
    # Add Poisson quantum noise
    noisy_counts = rng.poisson(expected_counts).astype(float)
    
    # Add Gaussian electronic readout noise
    noisy_counts += rng.normal(0, sigma_e, size=noisy_counts.shape)
    
    # Clip to avoid log(0)
    noisy_counts = np.clip(noisy_counts, 1.0, None)
    
    # Convert back to log-domain sinogram
    sino_noisy = -np.log(noisy_counts / I0)
    
    return sino_noisy, noisy_counts, expected_counts

# Run simulation
sino_noisy, raw_counts, expected_counts = simulate_low_dose(sino_clean, I0=1e3, sigma_e=5.0)


# Validation
noise_diff = sino_noisy - sino_clean
high_att_mask = sino_clean > 2.0
low_att_mask = sino_clean < 0.5

std_high = np.std(noise_diff[high_att_mask])
std_low = np.std(noise_diff[low_att_mask])

print(f"\nLow-Dose Simulation")
print(f"   Noise std (high attenuation): {std_high:.4f}")
print(f"   Noise std (low attenuation):  {std_low:.4f}")
assert std_high > std_low, "Heteroscedasticity check failed"

# Baseline reconstruction
x_noisy = iradon(sino_noisy, theta=angles, filter_name='ramp', circle=True)
y_grid, x_grid = np.ogrid[:512, :512]
mask = (x_grid - 255.5)**2 + (y_grid - 255.5)**2 <= 255**2

rmse_noisy = np.sqrt(np.mean((x_true[mask] - x_noisy[mask])**2))
ssim_noisy = ssim(x_true, x_noisy, data_range=x_true.max())
print(f"   Baseline FBP RMSE: {rmse_noisy:.4f} | SSIM: {ssim_noisy:.4f}")

# Save & Visualize
np.save('results/p2_results/x_true.npy', x_true)
np.save('results/p2_results/sino_clean.npy', sino_clean)
np.save('results/p2_results/sino_noisy.npy', sino_noisy)
np.save('results/p2_results/raw_counts.npy', raw_counts)
np.save('results/p2_results/scaling_factor.npy', np.array(scaling_factor))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(x_true, cmap='gray')
axes[0,0].set_title(f'Ground Truth (Scaled)\nMax μ: {x_true.max():.3f}')
axes[0,0].axis('off')

axes[0,1].imshow(sino_clean, cmap='gray', aspect='auto')
axes[0,1].set_title(f'Clean Sinogram\nMax l: {sino_clean.max():.2f}')
axes[0,1].axis('off')

axes[0,2].imshow(sino_noisy, cmap='gray', aspect='auto')
axes[0,2].set_title('Noisy Sinogram (I0=5e4)')
axes[0,2].axis('off')

axes[1,0].imshow(x_noisy, cmap='gray')
axes[1,0].set_title(f'Noisy FBP\nRMSE: {rmse_noisy:.4f}')
axes[1,0].axis('off')

axes[1,1].imshow(noise_diff, cmap='bwr', aspect='auto')
axes[1,1].set_title('Noise Residual (ỹ - l)')
axes[1,1].axis('off')

axes[1,2].plot(sino_clean.ravel(), noise_diff.ravel(), 'k.', alpha=0.1)
axes[1,2].set_xlabel('Clean Sinogram Value (l)')
axes[1,2].set_ylabel('Noise Residual')
axes[1,2].set_title('Heteroscedastic Noise')
axes[1,2].axhline(0, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('results\p2_results\phase2_validation.png', dpi=150)
plt.show()