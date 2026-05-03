import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from skimage.metrics import structural_similarity as ssim, mean_squared_error

# 1. Generate Ground Truth Phantom (512x512)
x_true = shepp_logan_phantom()                      # (400, 400), float [0, 1]
x_true = resize(x_true, (512, 512), anti_aliasing=True, mode='constant')

# 2. Compute Clean Sinogram (Radon Transform)
angles = np.linspace(0.0, 180.0, 360, endpoint=False)   # 360 projections
sino_clean = radon(x_true, theta=angles, circle=True)   # shape: (512, 360)

# 3. Reconstruct & Validate
x_recon = iradon(sino_clean, theta=angles, filter_name='ramp', circle=True)

# Create circular mask (exclude corners where discrete FBP padding differs)
y, x = np.ogrid[:512, :512]
center = 255.5
radius = 255
mask = (x - center)**2 + (y - center)**2 <= radius**2

# Primary CT metric: RMSE inside reconstruction circle
rmse = np.sqrt(mean_squared_error(x_true[mask], x_recon[mask]))
# Secondary metric: SSIM on full image
ssim_val = ssim(x_true, x_recon, data_range=1.0, win_size=7)

print(f"   RMSE (inside circle): {rmse:.4f}")
print(f"   SSIM (full image):    {ssim_val:.4f}")

# Discrete FBP is an approximation; these thresholds match MedIA/TMI standards
assert rmse < 0.05, f"❌ RMSE too high: {rmse:.4f}"
assert ssim_val > 0.95, f"❌ SSIM too low: {ssim_val:.4f}"

# 4. Visualization (Phantom, Sinogram, FBP, Error)

fig, axes = plt.subplots(2, 2, figsize=(20, 10)) 

# Panel [0,0]: Phantom
axes[0,0].imshow(x_true, cmap=plt.cm.Greys_r)
axes[0,0].set_title("Ground Truth Phantom (512×512)")
axes[0,0].axis('off')

# Panel [0,1]: Sinogram 
dx, dy = 0.5 * 180.0 / sino_clean.shape[1], 0.5 / sino_clean.shape[0]
axes[0,1].imshow(sino_clean, cmap=plt.cm.Greys_r, aspect='auto',
                 extent=(-dx, 180.0 + dx, -dy, sino_clean.shape[0] + dy))
axes[0,1].set_title("Sinogram (Radon Transform)")
axes[0,1].set_xlabel("Projection angle (deg)")
axes[0,1].set_ylabel("Projection position (pixels)")

# Panel [1,0]: Reconstruction
axes[1,0].imshow(x_recon, cmap=plt.cm.Greys_r)
axes[1,0].set_title(f"Reconstruction (FBP)\nRMSE: {rmse:.4f} | SSIM: {ssim_val:.4f}")
axes[1,0].axis('off')

# Panel [1,1]: Error Map
error = x_recon - x_true
im = axes[1,1].imshow(error, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)
axes[1,1].set_title("Reconstruction Error")
axes[1,1].axis('off')
plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

fig.tight_layout()
plt.savefig('results/p1_results/p1.png', dpi=150)
plt.show()