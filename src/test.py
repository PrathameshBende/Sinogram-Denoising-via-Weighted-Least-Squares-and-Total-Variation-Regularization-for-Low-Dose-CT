import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

# ─────────────────────────────────────────────
# PHASE 1: DATA MODELING
# ─────────────────────────────────────────────

x_true = resize(shepp_logan_phantom(), (256, 256))
angles = np.linspace(0, 180, 180, endpoint=False)

sino_clean = radon(x_true, theta=angles, circle=True)

I0 = 5e4
counts = I0 * np.exp(-sino_clean)
noisy_counts = np.random.poisson(counts)

sino_noisy = -np.log(np.clip(noisy_counts, 1, None) / I0)

# ─────────────────────────────────────────────
# PATCH EXTRACTION
# ─────────────────────────────────────────────

def extract_patches(img, patch_size=8, stride=2):
    patches = []
    positions = []
    means = []
    stds = []

    h, w = img.shape

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]

            m = np.mean(patch)
            s = np.std(patch) + 1e-8

            patch_norm = (patch - m) / s

            patches.append(patch_norm.flatten())
            positions.append((i, j))
            means.append(m)
            stds.append(s)

    return np.array(patches), positions, np.array(means), np.array(stds)

def reconstruct(patches, positions, means, stds, shape, patch_size=8):
    img = np.zeros(shape)
    count = np.zeros(shape)

    for idx, (i, j) in enumerate(positions):
        patch = patches[idx].reshape(patch_size, patch_size)

        # denormalize
        patch = patch * stds[idx] + means[idx]

        img[i:i+patch_size, j:j+patch_size] += patch
        count[i:i+patch_size, j:j+patch_size] += 1

    return img / (count + 1e-8)

patch_size = 8
stride = 2

patches, positions, means, stds = extract_patches(sino_noisy, patch_size, stride)

# shape: (num_patches, patch_dim) → transpose for math
X = patches.T

# ─────────────────────────────────────────────
# DICTIONARY (DCT-based)
# ─────────────────────────────────────────────

def create_dct_dictionary(n):
    D = np.zeros((n, n))
    for k in range(n):
        v = np.cos(np.arange(n) * k * np.pi / n)
        D[:, k] = v
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-8
    return D

n = X.shape[0]
D = create_dct_dictionary(n)

# ─────────────────────────────────────────────
# ISTA (L1 SPARSE CODING)
# ─────────────────────────────────────────────

def ista(D, X, lam=0.005, iters=10):
    DtD = D.T @ D
    DtX = D.T @ X

    L = np.linalg.norm(DtD, 2)
    t = 1.0 / (L + 1e-8)

    R = np.zeros((D.shape[1], X.shape[1]))

    for _ in range(iters):
        grad = DtD @ R - DtX
        R = R - t * grad
        R = np.sign(R) * np.maximum(np.abs(R) - lam * t, 0)

    return R

R = ista(D, X, lam=0.005, iters=10)

# ─────────────────────────────────────────────
# RECONSTRUCT SINOGRAM
# ─────────────────────────────────────────────

patches_rec = (D @ R).T

sino_denoised = reconstruct(
    patches_rec,
    positions,
    means,
    stds,
    sino_noisy.shape,
    patch_size
)

# ─────────────────────────────────────────────
# FBP RECONSTRUCTION
# ─────────────────────────────────────────────

x_noisy = iradon(sino_noisy, theta=angles, filter_name='ramp', circle=True)
x_denoised = iradon(sino_denoised, theta=angles, filter_name='ramp', circle=True)

# ─────────────────────────────────────────────
# SNR
# ─────────────────────────────────────────────

def compute_snr(x):
    return np.mean(x) / (np.std(x) + 1e-8)

snr_noisy = compute_snr(x_noisy)
snr_denoised = compute_snr(x_denoised)

print("\n📊 RESULTS")
print(f"Noisy SNR:    {snr_noisy:.3f}")
print(f"Denoised SNR: {snr_denoised:.3f}")

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15,5))

axes[0].imshow(x_true, cmap='gray')
axes[0].set_title("Ground Truth")
axes[0].axis('off')

axes[1].imshow(x_noisy, cmap='gray')
axes[1].set_title(f"Noisy FBP\nSNR={snr_noisy:.2f}")
axes[1].axis('off')

axes[2].imshow(x_denoised, cmap='gray')
axes[2].set_title(f"Sparse Denoised FBP\nSNR={snr_denoised:.2f}")
axes[2].axis('off')

plt.tight_layout()
plt.show()