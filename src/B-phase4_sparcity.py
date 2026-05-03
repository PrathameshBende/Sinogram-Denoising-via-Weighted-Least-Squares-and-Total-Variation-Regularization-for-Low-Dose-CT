import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error
import os

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0. Load Data
# ─────────────────────────────────────────────────────────────
sino_noisy  = np.load('results/p2_results/sino_noisy.npy')
raw_counts  = np.load('results/p2_results/raw_counts.npy')
sino_clean  = np.load('results/p2_results/sino_clean.npy')
x_true      = np.load('results/p2_results/x_true.npy')
angles      = np.linspace(0, 180, 360, endpoint=False)

# ─────────────────────────────────────────────────────────────
# 1. Weight Matrix
# ─────────────────────────────────────────────────────────────
y_counts = np.clip(raw_counts, 1.0, None)
sigma_e  = 5.0
var_log  = (y_counts + sigma_e**2) / (y_counts**2)
W        = 1.0 / var_log
W        = W / (W.max() + 1e-8)

# ─────────────────────────────────────────────────────────────
# 2. Patch Extraction
# ─────────────────────────────────────────────────────────────
def extract_patches(img, weight_map, patch_size=8, stride=4):
    patches, w_patches, positions, patch_means = [], [], [], []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            p  = img[i:i+patch_size, j:j+patch_size].flatten()
            m  = p.mean()
            patches.append(p - m)
            patch_means.append(m)
            wp = weight_map[i:i+patch_size, j:j+patch_size].flatten()
            w_patches.append(wp.mean())
            positions.append((i, j))
    return (np.array(patches), np.array(w_patches), positions, np.array(patch_means))

def reconstruct(patches, patch_means, positions, shape, patch_size=8):
    img   = np.zeros(shape)
    count = np.zeros(shape)
    for patch, mean, (i, j) in zip(patches, patch_means, positions):
        p = patch.reshape(patch_size, patch_size) + mean
        img[i:i+patch_size, j:j+patch_size]   += p
        count[i:i+patch_size, j:j+patch_size] += 1
    return img / (count + 1e-8)

patch_size = 8
patches, patch_weights, positions, patch_means = extract_patches(sino_noisy, W, patch_size)
Y = patches.copy()
print(f"Patches: {Y.shape[0]:,} of size {patch_size}x{patch_size}")

# ─────────────────────────────────────────────────────────────
# 3. Overcomplete DCT Dictionary  (built once, reused for all lambda)
# ─────────────────────────────────────────────────────────────
n = patches.shape[1]    # 64
K = 4 * n               # 256 atoms

def make_overcomplete_dct(n, K):
    i  = np.arange(n)
    ks = np.arange(K)
    D  = np.cos(np.pi * ks[None, :] * (2 * i[:, None] + 1) / (2 * K))
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return D / norms

print("Building overcomplete DCT dictionary ...", end=" ")
D   = make_overcomplete_dct(n, K)
DtD = D.T @ D
DtY = D.T @ Y.T
L   = np.linalg.norm(DtD, 2)
print(f"done  ({n} x {K}, x{K//n})")

# ─────────────────────────────────────────────────────────────
# 4. Weighted ISTA
# ─────────────────────────────────────────────────────────────
def weighted_ista(DtD, DtY, L, patch_weights, lam, iters=120):
    K, N   = DtY.shape
    X      = np.zeros((K, N))
    Wrow   = patch_weights[None, :]
    thresh = lam / (patch_weights * L + 1e-8)
    step   = 1.0 / (Wrow * L + 1e-8)
    for _ in range(iters):
        grad = Wrow * (DtD @ X - DtY)
        X    = X - step * grad
        X    = np.sign(X) * np.maximum(np.abs(X) - thresh[None, :], 0.0)
    return X

# ─────────────────────────────────────────────────────────────
# 5. Baseline reconstructions (computed once)
# ─────────────────────────────────────────────────────────────
x_noisy  = iradon(sino_noisy, theta=angles, filter_name='ramp', circle=True)
x_oracle = iradon(sino_clean, theta=angles, filter_name='ramp', circle=True)

def image_metrics(x_est):
    yg, xg = np.ogrid[:512, :512]
    mask   = (xg - 255.5)**2 + (yg - 255.5)**2 <= 255**2
    rmse   = np.sqrt(mean_squared_error(x_true[mask], x_est[mask]))
    psnr_v = psnr(x_true[mask], x_est[mask], data_range=x_true.max())
    ssim_v = ssim(x_true[mask], x_est[mask], data_range=x_true.max())
    return rmse, psnr_v, ssim_v

r_noisy, p_noisy, s_noisy = image_metrics(x_noisy)
rmse_noisy_sino = np.sqrt(np.mean((sino_noisy - sino_clean)**2))

# ─────────────────────────────────────────────────────────────
# 6. Lambda sweep
# ─────────────────────────────────────────────────────────────
LAMBDAS = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]
ITERS   = 120

results = []

print(f"\n{'─'*75}")
print(f"  {'lam':>6}  {'Sparsity':>9}  {'Sino-RMSE':>10}  "
      f"{'RMSE':>7}  {'PSNR':>7}  {'SSIM':>7}  {'Improv%':>8}")
print(f"{'─'*75}")

best_ssim_result = None
best_rmse_result = None

for lam in LAMBDAS:
    X           = weighted_ista(DtD, DtY, L, patch_weights, lam=lam, iters=ITERS)
    patches_rec = (D @ X).T
    sino_sp     = reconstruct(patches_rec, patch_means, positions, sino_noisy.shape, patch_size)

    sparsity    = 100.0 * np.mean(np.abs(X) < 1e-8)
    rmse_sino   = np.sqrt(np.mean((sino_sp - sino_clean)**2))

    x_sp              = iradon(sino_sp, theta=angles, filter_name='ramp', circle=True)
    rmse, psnr_v, ssim_v = image_metrics(x_sp)
    improvement       = (r_noisy - rmse) / r_noisy * 100

    row = dict(lam=lam, sparsity=sparsity, sino_rmse=rmse_sino,
               rmse=rmse, psnr=psnr_v, ssim=ssim_v,
               improvement=improvement, x_sp=x_sp, sino_sp=sino_sp)
    results.append(row)

    print(f"  {lam:>6.3f}  {sparsity:>8.1f}%  {rmse_sino:>10.5f}  "
          f"{rmse:>7.4f}  {psnr_v:>7.2f}  {ssim_v:>7.4f}  {improvement:>7.1f}%")

    if best_ssim_result is None or ssim_v > best_ssim_result['ssim']:
        best_ssim_result = row
    if best_rmse_result is None or rmse < best_rmse_result['rmse']:
        best_rmse_result = row

print(f"{'─'*75}")
print(f"\n  Best SSIM : lam={best_ssim_result['lam']:.3f}  "
      f"-> SSIM={best_ssim_result['ssim']:.4f}  RMSE={best_ssim_result['rmse']:.4f}")
print(f"  Best RMSE : lam={best_rmse_result['lam']:.3f}  "
      f"-> RMSE={best_rmse_result['rmse']:.4f}  SSIM={best_rmse_result['ssim']:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. Figures
# ─────────────────────────────────────────────────────────────
lams       = [r['lam']       for r in results]
rmses      = [r['rmse']      for r in results]
ssims      = [r['ssim']      for r in results]
sparses    = [r['sparsity']  for r in results]
sino_rmses = [r['sino_rmse'] for r in results]

# Figure 1: sweep summary
fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))
fig1.suptitle('Lambda Sweep — Weighted ISTA (fixed overcomplete DCT)', fontsize=13)

axes1[0].semilogx(lams, rmses, 'b-o', linewidth=2)
axes1[0].axvline(best_rmse_result['lam'], color='b', linestyle=':', alpha=0.7,
                 label=f"Best RMSE lam={best_rmse_result['lam']}")
axes1[0].axvline(best_ssim_result['lam'], color='r', linestyle='--', alpha=0.8,
                 label=f"Best SSIM lam={best_ssim_result['lam']}")
axes1[0].set_title('Image RMSE vs lam'); axes1[0].set_xlabel('lambda')
axes1[0].set_ylabel('RMSE'); axes1[0].legend(fontsize=8); axes1[0].grid(True, alpha=0.3)

axes1[1].semilogx(lams, ssims, 'r-o', linewidth=2)
axes1[1].axvline(best_ssim_result['lam'], color='r', linestyle='--', alpha=0.8,
                 label=f"Best lam={best_ssim_result['lam']}")
axes1[1].set_title('Image SSIM vs lam'); axes1[1].set_xlabel('lambda')
axes1[1].set_ylabel('SSIM'); axes1[1].legend(fontsize=8); axes1[1].grid(True, alpha=0.3)

axes1[2].semilogx(lams, sino_rmses, 'g-o', linewidth=2)
axes1[2].axhline(rmse_noisy_sino, color='k', linestyle='--', label='Noisy baseline')
axes1[2].set_title('Sinogram RMSE vs lam'); axes1[2].set_xlabel('lambda')
axes1[2].set_ylabel('Sino-RMSE'); axes1[2].legend(fontsize=8); axes1[2].grid(True, alpha=0.3)

axes1[3].semilogx(lams, sparses, 'm-o', linewidth=2)
axes1[3].set_title('Sparsity % vs lam'); axes1[3].set_xlabel('lambda')
axes1[3].set_ylabel('Sparsity (%)'); axes1[3].grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig('results/lambda_sweep.png', dpi=150)
print("Saved: results/lambda_sweep.png")

# Figure 2: best-lambda image comparison
best   = best_ssim_result
x_best = best['x_sp']

fig2, axes2 = plt.subplots(2, 4, figsize=(22, 11))
fig2.suptitle(f"Best Result  (lam={best['lam']:.3f}, best SSIM)", fontsize=13)

axes2[0,0].imshow(x_true,   cmap='gray'); axes2[0,0].set_title('Ground Truth');         axes2[0,0].axis('off')
axes2[0,1].imshow(x_oracle, cmap='gray'); axes2[0,1].set_title('Oracle FBP');            axes2[0,1].axis('off')
axes2[0,2].imshow(x_noisy,  cmap='gray'); axes2[0,2].set_title('Noisy FBP (baseline)'); axes2[0,2].axis('off')
axes2[0,3].imshow(x_best,   cmap='gray')
axes2[0,3].set_title(f"Sparse Dict [Ours]  lam={best['lam']:.3f}\n"
                     f"PSNR: {best['psnr']:.2f} dB | SSIM: {best['ssim']:.4f}")
axes2[0,3].axis('off')

vmax = 0.15
im1  = axes2[1,0].imshow(np.abs(x_true - x_noisy), cmap='hot', vmin=0, vmax=vmax)
axes2[1,0].set_title('Error: Noisy FBP'); axes2[1,0].axis('off')
plt.colorbar(im1, ax=axes2[1,0], fraction=0.046)

im2  = axes2[1,1].imshow(np.abs(x_true - x_best),  cmap='hot', vmin=0, vmax=vmax)
axes2[1,1].set_title(f"Error: Sparse [Ours]  lam={best['lam']:.3f}"); axes2[1,1].axis('off')
plt.colorbar(im2, ax=axes2[1,1], fraction=0.046)

row = 256
axes2[1,2].plot(x_true[row],  'k-',  linewidth=2,  label='Ground Truth')
axes2[1,2].plot(x_noisy[row], 'b--', alpha=0.6,     label='Noisy FBP')
axes2[1,2].plot(x_best[row],  'r-',  linewidth=1.5, label=f"Sparse lam={best['lam']:.3f}")
axes2[1,2].set_title(f'Image Profile (Row {row})')
axes2[1,2].legend(); axes2[1,2].grid(True, alpha=0.3)

sc = axes2[1,3].scatter(rmses, ssims, c=lams, cmap='plasma', s=100, zorder=3,
                        norm=plt.matplotlib.colors.LogNorm())
for r in results:
    axes2[1,3].annotate(f"lam={r['lam']}", (r['rmse'], r['ssim']),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)
axes2[1,3].scatter([best['rmse']], [best['ssim']], s=200, marker='*',
                   color='red', zorder=5, label='Best SSIM')
plt.colorbar(sc, ax=axes2[1,3], label='lambda')
axes2[1,3].set_xlabel('Image RMSE'); axes2[1,3].set_ylabel('SSIM')
axes2[1,3].set_title('RMSE-SSIM Trade-off'); axes2[1,3].legend(fontsize=8)
axes2[1,3].grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig('results/advanced_sparse.png', dpi=150)
print("Saved: results/advanced_sparse.png")

plt.show()