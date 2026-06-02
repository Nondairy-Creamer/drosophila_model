import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# --- model parameters ---
num_odors = 1000          # number of distinct odors
num_glom = 50            # number of glomeruli (~Drosophila has ~50)
n_trials = 5000         # presentations (each is one odor at one concentration)
noise_std = 0.02        # additive noise on glomerular drive
sparsity = 0.3          # fraction of glomeruli each odor activates strongly

# divisive normalization (PN-like lateral inhibition)
norm_sigma = 0.1

# concentrations span 3 orders of magnitude (log-uniform)
log_int_lo, log_int_hi = -1.5, 1.5

rng = np.random.default_rng(0)

# --- encoding matrix E (sparse, non-negative) ---
# Each odor activates a random subset of glomeruli with random non-negative weights.
# Sparsity matters: with dense E, all glomeruli saturate at high concentration and
# identity is lost in ORN; with sparse E, only some saturate and identity persists
# in the pattern.
E = np.abs(rng.normal(size=(num_odors, num_glom)))
E *= (rng.random(size=E.shape) < sparsity)

# --- generate trials: one odor per trial at log-uniform concentration ---
trial_odor = rng.integers(0, num_odors, size=n_trials)
trial_log_int = rng.uniform(log_int_lo, log_int_hi, size=n_trials)
trial_int = 10 ** trial_log_int

# concentration matrix: one row per trial, one-hot scaled by intensity
C = np.zeros((n_trials, num_odors))
C[np.arange(n_trials), trial_odor] = trial_int


# --- step 1: linear glomerular drive (oracle, biologically inaccessible) ---
drive = C @ E + rng.normal(size=(n_trials, num_glom)) * noise_std


# --- nonlinearities ---
def divisive_norm(x, sigma):
    pooled = x.mean(axis=1, keepdims=True)
    return x / (sigma + pooled)


# ORN: raw glomerular drive (rectified to non-negative firing rates)
orn = np.clip(drive, 0, None)

# PN: divisive normalization across glomeruli (lateral inhibition)
pn = divisive_norm(orn, norm_sigma)


# --- linear identity decoding: regress one-hot identity, argmax ---
# This is the actual task the antennal lobe is optimized for: tell *which* odor
# is present despite huge concentration variation.

idx = rng.permutation(n_trials)
n_train = n_trials // 2
train, test = idx[:n_train], idx[n_train:]

Y = np.eye(num_odors)[trial_odor]
true_class = trial_odor[test]
test_log_int = trial_log_int[test]


def fit_predict_logistic(X, y, train, test, n_classes, l2=1e-3, maxiter=300):
    Xb_train = np.hstack([X[train], np.ones((len(train), 1))])
    Xb_test = np.hstack([X[test], np.ones((len(test), 1))])
    Y_train = np.eye(n_classes)[y[train]]
    n, d = Xb_train.shape

    def loss_and_grad(w_flat):
        W = w_flat.reshape(d, n_classes)
        logits = Xb_train @ W
        logits -= logits.max(axis=1, keepdims=True)
        ez = np.exp(logits)
        probs = ez / ez.sum(axis=1, keepdims=True)
        ll = -np.mean(np.log(np.clip(probs[np.arange(n), y[train]], 1e-15, None)))
        reg = 0.5 * l2 * (W[:-1] ** 2).sum()  # don't regularize bias
        grad = Xb_train.T @ (probs - Y_train) / n
        grad[:-1] += l2 * W[:-1]
        return ll + reg, grad.flatten()

    w0 = np.zeros(d * n_classes)
    result = minimize(loss_and_grad, w0, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter})
    W = result.x.reshape(d, n_classes)
    return (Xb_test @ W).argmax(axis=1)


representations = [
    ('ORN (raw drive)', orn),
    ('PN (div norm)',   pn),
]

accs = {}
preds = {}
print('Identity classification accuracy (logistic regression):')
for name, X in representations:
    pred = fit_predict_logistic(X, trial_odor, train, test, num_odors)
    preds[name] = pred
    accs[name] = (pred == true_class).mean()
    print(f'  {name:28s} acc = {accs[name]:.3f}')


# --- accuracy as a function of concentration ---
n_bins = 8
edges = np.linspace(log_int_lo, log_int_hi, n_bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
bin_idx = np.clip(np.digitize(test_log_int, edges) - 1, 0, n_bins - 1)

per_bin_acc = {}
for name in accs:
    pred = preds[name]
    per_bin_acc[name] = np.array([
        (pred[bin_idx == b] == true_class[bin_idx == b]).mean()
        if (bin_idx == b).any() else np.nan
        for b in range(n_bins)
    ])


# --- plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = {'ORN (raw drive)': '#d35',
          'PN (div norm)':   '#3a9'}

axes[0].bar(accs.keys(), accs.values(), color=[colors[k] for k in accs])
axes[0].axhline(1 / num_odors, color='gray', ls='--', label='chance')
axes[0].set_ylabel('classification accuracy')
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Overall identity decoding (logistic regression)')
axes[0].legend(loc='lower right')

for name in accs:
    axes[1].plot(centers, per_bin_acc[name], 'o-',
                 label=name, color=colors[name])
axes[1].axhline(1 / num_odors, color='gray', ls='--')
axes[1].set_xlabel(r'$\log_{10}$ concentration')
axes[1].set_ylabel('accuracy')
axes[1].set_ylim(0, 1.05)
axes[1].set_title('Accuracy vs concentration')
axes[1].legend(loc='lower right', fontsize=8)

# show the response of all 50 glomeruli to one odor across concentrations,
# for each representation. This visualizes why PN helps.
demo_odor = 0
demo_concs = np.logspace(log_int_lo, log_int_hi, 50)
demo_C = np.zeros((len(demo_concs), num_odors))
demo_C[:, demo_odor] = demo_concs
demo_drive = demo_C @ E
demo_orn = np.clip(demo_drive, 0, None)
demo_pn = divisive_norm(demo_orn, norm_sigma)

axes[2].plot(demo_concs, demo_orn, color='#d35', alpha=0.4, lw=0.7)
axes[2].plot(demo_concs, demo_pn,  color='#3a9', alpha=0.4, lw=0.7)
axes[2].set_xscale('log')
axes[2].set_xlabel('concentration')
axes[2].set_ylabel('response per glomerulus')
axes[2].set_title(f'Odor {demo_odor}: each line is one glomerulus\n'
                  'red=ORN, teal=PN')

plt.tight_layout()


# --- PCA scatter: first two principal components of the glomerular signal ---
# Projects each representation onto its top two PCs so the odor geometry can be
# visualized for any number of glomeruli. Color encodes odor identity; this shows
# how divisive normalization reshapes the odor clusters relative to raw ORN drive.
def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


# restrict to a single concentration bin centered on concentration 1
# (log10 = 0), using the same bin width as the accuracy-vs-concentration plot
bin_width = (log_int_hi - log_int_lo) / n_bins
conc_mask = np.abs(trial_log_int) <= bin_width / 2

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
odor_colors = plt.cm.tab10(np.arange(num_odors) % 10)
for ax, (name, X) in zip(axes2, representations):
    proj = pca_2d(X)
    for o in range(num_odors):
        mask = (trial_odor == o) & conc_mask
        ax.scatter(proj[mask, 0], proj[mask, 1], s=12, alpha=0.5,
                   color=odor_colors[o], label=f'odor {o}')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(name)
axes2[0].legend(loc='best', fontsize=8, markerscale=1.5)
fig2.suptitle('Per-trial responses in top-2 PC space (color = odor)\n'
              f'concentration $\\approx$ 1 (bin width {bin_width:.3f} in $\\log_{{10}}$)')
fig2.tight_layout()

plt.show()
