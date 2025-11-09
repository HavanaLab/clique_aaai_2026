import numpy as np
from sklearn.decomposition import SparsePCA


def run_sklearn_sparse_pca(X, k, true_idx, n_components=1,
                           alpha=1, ridge_alpha=0.01, tol=1e-6):
    """
    Fit scikit-learn SparsePCA and compare recovered support to ground truth.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix.
    k : int
        Desired sparsity level (only used for metrics; the solver controls
        sparsity via 'alpha').
    true_idx : array-like of int
        Ground-truth support indices.
    n_components : int, default 1
        How many sparse PCs to fit.
    alpha : float, default 1
        L1 penalty strength (higher → sparser).
    ridge_alpha : float, default 0.01
        Small ℓ₂ term for numerical stability.
    tol : float, default 1e-6
        Coefficients whose |loading| ≤ tol are treated as zero.

    Returns
    -------
    recovered_idx : ndarray
    precision, recall, f1 : floats
    """
    spca = SparsePCA(n_components=n_components,
                     alpha=alpha,
                     ridge_alpha=ridge_alpha,
                     random_state=0)
    spca.fit(X)

    # Use the first component’s loadings
    v = spca.components_[0]          # shape = (n_features,)
    recovered_idx = np.flatnonzero(np.abs(v) > tol)
    true_idx = np.asarray(true_idx)

    tp = len(set(recovered_idx) & set(true_idx))
    precision = tp / len(recovered_idx) if recovered_idx.size else 0.0
    recall = tp / len(true_idx) if true_idx.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Recovered indices: {sorted(recovered_idx.tolist())}")
    print(f"True indices     : {sorted(true_idx.tolist())}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return recovered_idx, precision, recall, f1

def covariance_thresholding_S(S, k, lambda_thresh):
    """
    Perform covariance thresholding on a sample covariance matrix S.

    Parameters:
    -----------
    S : ndarray of shape (p, p)
        Sample covariance matrix.
    lambda_thresh : float
        Threshold value; entries smaller (in magnitude) than this are set to 0.
    diagonal_only : bool, default False
        If True, only threshold variances (diagonal entries).

    Returns:
    --------
    Sigma_thresh : ndarray of shape (p, p)
        Thresholded sample covariance matrix.

    selected_indices : ndarray of ints
        Indices of features with diagonal ≥ lambda_thresh (i.e., variance survived threshold).
    """
    #anything below the threshold that is not on the diagonal will be set to 0
    # compute pca on sample covariance matrix S after the thresholding
    # take the PC1 and get the k largest entries in absolute value
    #return the indices of these entries
    S_thr = S.copy()

    # create a mask for off-diagonal elements
    off_diag_mask = ~np.eye(S.shape[0], dtype=bool)

    # zero out off-diagonal entries below threshold
    S_thr[off_diag_mask & (np.abs(S_thr) < lambda_thresh)] = 0.0

    # compute PCA (eigen-decomposition since S is symmetric)
    eigvals, eigvecs = np.linalg.eigh(S_thr)

    # take the eigenvector corresponding to largest eigenvalue
    pc1 = eigvecs[:, np.argmax(eigvals)]

    # take indices of k largest absolute entries
    idx_sorted = np.argsort(np.abs(pc1))[::-1]
    topk_indices = idx_sorted[:k]

    return topk_indices


def covariance_thresholding_grid_search(S, k, true_idx):
    c = 0
    max_ratio = 0
    for c in range(1, 100):
        lambda_thresh = c * np.sqrt(np.log(n) / n)
        print("\t", lambda_thresh)
        selected_indices = covariance_thresholding_S(S,k, lambda_thresh=lambda_thresh)
        ratio = len(
                set(true_idx).intersection(set(selected_indices))
            )/len(true_idx)
        max_ratio = max(max_ratio, ratio)
        if max_ratio == 1:
            break
    return c -1, max_ratio


def generate_sample_covariance(n, p, k, beta, seed=None) :
    if seed is not None:
        np. random. seed (seed)

    # Generate spike vector v
    v = np.zeros(p)
    support = np.random.choice(p, k, replace=False)
    v [support] = 1 / np.sqrt(k)

    # Population covariance matrix
    Sigma = np.eye(p) + beta * np.outer(v, v)

    # Sample X directly from multivariate normal
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

    # Sample covariance matrix
    S = (X.T @ X) / n

    return X, S, Sigma, v




def compute_reconstruction_loss(M, v_true, indices, theta=None):
    """
    Compute the reconstruction loss of a symmetric matrix M,
    comparing:
    - the true spike component: theta * v_true v_true^T
    - the estimated spike using only the chosen indices from M

    Also compute the variance captured by the true and estimated spikes.

    Note:
    v_true here refers to the second output of the `generate_symmetric_sparse_spike` function
    (the ground-truth planted sparse spike vector).

    Parameters
    ----------
    M : ndarray of shape (p, p)
        Observed symmetric matrix (e.g., from generate_symmetric_sparse_spike)
    v_true : ndarray of shape (p,)
        Ground-truth sparse spike vector (unit norm), from generate_symmetric_sparse_spike
    indices : list or ndarray of ints
        Chosen indices (support of estimate)
    theta : float or None, optional
        Known signal strength. If None, it will be estimated as v_true.T @ M @ v_true.

    Returns
    -------
    true_loss : float
        Squared Frobenius norm of (M - true spike)
    est_loss : float
        Squared Frobenius norm of (M - estimated spike)
    true_signal : float
        Variance captured by true spike direction
    est_signal : float
        Variance captured by estimated spike direction
    """
    p = M.shape[0]

    # Use provided theta if known, otherwise estimate from M and v_true
    if theta is None:
        theta_hat = v_true.T @ M @ v_true
    else:
        theta_hat = theta

    # true spike component
    S_true = theta_hat * np.outer(v_true, v_true)

    # estimated spike using only M and chosen indices
    submatrix = M[np.ix_(indices, indices)]
    eigvals, eigvecs = np.linalg.eigh(submatrix)
    leading_eigvec = eigvecs[:, -1]

    v_est = np.zeros(p)
    v_est[indices] = leading_eigvec
    v_est /= np.linalg.norm(v_est)

    S_est = theta_hat * np.outer(v_est, v_est)

    # compute losses
    true_loss = np.linalg.norm(M - S_true, ord='fro')**2
    est_loss = np.linalg.norm(M - S_est, ord='fro')**2

    # compute variance captured
    true_signal = v_true.T @ M @ v_true
    est_signal = v_est.T @ M @ v_est

    return true_loss, est_loss, true_signal, est_signal

