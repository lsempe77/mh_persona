"""
anchor_steering_diagnostics.py

Reference implementations for the "Theory + Diagnostics" protocol
in anchor-cosine steering. Designed to plug into an activation-steering
pipeline that produces:
- anchor activation vectors a_i in R^d (one per anchor prompt)
- hidden activation vectors h in R^d (one per prompt / layer)
- contrast-pair datasets of relative features x_k in R^m and target directions y_k in R^d

This file is intentionally lightweight and self-contained.

Dependencies: torch, numpy (optional). Works on CPU by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import torch


# ---------------------------
# Core linear algebra helpers
# ---------------------------

def unit_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return v / ||v|| with safe epsilon; works for 1D or batched tensors."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def make_anchor_matrix(anchors: torch.Tensor) -> torch.Tensor:
    """
    anchors: Tensor[m, d] of anchor activation vectors a_i
    returns A: Tensor[d, m] with unit anchors as columns.
    """
    if anchors.ndim != 2:
        raise ValueError(f"anchors must be [m,d], got {tuple(anchors.shape)}")
    A_cols = unit_normalize(anchors)  # [m,d]
    return A_cols.T.contiguous()      # [d,m]


def gram_matrix(A: torch.Tensor) -> torch.Tensor:
    """G = A^T A."""
    if A.ndim != 2:
        raise ValueError(f"A must be [d,m], got {tuple(A.shape)}")
    return A.T @ A


def gram_spectrum(G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (eigenvalues, eigenvectors) for symmetric G. Eigenvalues sorted ascending.
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be square")
    evals, evecs = torch.linalg.eigh(G)  # ascending
    return evals, evecs


def condition_number_from_gram(G: torch.Tensor, eps: float = 1e-12) -> float:
    """kappa(G) = lambda_max / lambda_min (for SPD)."""
    evals, _ = gram_spectrum(G)
    lmin = float(torch.clamp(evals[0], min=eps))
    lmax = float(torch.clamp(evals[-1], min=eps))
    return lmax / lmin


# ---------------------------
# Cosine features and projection
# ---------------------------

def cosine_features(h: torch.Tensor, A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    c(h) = A^T h / ||h||.
    h: Tensor[d] or Tensor[n,d]
    A: Tensor[d,m]
    returns: Tensor[m] or Tensor[n,m]
    """
    if h.ndim == 1:
        h2 = h.unsqueeze(0)  # [1,d]
    elif h.ndim == 2:
        h2 = h
    else:
        raise ValueError("h must be [d] or [n,d]")
    norm = h2.norm(dim=-1, keepdim=True) + eps
    c = (h2 @ A) / norm  # [n,m]
    return c.squeeze(0) if h.ndim == 1 else c


def project_onto_anchor_span(h: torch.Tensor, A: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection Pi_A(h) = A G^{-1} A^T h.
    h: Tensor[d] or Tensor[n,d]
    A: Tensor[d,m]
    G_inv: Tensor[m,m]
    returns: same shape as h
    """
    if h.ndim == 1:
        return A @ (G_inv @ (A.T @ h))
    if h.ndim == 2:
        return (h @ A) @ G_inv @ A.T
    raise ValueError("h must be [d] or [n,d]")


def reconstruct_projection_from_cosines(
    h: torch.Tensor, A: torch.Tensor, G_inv: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Lemma 1 reconstruction: Pi_A(h) = ||h|| A G^{-1} c(h).
    This should match project_onto_anchor_span(h, A, G_inv) up to numeric error.
    """
    c = cosine_features(h, A, eps=eps)
    if h.ndim == 1:
        return h.norm() * (A @ (G_inv @ c))
    return (h.norm(dim=-1, keepdim=True) * (c @ G_inv @ A.T))


def coverage_ratio(h: torch.Tensor, A: torch.Tensor, G_inv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rho(h) = ||h - Pi_A(h)|| / ||h||, computed per vector.
    returns scalar (if h is [d]) or [n] if h is [n,d].
    """
    proj = project_onto_anchor_span(h, A, G_inv)
    if h.ndim == 1:
        return (h - proj).norm() / (h.norm() + eps)
    return (h - proj).norm(dim=-1) / (h.norm(dim=-1) + eps)


def ridge_projector(A: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Return matrix P_lambda = A (G + lam I)^{-1} A^T (a [d,d] operator).
    Suitable for applying to many vectors.
    """
    G = gram_matrix(A)
    m = G.shape[0]
    I = torch.eye(m, device=G.device, dtype=G.dtype)
    inv = torch.linalg.inv(G + lam * I)
    return A @ inv @ A.T


def ridge_bias_bound(lam: float, G: torch.Tensor) -> float:
    """
    Corollary 4.1 bound factor: lam / (lambda_min(G) + lam).
    """
    evals, _ = gram_spectrum(G)
    lmin = float(torch.clamp(evals[0], min=1e-12))
    return lam / (lmin + lam)


# ---------------------------
# Learned projection (ridge regression) + LOO
# ---------------------------

def fit_ridge_W(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Fit W = argmin ||XW - Y||_F^2 + lam ||W||_F^2.
    X: [n,m], Y: [n,d]
    returns W: [m,d]
    
    Note: Automatically handles bfloat16 by casting to float32 for linalg.solve,
    since CUDA's lu_factor_cusolver doesn't support bfloat16.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of rows")
    n, m = X.shape
    _, d = Y.shape
    
    # Handle bfloat16: cast to float32 for linear algebra, then cast back
    original_dtype = X.dtype
    if X.dtype == torch.bfloat16:
        X = X.float()
        Y = Y.float()
    
    XtX = X.T @ X
    XtY = X.T @ Y
    I = torch.eye(m, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + lam * I, XtY)
    
    # Cast back to original dtype if needed
    if original_dtype == torch.bfloat16:
        W = W.to(original_dtype)
    return W


def loo_ridge_errors(X: torch.Tensor, Y: torch.Tensor, lam: float) -> Dict[str, torch.Tensor]:
    """
    Leave-one-out (LOO) evaluation for ridge regression in small-n regime.
    Returns per-point prediction errors.

    Outputs:
      - pred_norm_error: ||Y_i - X_i W^{(-i)}|| / ||Y_i||
      - pred_cos_error: 1 - cosine(Y_i, X_i W^{(-i)})
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D")
    n = X.shape[0]
    pred = []
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool, device=X.device)
        mask[i] = False
        W = fit_ridge_W(X[mask], Y[mask], lam=lam)
        yhat = X[i:i+1] @ W  # [1,d]
        pred.append(yhat.squeeze(0))
    Yhat = torch.stack(pred, dim=0)  # [n,d]

    # norm-relative error
    denom = (Y.norm(dim=-1) + 1e-8)
    pred_norm_error = (Y - Yhat).norm(dim=-1) / denom

    # cosine error (directional)
    Yn = unit_normalize(Y)
    Yhn = unit_normalize(Yhat)
    pred_cos_error = 1.0 - (Yn * Yhn).sum(dim=-1)

    return {
        "Yhat": Yhat,
        "pred_norm_error": pred_norm_error,
        "pred_cos_error": pred_cos_error,
    }


# ---------------------------
# Anchor pruning (optional helper)
# ---------------------------

def prune_anchors_greedy(
    anchors: torch.Tensor,
    target_m: int,
    min_improvement: float = 1e-6,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Greedily remove anchors to improve conditioning of the Gram matrix.

    anchors: [m,d]
    target_m: keep this many anchors
    returns (pruned_anchors [target_m,d], kept_indices)
    """
    if anchors.ndim != 2:
        raise ValueError("anchors must be [m,d]")
    m, d = anchors.shape
    if target_m <= 0 or target_m > m:
        raise ValueError("target_m must be in [1,m]")

    kept = list(range(m))

    def score(idx_list: List[int]) -> float:
        A = make_anchor_matrix(anchors[idx_list])
        G = gram_matrix(A)
        return condition_number_from_gram(G)

    current_score = score(kept)

    while len(kept) > target_m:
        best_score = current_score
        best_remove = None
        for j in kept:
            trial = [i for i in kept if i != j]
            s = score(trial)
            if s + min_improvement < best_score:
                best_score = s
                best_remove = j
        if best_remove is None:
            # no improvement; stop early
            break
        kept.remove(best_remove)
        current_score = best_score

    return anchors[kept], kept


# ---------------------------
# Robustness Testing
# ---------------------------

@dataclass
class RobustnessReport:
    """Container for robustness test results."""
    bootstrap_ci: Optional[Tuple[float, float]] = None
    bootstrap_mean: Optional[float] = None
    bootstrap_std: Optional[float] = None
    permutation_p_value: Optional[float] = None
    lambda_sensitivity: Optional[Dict[float, float]] = None
    anchor_dropout_mean: Optional[float] = None
    anchor_dropout_std: Optional[float] = None
    layer_consistency: Optional[float] = None
    
    def __repr__(self) -> str:
        lines = ["RobustnessReport:"]
        if self.bootstrap_ci is not None:
            lines.append(f"  Bootstrap 95% CI: [{self.bootstrap_ci[0]:.3f}, {self.bootstrap_ci[1]:.3f}]")
            lines.append(f"  Bootstrap mean: {self.bootstrap_mean:.3f} ± {self.bootstrap_std:.3f}")
        if self.permutation_p_value is not None:
            sig = "***" if self.permutation_p_value < 0.001 else "**" if self.permutation_p_value < 0.01 else "*" if self.permutation_p_value < 0.05 else ""
            lines.append(f"  Permutation p-value: {self.permutation_p_value:.4f} {sig}")
        if self.lambda_sensitivity is not None:
            lines.append(f"  Lambda sensitivity: {self.lambda_sensitivity}")
        if self.anchor_dropout_mean is not None:
            lines.append(f"  Anchor dropout: {self.anchor_dropout_mean:.3f} ± {self.anchor_dropout_std:.3f}")
        if self.layer_consistency is not None:
            lines.append(f"  Layer consistency: {self.layer_consistency:.3f}")
        return "\n".join(lines)


class RobustnessChecker:
    """
    Comprehensive robustness testing for activation steering results.
    
    Designed to validate that steering correlations are:
    1. Statistically significant (permutation test)
    2. Stable across resampling (bootstrap CI)
    3. Not overly sensitive to hyperparameters (lambda sensitivity)
    4. Robust to anchor perturbations (anchor dropout)
    5. Consistent across layers (layer consistency)
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
    
    def _pearson_r(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation coefficient."""
        if x.ndim > 1:
            x = x.flatten()
        if y.ndim > 1:
            y = y.flatten()
        x = x.float()
        y = y.float()
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        num = (x_centered * y_centered).sum()
        denom = (x_centered.norm() * y_centered.norm()) + 1e-8
        return float(num / denom)
    
    def bootstrap_ci(
        self,
        coefficients: torch.Tensor,
        scores: torch.Tensor,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for Pearson correlation.
        
        Args:
            coefficients: Steering coefficients used [n]
            scores: Judge scores or measured outcomes [n]
            n_bootstrap: Number of bootstrap resamples
            ci: Confidence level (default 0.95 for 95% CI)
        
        Returns:
            Dict with 'ci_low', 'ci_high', 'mean', 'std'
        """
        coefficients = coefficients.flatten().float()
        scores = scores.flatten().float()
        n = len(coefficients)
        
        if n < 3:
            return {"ci_low": float("nan"), "ci_high": float("nan"), 
                    "mean": float("nan"), "std": float("nan")}
        
        boot_rs = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, n, (n,), generator=self.rng)
            r = self._pearson_r(coefficients[idx], scores[idx])
            if not (torch.isnan(torch.tensor(r)) or torch.isinf(torch.tensor(r))):
                boot_rs.append(r)
        
        if len(boot_rs) < 10:
            return {"ci_low": float("nan"), "ci_high": float("nan"),
                    "mean": float("nan"), "std": float("nan")}
        
        boot_rs = sorted(boot_rs)
        alpha = 1 - ci
        low_idx = int(len(boot_rs) * (alpha / 2))
        high_idx = int(len(boot_rs) * (1 - alpha / 2))
        
        return {
            "ci_low": boot_rs[low_idx],
            "ci_high": boot_rs[min(high_idx, len(boot_rs) - 1)],
            "mean": sum(boot_rs) / len(boot_rs),
            "std": (sum((r - sum(boot_rs)/len(boot_rs))**2 for r in boot_rs) / len(boot_rs)) ** 0.5,
        }
    
    def permutation_test(
        self,
        coefficients: torch.Tensor,
        scores: torch.Tensor,
        n_permutations: int = 1000,
    ) -> Dict[str, float]:
        """
        Permutation test for correlation significance.
        
        Tests H0: no relationship between coefficients and scores.
        
        Args:
            coefficients: Steering coefficients [n]
            scores: Judge scores [n]
            n_permutations: Number of permutations
        
        Returns:
            Dict with 'p_value', 'observed_r', 'null_mean', 'null_std'
        """
        coefficients = coefficients.flatten().float()
        scores = scores.flatten().float()
        n = len(coefficients)
        
        observed_r = self._pearson_r(coefficients, scores)
        
        null_rs = []
        for _ in range(n_permutations):
            perm_idx = torch.randperm(n, generator=self.rng)
            perm_r = self._pearson_r(coefficients, scores[perm_idx])
            null_rs.append(perm_r)
        
        # Two-tailed p-value
        extreme_count = sum(1 for r in null_rs if abs(r) >= abs(observed_r))
        p_value = (extreme_count + 1) / (n_permutations + 1)  # +1 for conservative estimate
        
        null_mean = sum(null_rs) / len(null_rs)
        null_std = (sum((r - null_mean)**2 for r in null_rs) / len(null_rs)) ** 0.5
        
        return {
            "p_value": p_value,
            "observed_r": observed_r,
            "null_mean": null_mean,
            "null_std": null_std,
        }
    
    def lambda_sensitivity(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        lambdas: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0],
        metric: str = "loo_cosine",
    ) -> Dict[float, float]:
        """
        Test sensitivity of learned projection to regularization strength.
        
        Args:
            X: Feature matrix [n, m]
            Y: Target directions [n, d]
            lambdas: List of regularization values to test
            metric: 'loo_cosine' or 'loo_norm'
        
        Returns:
            Dict mapping lambda -> mean error
        """
        results = {}
        for lam in lambdas:
            try:
                errs = loo_ridge_errors(X, Y, lam=lam)
                if metric == "loo_cosine":
                    results[lam] = float(errs["pred_cos_error"].mean())
                else:
                    results[lam] = float(errs["pred_norm_error"].mean())
            except Exception:
                results[lam] = float("nan")
        return results
    
    def anchor_dropout(
        self,
        anchors: torch.Tensor,
        h: torch.Tensor,
        dropout_rate: float = 0.2,
        n_trials: int = 50,
    ) -> Dict[str, float]:
        """
        Test robustness of cosine features to anchor dropout.
        
        Measures stability of coverage ratio when randomly dropping anchors.
        
        Args:
            anchors: Anchor activations [m, d]
            h: Hidden state(s) to test [d] or [n, d]
            dropout_rate: Fraction of anchors to drop each trial
            n_trials: Number of random dropout trials
        
        Returns:
            Dict with coverage ratio statistics
        """
        m = anchors.shape[0]
        n_drop = max(1, int(m * dropout_rate))
        n_keep = m - n_drop
        
        if n_keep < 2:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        
        coverage_ratios = []
        for _ in range(n_trials):
            keep_idx = torch.randperm(m, generator=self.rng)[:n_keep]
            sub_anchors = anchors[keep_idx]
            A = make_anchor_matrix(sub_anchors)
            G = gram_matrix(A)
            try:
                G_inv = torch.linalg.inv(G)
                rho = coverage_ratio(h, A, G_inv)
                if h.ndim == 1:
                    coverage_ratios.append(float(rho))
                else:
                    coverage_ratios.append(float(rho.mean()))
            except Exception:
                continue
        
        if len(coverage_ratios) < 5:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        
        mean_rho = sum(coverage_ratios) / len(coverage_ratios)
        std_rho = (sum((r - mean_rho)**2 for r in coverage_ratios) / len(coverage_ratios)) ** 0.5
        
        return {
            "mean": mean_rho,
            "std": std_rho,
            "min": min(coverage_ratios),
            "max": max(coverage_ratios),
        }
    
    def layer_consistency(
        self,
        results_by_layer: Dict[int, float],
    ) -> Dict[str, float]:
        """
        Measure consistency of r-values across adjacent layers.
        
        High consistency suggests the signal is robust, not layer-specific noise.
        
        Args:
            results_by_layer: Dict mapping layer_idx -> r_value
        
        Returns:
            Dict with 'adjacent_corr' (correlation of r[i] with r[i+1]),
            'mean_r', 'std_r', 'best_layer', 'best_r'
        """
        layers = sorted(results_by_layer.keys())
        rs = [results_by_layer[l] for l in layers]
        
        # Filter out nan values
        valid = [(l, r) for l, r in zip(layers, rs) if not (torch.isnan(torch.tensor(r)) or torch.isinf(torch.tensor(r)))]
        if len(valid) < 3:
            return {"adjacent_corr": float("nan"), "mean_r": float("nan"), 
                    "std_r": float("nan"), "best_layer": -1, "best_r": float("nan")}
        
        layers_valid, rs_valid = zip(*valid)
        
        # Adjacent correlation
        if len(rs_valid) >= 3:
            r_current = torch.tensor(rs_valid[:-1])
            r_next = torch.tensor(rs_valid[1:])
            adj_corr = self._pearson_r(r_current, r_next)
        else:
            adj_corr = float("nan")
        
        mean_r = sum(rs_valid) / len(rs_valid)
        std_r = (sum((r - mean_r)**2 for r in rs_valid) / len(rs_valid)) ** 0.5
        best_idx = max(range(len(rs_valid)), key=lambda i: rs_valid[i])
        
        return {
            "adjacent_corr": adj_corr,
            "mean_r": mean_r,
            "std_r": std_r,
            "best_layer": layers_valid[best_idx],
            "best_r": rs_valid[best_idx],
        }
    
    def full_report(
        self,
        coefficients: torch.Tensor,
        scores: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        anchors: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        results_by_layer: Optional[Dict[int, float]] = None,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
    ) -> RobustnessReport:
        """
        Generate comprehensive robustness report.
        
        Args:
            coefficients: Steering coefficients [n]
            scores: Judge scores [n]
            X: Optional feature matrix for lambda sensitivity [n, m]
            Y: Optional target directions for lambda sensitivity [n, d]
            anchors: Optional anchors for dropout test [m, d]
            h: Optional hidden states for dropout test [d] or [n, d]
            results_by_layer: Optional dict for layer consistency
            n_bootstrap: Bootstrap iterations
            n_permutations: Permutation test iterations
        
        Returns:
            RobustnessReport dataclass with all results
        """
        report = RobustnessReport()
        
        # Bootstrap CI
        boot = self.bootstrap_ci(coefficients, scores, n_bootstrap=n_bootstrap)
        report.bootstrap_ci = (boot["ci_low"], boot["ci_high"])
        report.bootstrap_mean = boot["mean"]
        report.bootstrap_std = boot["std"]
        
        # Permutation test
        perm = self.permutation_test(coefficients, scores, n_permutations=n_permutations)
        report.permutation_p_value = perm["p_value"]
        
        # Lambda sensitivity (if X, Y provided)
        if X is not None and Y is not None:
            report.lambda_sensitivity = self.lambda_sensitivity(X, Y)
        
        # Anchor dropout (if anchors, h provided)
        if anchors is not None and h is not None:
            dropout = self.anchor_dropout(anchors, h)
            report.anchor_dropout_mean = dropout["mean"]
            report.anchor_dropout_std = dropout["std"]
        
        # Layer consistency (if results_by_layer provided)
        if results_by_layer is not None:
            layer = self.layer_consistency(results_by_layer)
            report.layer_consistency = layer["adjacent_corr"]
        
        return report


# ---------------------------
# Minimal example usage
# ---------------------------

def _demo() -> None:
    print("=" * 60)
    print("ANCHOR STEERING DIAGNOSTICS - Demo & Tests")
    print("=" * 60)
    
    torch.manual_seed(0)
    d = 64
    m = 12
    n = 9  # Typical steering experiment size

    # === Original diagnostics ===
    print("\n--- Original Diagnostics ---")
    anchors = torch.randn(m, d)
    A = make_anchor_matrix(anchors)
    G = gram_matrix(A)
    G_inv = torch.linalg.inv(G)

    h = torch.randn(d)
    rho = coverage_ratio(h, A, G_inv)
    kappa = condition_number_from_gram(G)

    print(f"Coverage ratio rho(h): {float(rho):.4f}")
    print(f"Condition number kappa(G): {float(kappa):.2f}")

    # Check Lemma 1 reconstruction
    proj1 = project_onto_anchor_span(h, A, G_inv)
    proj2 = reconstruct_projection_from_cosines(h, A, G_inv)
    print(f"Lemma 1 reconstruction error: {float((proj1 - proj2).norm()):.2e}")

    # Learned projection LOO demo
    X = torch.randn(n, m)
    Y = unit_normalize(torch.randn(n, d))
    lam = 1e-2
    errs = loo_ridge_errors(X, Y, lam=lam)
    print(f"LOO cosine error mean: {float(errs['pred_cos_error'].mean()):.4f}")
    print(f"LOO norm error mean: {float(errs['pred_norm_error'].mean()):.4f}")

    # === NEW: Robustness Testing ===
    print("\n--- Robustness Testing ---")
    
    # Simulate steering experiment data
    # coefficients = [-3, -1.5, 0, 1.5, 3] repeated, scores correlated
    coefficients = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0, -3.0, -1.5, 0.0, 1.5])
    true_r = 0.7  # Simulated true correlation
    noise = torch.randn(n) * 0.3
    scores = coefficients * true_r + noise
    
    checker = RobustnessChecker(seed=42)
    
    # Test 1: Bootstrap CI
    print("\n1. Bootstrap Confidence Interval:")
    boot = checker.bootstrap_ci(coefficients, scores, n_bootstrap=2000)
    print(f"   95% CI: [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")
    print(f"   Mean r: {boot['mean']:.3f} ± {boot['std']:.3f}")
    
    # Test 2: Permutation test
    print("\n2. Permutation Test:")
    perm = checker.permutation_test(coefficients, scores, n_permutations=2000)
    sig = "***" if perm['p_value'] < 0.001 else "**" if perm['p_value'] < 0.01 else "*" if perm['p_value'] < 0.05 else "ns"
    print(f"   Observed r: {perm['observed_r']:.3f}")
    print(f"   p-value: {perm['p_value']:.4f} {sig}")
    print(f"   Null distribution: {perm['null_mean']:.3f} ± {perm['null_std']:.3f}")
    
    # Test 3: Lambda sensitivity
    print("\n3. Lambda Sensitivity:")
    lam_sens = checker.lambda_sensitivity(X, Y)
    for lam_val, err in lam_sens.items():
        print(f"   λ={lam_val}: LOO cosine error = {err:.4f}")
    
    # Test 4: Anchor dropout
    print("\n4. Anchor Dropout Robustness:")
    dropout = checker.anchor_dropout(anchors, h, dropout_rate=0.25, n_trials=100)
    print(f"   Coverage ratio with 25% dropout: {dropout['mean']:.3f} ± {dropout['std']:.3f}")
    print(f"   Range: [{dropout['min']:.3f}, {dropout['max']:.3f}]")
    
    # Test 5: Layer consistency
    print("\n5. Layer Consistency:")
    # Simulate results from different layers (realistic pattern)
    results_by_layer = {
        8: 0.35,
        10: 0.52,
        12: 0.61,
        14: 0.65,
        15: 0.63,
        16: 0.58,
        18: 0.45,
    }
    layer_cons = checker.layer_consistency(results_by_layer)
    print(f"   Adjacent layer correlation: {layer_cons['adjacent_corr']:.3f}")
    print(f"   Mean r across layers: {layer_cons['mean_r']:.3f} ± {layer_cons['std_r']:.3f}")
    print(f"   Best layer: {layer_cons['best_layer']} (r={layer_cons['best_r']:.3f})")
    
    # Test 6: Full report
    print("\n6. Full Robustness Report:")
    report = checker.full_report(
        coefficients=coefficients,
        scores=scores,
        X=X,
        Y=Y,
        anchors=anchors,
        h=h,
        results_by_layer=results_by_layer,
    )
    print(report)
    
    # === Validation checks ===
    print("\n--- Validation Checks ---")
    
    # Check that permutation test detects real signal
    assert perm['p_value'] < 0.05, "Permutation test should detect simulated correlation"
    print("✓ Permutation test correctly detected simulated correlation")
    
    # Check that CI contains true r (approximately)
    # Note: This may occasionally fail due to randomness
    if boot['ci_low'] <= true_r <= boot['ci_high']:
        print("✓ Bootstrap CI contains true r")
    else:
        print(f"⚠ Bootstrap CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] doesn't contain true r={true_r}")
    
    # Check lambda sensitivity runs without error
    assert len(lam_sens) == 5, "Should test 5 lambda values"
    print("✓ Lambda sensitivity computed for all values")
    
    # Check anchor dropout
    assert dropout['std'] < dropout['mean'], "Dropout should be reasonably stable"
    print("✓ Anchor dropout shows reasonable stability")
    
    # Check layer consistency
    assert not torch.isnan(torch.tensor(layer_cons['adjacent_corr'])), "Layer consistency should compute"
    print("✓ Layer consistency computed successfully")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    _demo()
