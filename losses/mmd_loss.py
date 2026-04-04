from __future__ import annotations
import torch


def _gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:

    dist2 = torch.cdist(x, y, p=2).pow(2)
    beta = 1.0 / (2.0 * sigmas.view(-1, 1, 1).pow(2))
    return torch.exp(-beta * dist2.unsqueeze(0)).sum(dim=0)


def _median_heuristic_sigma(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        if z.shape[0] > 256:
            idx = torch.randperm(z.shape[0], device=z.device)[:256]
            z = z[idx]
        d2 = torch.cdist(z, z, p=2).pow(2)
        med = torch.median(d2[d2 > 0])
        med = med.clamp_min(eps)
    return torch.sqrt(med)


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_multipliers: list[float] = None,
    unbiased: bool = True,
    clamp_nonneg: bool = True,
) -> torch.Tensor:

    if sigma_multipliers is None:
        sigma_multipliers = [0.5, 1.0, 2.0]

    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    base_sigma = _median_heuristic_sigma(x, y)
    sigmas = base_sigma * torch.tensor(sigma_multipliers, device=x.device, dtype=x.dtype)

    Kxx = _gaussian_kernel_matrix(x, x, sigmas)
    Kyy = _gaussian_kernel_matrix(y, y, sigmas)
    Kxy = _gaussian_kernel_matrix(x, y, sigmas)

    if unbiased:
        n = x.shape[0]
        m = y.shape[0]
        eps = 1e-8
        if n > 1:
            Kxx_mean = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1) + eps)
        else:
            Kxx_mean = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if m > 1:
            Kyy_mean = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1) + eps)
        else:
            Kyy_mean = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        Kxy_mean = Kxy.mean()
        mmd2 = Kxx_mean + Kyy_mean - 2.0 * Kxy_mean
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    if clamp_nonneg:
        mmd2 = torch.relu(mmd2)
    return mmd2
