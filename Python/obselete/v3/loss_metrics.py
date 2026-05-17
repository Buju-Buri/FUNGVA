"""
Python conversion of loss_metrics(1).R.

This module preserves the same statistical objective as the R code:
    total VAE loss = reconstruction loss + KL loss + lambda_pen * cross-covariance penalty

R -> Python dimension mapping note:
- R torch uses 1-based dim indexing.
- R dim = 1 corresponds to Python dim = 0.
- R dim = 2 corresponds to Python dim = 1.

The functions below accept predictions either as a dict with keys
"recon", "mu", "logvar", and "cov", or as an object with attributes
.recon, .mu, .logvar, and .cov.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F


def _get(preds: Any, name: str) -> torch.Tensor:
    """Read a tensor from either a dict-style or attribute-style prediction object."""
    if isinstance(preds, dict):
        return preds[name]
    return getattr(preds, name)


def cross_cov_penalty(mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    Python equivalent of R cross_cov_penalty(mu, cov).

    R code:
        B <- mu$size(1)
        mu_c  <- mu  - mu$mean(dim = 1, keepdim = TRUE)
        cov_c <- cov - cov$mean(dim = 1, keepdim = TRUE)
        cov_mat <- torch_matmul(mu_c$t(), cov_c) / max(B - 1, 1)
        torch_mean(cov_mat$pow(2))

    Since R torch dimensions are 1-based, R dim = 1 maps to Python dim = 0.
    """
    B = mu.size(0)

    mu_c = mu - mu.mean(dim=0, keepdim=True)
    cov_c = cov - cov.mean(dim=0, keepdim=True)

    denom = max(B - 1, 1)
    cov_mat = torch.matmul(mu_c.t(), cov_c) / denom

    return torch.mean(cov_mat.pow(2))


def make_vae_loss(lambda_pen: float = 0.0) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """
    Return the VAE loss function with the same components as the R closure.

    Expected preds fields:
        preds["recon"] or preds.recon
        preds["mu"] or preds.mu
        preds["logvar"] or preds.logvar
        preds["cov"] or preds.cov
    """

    def vae_loss(preds: Any, target: torch.Tensor) -> torch.Tensor:
        recon = _get(preds, "recon")
        mu = _get(preds, "mu")
        logvar = _get(preds, "logvar")
        cov = _get(preds, "cov")

        B = target.size(0)

        recon_loss = 0.5 * F.mse_loss(recon, target, reduction="sum") / B

        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - torch.exp(logvar)
        ) / B

        pen = cross_cov_penalty(mu, cov)

        return recon_loss + kl + lambda_pen * pen

    return vae_loss


def recon_metric(preds: Any, target: torch.Tensor) -> float:
    """Batch reconstruction metric matching the R luz recon metric update value."""
    recon = _get(preds, "recon")
    B = target.size(0)
    val = 0.5 * F.mse_loss(recon, target, reduction="sum") / B
    return float(val.detach().cpu().item())


def kl_metric(preds: Any, target: Optional[torch.Tensor] = None) -> float:
    """Batch KL metric matching the R luz KL metric update value."""
    mu = _get(preds, "mu")
    logvar = _get(preds, "logvar")
    B = mu.size(0)
    val = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - torch.exp(logvar)
    ) / B
    return float(val.detach().cpu().item())


def pen_metric(preds: Any, target: Optional[torch.Tensor] = None) -> float:
    """Batch penalty metric matching the R luz penalty metric update value."""
    mu = _get(preds, "mu")
    cov = _get(preds, "cov")
    val = cross_cov_penalty(mu, cov)
    return float(val.detach().cpu().item())


def active_dims_value(
    preds: Any,
    target: Optional[torch.Tensor] = None,
    threshold: float = 0.01,
) -> int:
    """
    Compute active latent dimensions for a batch.

    R code accumulates the KL contribution per latent dimension and counts the
    dimensions whose mean KL exceeds threshold. R dim = 1 maps to Python dim = 0,
    so this sums over the batch axis.
    """
    mu = _get(preds, "mu")
    logvar = _get(preds, "logvar")

    kl_mat = -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))
    kl_dim_mean = kl_mat.mean(dim=0)

    return int((kl_dim_mean > threshold).sum().detach().cpu().item())


def active_dims_metric_gen(threshold: float = 0.01) -> Callable[[Any, Optional[torch.Tensor]], int]:
    """Return an active-dimensions metric callable with a fixed threshold."""

    def metric(preds: Any, target: Optional[torch.Tensor] = None) -> int:
        return active_dims_value(preds, target, threshold=threshold)

    return metric


active_dims_metric = active_dims_metric_gen(threshold=0.01)


class RunningMeanMetric:
    """
    Minimal stateful metric helper, useful when you want epoch-level metrics.

    This mimics the luz pattern:
        initialize -> update repeatedly -> compute
    """

    def __init__(self, metric_fn: Callable[[Any, torch.Tensor], float]) -> None:
        self.metric_fn = metric_fn
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, preds: Any, target: torch.Tensor) -> None:
        B = target.size(0)
        val = self.metric_fn(preds, target)
        self.sum += val * B
        self.n += B

    def compute(self) -> float:
        return self.sum / max(self.n, 1)


class ActiveDimsRunningMetric:
    """Stateful epoch-level active-dimension metric matching the R accumulation."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.kl_sum: Optional[torch.Tensor] = None
        self.n = 0

    @torch.no_grad()
    def update(self, preds: Any, target: torch.Tensor) -> None:
        mu = _get(preds, "mu")
        logvar = _get(preds, "logvar")
        B = target.size(0)

        kl_mat = -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))
        kl_dim_sum = kl_mat.sum(dim=0).detach().cpu()

        if self.kl_sum is None:
            self.kl_sum = kl_dim_sum
        else:
            self.kl_sum = self.kl_sum + kl_dim_sum

        self.n += B

    def compute(self) -> int:
        if self.kl_sum is None or self.n == 0:
            return 0
        kl_mean = self.kl_sum / self.n
        return int((kl_mean > self.threshold).sum().item())


def make_running_metrics(threshold: float = 0.01) -> Dict[str, Any]:
    """Convenience factory for epoch-level metric objects."""
    return {
        "recon": RunningMeanMetric(recon_metric),
        "kl": RunningMeanMetric(kl_metric),
        "penalty": RunningMeanMetric(pen_metric),
        "active_dims": ActiveDimsRunningMetric(threshold=threshold),
    }
