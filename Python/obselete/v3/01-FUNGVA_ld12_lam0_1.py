"""
Python conversion of FUNGVA_ld12_lam0.1(1).R.

This script preserves the same statistical training goal:
- FUNGVA VAE training
- latent_dim = 12
- lambda_pen = 0.1
- seeds = 101, 202, 303
- Adam optimizer
- OneCycleLR with max_lr = 0.025270402
- early stopping with patience = 30
- keep/save best model per seed
- epochs = 800

Assumptions:
- You have Python equivalents of:
  FUNGVA_model_definition.R -> FUNGVA_model_definition.py
  datasets.R                -> datasets.py
  loss_metrics.R            -> loss_metrics.py
- train_dl and val_dl are PyTorch DataLoader objects.
- FUNGVA_Trainer is a torch.nn.Module-compatible class.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from FUNGVA_model_definition import FUNGVA_Trainer
from datasets import train_dl, val_dl, masks, cov_dim
from loss_metrics import (
    make_vae_loss,
    recon_metric,
    kl_metric,
    pen_metric,
    active_dims_metric,
)


def set_seed(seed_id: int) -> None:
    """Equivalent intent to R's set.seed(seed_id) and torch_manual_seed(seed_id)."""
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_id)
        torch.cuda.manual_seed_all(seed_id)


def move_to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to the selected device."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    return x


def model_forward(model: nn.Module, batch: Any) -> Any:
    """
    Generic model call.

    If your model expects a different input structure, adjust only this function.
    The training objective and hyperparameters stay the same.
    """
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def compute_loss(loss_fn: Callable[..., Any], output: Any, batch: Any) -> torch.Tensor:
    """
    Generic loss call.

    This supports common Python VAE loss styles:
      loss_fn(output, batch)
      loss_fn(batch, output)
      loss_fn(output)
    If your converted make_vae_loss has a known signature, simplify this.
    """
    last_error = None

    for args in ((output, batch), (batch, output), (output,)):
        try:
            loss = loss_fn(*args)
            if isinstance(loss, dict):
                loss = loss["loss"]
            elif isinstance(loss, (tuple, list)):
                loss = loss[0]
            return loss
        except TypeError as err:
            last_error = err

    raise TypeError(
        "Could not call loss_fn. Edit compute_loss() to match your make_vae_loss signature."
    ) from last_error


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[Any],
    loss_fn: Callable[..., Any],
    device: torch.device,
) -> float:
    """Validation loss used for early stopping and best-model selection."""
    model.eval()

    total_loss = 0.0
    total_n = 0

    for batch in data_loader:
        batch = move_to_device(batch, device)
        output = model_forward(model, batch)
        loss = compute_loss(loss_fn, output, batch)

        batch_n = infer_batch_size(batch)
        total_loss += float(loss.detach().cpu()) * batch_n
        total_n += batch_n

    return total_loss / max(total_n, 1)


def infer_batch_size(batch: Any) -> int:
    """Infer batch size from common DataLoader batch structures."""
    if torch.is_tensor(batch):
        return batch.shape[0]
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value):
                return value.shape[0]
    if isinstance(batch, (tuple, list)):
        for value in batch:
            if torch.is_tensor(value):
                return value.shape[0]
            if isinstance(value, dict):
                return infer_batch_size(value)
    return 1


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    batch: Any,
    output: Any,
    metric_fns: Dict[str, Callable[..., Any]],
) -> Dict[str, float]:
    """
    Optional metric computation matching the R luz metrics list:
    recon_metric, kl_metric, pen_metric, active_dims_metric.

    Metric signatures vary, so this uses the same flexible calling pattern.
    """
    values: Dict[str, float] = {}

    for name, fn in metric_fns.items():
        metric_value = None
        for args in ((output, batch), (batch, output), (output,), (model, batch, output)):
            try:
                metric_value = fn(*args)
                break
            except TypeError:
                continue

        if metric_value is not None:
            if torch.is_tensor(metric_value):
                metric_value = metric_value.detach().cpu().item()
            values[name] = float(metric_value)

    return values


def fit_one_seed_graph(
    seed_id: int,
    train_dl: Iterable[Any],
    val_dl: Iterable[Any],
    masks: Any,
    cov_dim: int,
    latent_dim: int = 12,
    lambda_pen: float = 0.1,
    max_lr: float = 0.025270402,
    epochs: int = 800,
    patience: int = 30,
    model_dir: Optional[str | os.PathLike[str]] = None,
    device: Optional[str | torch.device] = None,
) -> nn.Module:
    """
    Python equivalent of the R fit_one_seed_graph() function.

    Returns the best fitted model for this seed.
    """
    print("\n---------------------------------")
    print(
        "FUNGVA model run:",
        "latent_dim =", latent_dim,
        "lambda_pen =", lambda_pen,
        "seed =", seed_id,
        "max_lr =", max_lr,
    )
    print("---------------------------------")

    set_seed(seed_id)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loss_fn = make_vae_loss(lambda_pen)

    model = FUNGVA_Trainer(
        latent_dim=latent_dim,
        cov_dim=cov_dim,
        masks=masks,
    ).to(device)

    optimizer = Adam(model.parameters())

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_dl),
    )

    metric_fns = {
        "recon": recon_metric,
        "kl": kl_metric,
        "pen": pen_metric,
        "active_dims": active_dims_metric,
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_n = 0

        for batch in train_dl:
            batch = move_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            output = model_forward(model, batch)
            loss = compute_loss(loss_fn, output, batch)

            loss.backward()
            optimizer.step()

            # Equivalent to luz_callback_lr_scheduler(..., call_on = "on_batch_end")
            scheduler.step()

            batch_n = infer_batch_size(batch)
            running_loss += float(loss.detach().cpu()) * batch_n
            running_n += batch_n

        train_loss = running_loss / max(running_n, 1)
        val_loss = evaluate(model, val_dl, loss_fn, device)

        print(
            f"Epoch {epoch:04d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # Equivalent to luz_callback_keep_best_model()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Equivalent to luz_callback_early_stopping(patience = patience)
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best_val_loss={best_val_loss:.6f}"
            )
            break

    model.load_state_dict(best_state)

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / f"FUNGVA_ld12_lam0.1_seed_{seed_id}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": latent_dim,
                "cov_dim": cov_dim,
                "lambda_pen": lambda_pen,
                "seed_id": seed_id,
                "max_lr": max_lr,
                "best_val_loss": best_val_loss,
            },
            model_file,
        )

    return model


if __name__ == "__main__":
    seed_vec = [101, 202, 303]

    model_dir = "FUNGVA_ld12_lam0.1"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    for sd in seed_vec:
        run_i = fit_one_seed_graph(
            seed_id=sd,
            train_dl=train_dl,
            val_dl=val_dl,
            masks=masks,
            cov_dim=cov_dim,
            latent_dim=12,
            lambda_pen=0.1,
            max_lr=0.025270402,
            epochs=800,
            patience=30,
            model_dir=model_dir,
        )
