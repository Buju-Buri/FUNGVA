#!/usr/bin/env python3
"""
Standalone Python version of the MLP_FiLM1 ld10 lambda=0.1 workflow.

This combines the R files:
  - libraries.R
  - datasets.R
  - loss_metrics.R
  - MLP_FiLM1_model_definition.R
  - MLP_FiLM1_ld10_lam0.1.R

Expected default data layout under --workdir:
  python_data/fc_data.h5
    datasets: X_train, X_val, X_test, train_ids, val_ids, test_ids
  python_data/covariates.parquet

The script accepts FC arrays in any of these shapes:
  (n, 2278), (2278, n), (n, 68, 68), or (68, 68, n)

It preserves the statistical goal of the original R/luz workflow:
  loss = reconstruction_loss + KL_loss + lambda_pen * cross_cov_penalty
with latent_dim=10, lambda_pen=0.1, max_lr=0.00538324276491867,
epochs=800, patience=30, and seeds 101, 202, 303.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

P = (68 * 67) // 2
NUM_VARS = ["AGE", "EDUCATION", "pTau_ratio_log", "abeta_ratio_log"]
OTHER_VARS = ["SEX", "PARTNERED", "APOE4"]


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed_id: int) -> None:
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_id)
        torch.cuda.manual_seed_all(seed_id)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def decode_h5_ids(raw: np.ndarray) -> np.ndarray:
    """Decode HDF5 IDs robustly into a string numpy array."""
    arr = np.asarray(raw)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in arr.reshape(-1)
        ], dtype=str)
    return arr.reshape(-1).astype(str)


def lt_rowwise_batch(mats: np.ndarray) -> np.ndarray:
    """Extract strict lower triangle row-wise from a batch of 68 x 68 matrices."""
    idx = np.tril_indices(68, k=-1)
    # np.tril_indices orders by row, matching the R lt_rowwise loop.
    return mats[:, idx[0], idx[1]].astype(np.float32)


def normalize_fc_array(x: np.ndarray, name: str) -> np.ndarray:
    """
    Convert FC input to shape (n, 2278) on correlation scale.

    Accepts:
      - (n, 2278): already vectorized
      - (2278, n): transposed vectorized
      - (n, 68, 68): matrix batch first
      - (68, 68, n): matrix batch last, as in the user's HDF5 file
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        if x.shape[1] == P:
            return x.astype(np.float32)
        if x.shape[0] == P:
            return x.T.astype(np.float32)
        raise ValueError(f"{name} has shape {x.shape}; expected (n, {P}) or ({P}, n).")

    if x.ndim == 3:
        if x.shape[1:] == (68, 68):
            mats = x
        elif x.shape[:2] == (68, 68):
            mats = np.moveaxis(x, 2, 0)
        else:
            raise ValueError(
                f"{name} has shape {x.shape}; expected (n, 68, 68) or (68, 68, n)."
            )
        return lt_rowwise_batch(mats)

    raise ValueError(f"{name} has {x.ndim} dimensions; expected 2D or 3D array.")


def fisher_z(fc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip correlations and apply Fisher atanh transformation."""
    return np.arctanh(np.clip(fc, -1.0 + eps, 1.0 - eps)).astype(np.float32)


def align_covariates(cov_df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Filter and order covariates to match train/val/test IDs exactly."""
    if "RID" not in cov_df.columns:
        raise KeyError("covariates.parquet must contain a column named 'RID'.")

    ids = np.asarray(ids).astype(str)
    cov_df = cov_df.copy()
    cov_df["RID"] = cov_df["RID"].astype(str)

    missing = [rid for rid in ids if rid not in set(cov_df["RID"])]
    if missing:
        raise KeyError(
            f"{len(missing)} subject IDs were not found in covariates.RID. "
            f"First missing IDs: {missing[:10]}"
        )

    return cov_df.set_index("RID").loc[ids].reset_index()


def scale_with_train(df: pd.DataFrame, vars_: Sequence[str], means: pd.Series, sds: pd.Series) -> np.ndarray:
    x = df[list(vars_)].to_numpy(dtype=np.float32)
    denom = sds.to_numpy(dtype=np.float32)
    if np.any(denom == 0):
        zero_vars = list(sds.index[denom == 0])
        raise ValueError(f"At least one training covariate SD is zero: {zero_vars}")
    return (x - means.to_numpy(dtype=np.float32)) / denom


class FungvaDataset(Dataset):
    def __init__(self, fc_mat: np.ndarray, cov_mat: np.ndarray):
        if fc_mat.shape[0] != cov_mat.shape[0]:
            raise ValueError(
                f"fc_mat and cov_mat must have same number of rows; got "
                f"{fc_mat.shape[0]} and {cov_mat.shape[0]}."
            )
        self.fc_mat = torch.tensor(fc_mat, dtype=torch.float32)
        self.cov_mat = torch.tensor(cov_mat, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.fc_mat.shape[0])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        fc = self.fc_mat[i]
        cov = self.cov_mat[i]
        return {"x": {"fc": fc, "cov": cov}, "y": fc, "indices": i}


def collate_fungva_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fc = torch.stack([item["x"]["fc"] for item in batch], dim=0)
    cov = torch.stack([item["x"]["cov"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    indices = torch.tensor([item["indices"] for item in batch], dtype=torch.long)
    return {"x": {"fc": fc, "cov": cov}, "y": y, "indices": indices}


def build_dataloaders(
    workdir: os.PathLike[str] | str,
    train_batch_size: int = 16,
    eval_batch_size: int = 15,
    h5_relpath: str = "python_data/fc_data.h5",
    covariates_relpath: str = "python_data/covariates.parquet",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    workdir = Path(workdir).expanduser().resolve()
    h5_path = workdir / h5_relpath
    cov_path = workdir / covariates_relpath

    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 data file: {h5_path}")
    if not cov_path.exists():
        raise FileNotFoundError(f"Could not find covariates parquet file: {cov_path}")

    with h5py.File(h5_path, "r") as f:
        X_train_raw = f["X_train"][:]
        X_val_raw = f["X_val"][:]
        X_test_raw = f["X_test"][:]
        train_ids = decode_h5_ids(f["train_ids"][:])
        val_ids = decode_h5_ids(f["val_ids"][:])
        test_ids = decode_h5_ids(f["test_ids"][:])

    cov_df = pd.read_parquet(cov_path)

    fc_train = fisher_z(normalize_fc_array(X_train_raw, "X_train"))
    fc_val = fisher_z(normalize_fc_array(X_val_raw, "X_val"))
    fc_test = fisher_z(normalize_fc_array(X_test_raw, "X_test"))

    required = ["RID", *NUM_VARS, *OTHER_VARS]
    missing_cols = [c for c in required if c not in cov_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required covariate columns: {missing_cols}")

    cov_train = align_covariates(cov_df, train_ids)
    cov_val = align_covariates(cov_df, val_ids)
    cov_test = align_covariates(cov_df, test_ids)

    train_means = cov_train[NUM_VARS].mean(skipna=True)
    train_sds = cov_train[NUM_VARS].std(skipna=True, ddof=1)

    cov_num_train = scale_with_train(cov_train, NUM_VARS, train_means, train_sds)
    cov_num_val = scale_with_train(cov_val, NUM_VARS, train_means, train_sds)
    cov_num_test = scale_with_train(cov_test, NUM_VARS, train_means, train_sds)

    cov_other_train = cov_train[OTHER_VARS].to_numpy(dtype=np.float32)
    cov_other_val = cov_val[OTHER_VARS].to_numpy(dtype=np.float32)
    cov_other_test = cov_test[OTHER_VARS].to_numpy(dtype=np.float32)

    cov_train_mat = np.column_stack([cov_num_train, cov_other_train]).astype(np.float32)
    cov_val_mat = np.column_stack([cov_num_val, cov_other_val]).astype(np.float32)
    cov_test_mat = np.column_stack([cov_num_test, cov_other_test]).astype(np.float32)
    cov_dim = int(cov_train_mat.shape[1])

    train_ds = FungvaDataset(fc_train, cov_train_mat)
    val_ds = FungvaDataset(fc_val, cov_val_mat)
    test_ds = FungvaDataset(fc_test, cov_test_mat)

    set_seed(1)
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fungva_batch)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fungva_batch)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fungva_batch)

    print(f"Loaded X_train raw shape: {tuple(X_train_raw.shape)} -> FC train shape: {tuple(fc_train.shape)}")
    print(f"Loaded X_val raw shape:   {tuple(X_val_raw.shape)} -> FC val shape:   {tuple(fc_val.shape)}")
    print(f"Loaded X_test raw shape:  {tuple(X_test_raw.shape)} -> FC test shape:  {tuple(fc_test.shape)}")
    print(f"cov_dim = {cov_dim}")

    return train_dl, val_dl, test_dl, cov_dim


# -----------------------------------------------------------------------------
# MLP_FiLM1 model definition
# -----------------------------------------------------------------------------

class FUNGVA_MLP_FiLM1(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cov_dim = int(cov_dim)

        self.en_mu_la1 = nn.Linear(P, 1024, bias=False)
        self.en_var_la1 = nn.Linear(P, 1024, bias=False)
        self.en_mu_la2 = nn.Linear(1024, 128, bias=False)
        self.en_var_la2 = nn.Linear(1024, 128, bias=False)
        self.en_mu_la3 = nn.Linear(128, latent_dim)
        self.en_var_la3 = nn.Linear(128, latent_dim)

        dec_hidden1 = 128
        dec_hidden2 = 1024
        self.dc_la1 = nn.Linear(latent_dim, dec_hidden1)
        self.dc_la2 = nn.Linear(dec_hidden1, dec_hidden2)
        self.dc_out = nn.Linear(dec_hidden2, P)

        film_hidden = 32
        self.film1 = nn.Sequential(
            nn.Linear(cov_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * dec_hidden1),
        )
        self.tanh_act = nn.Tanh()

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = F.relu(self.en_mu_la1(x))
        mu = F.relu(self.en_mu_la2(mu))
        mu = self.en_mu_la3(mu)

        logvar = F.relu(self.en_var_la1(x))
        logvar = F.relu(self.en_var_la2(logvar))
        logvar = self.en_var_la3(logvar)
        return {"mu": mu, "logvar": logvar}

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def film_modulate(h: torch.Tensor, cov: torch.Tensor, film_net: nn.Module) -> torch.Tensor:
        film_params = film_net(cov)
        out_dim = h.size(1)
        gamma = film_params[:, :out_dim]
        beta = film_params[:, out_dim:2 * out_dim]
        return (1.0 + gamma) * h + beta

    def decode(self, z: torch.Tensor, cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        h1 = F.relu(self.dc_la1(z))
        h1 = self.film_modulate(h1, cov, self.film1)
        h2 = F.relu(self.dc_la2(h1))
        dc_output = self.dc_out(h2)
        return {"recon": dc_output, "fc_pred": self.tanh_act(dc_output)}

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fc = x["fc"]
        cov = x["cov"]
        enc = self.encode(fc)
        mu = enc["mu"]
        logvar = enc["logvar"]
        z = self.reparameterize(mu, logvar)
        dec = self.decode(z, cov)
        return {
            "recon": dec["recon"],
            "mu": mu,
            "logvar": logvar,
            "fc_pred": dec["fc_pred"],
            "z": z,
            "cov": cov,
        }


class FUNGVA_MLP_FiLM1_Trainer(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        self.model = FUNGVA_MLP_FiLM1(latent_dim=latent_dim, cov_dim=cov_dim)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(x)


# -----------------------------------------------------------------------------
# Loss and metrics
# -----------------------------------------------------------------------------

def _get(preds: Any, name: str) -> torch.Tensor:
    if isinstance(preds, dict):
        return preds[name]
    return getattr(preds, name)


def cross_cov_penalty(mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    B = mu.size(0)
    mu_c = mu - mu.mean(dim=0, keepdim=True)
    cov_c = cov - cov.mean(dim=0, keepdim=True)
    cov_mat = torch.matmul(mu_c.t(), cov_c) / max(B - 1, 1)
    return torch.mean(cov_mat.pow(2))


def make_vae_loss(lambda_pen: float = 0.0) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    def vae_loss(preds: Any, target: torch.Tensor) -> torch.Tensor:
        recon = _get(preds, "recon")
        mu = _get(preds, "mu")
        logvar = _get(preds, "logvar")
        cov = _get(preds, "cov")
        B = target.size(0)
        recon_loss = 0.5 * F.mse_loss(recon, target, reduction="sum") / B
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) / B
        pen = cross_cov_penalty(mu, cov)
        return recon_loss + kl + lambda_pen * pen
    return vae_loss


def recon_metric(preds: Any, target: torch.Tensor) -> float:
    val = 0.5 * F.mse_loss(_get(preds, "recon"), target, reduction="sum") / target.size(0)
    return float(val.detach().cpu().item())


def kl_metric(preds: Any, target: Optional[torch.Tensor] = None) -> float:
    del target
    mu = _get(preds, "mu")
    logvar = _get(preds, "logvar")
    val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) / mu.size(0)
    return float(val.detach().cpu().item())


def pen_metric(preds: Any, target: Optional[torch.Tensor] = None) -> float:
    del target
    return float(cross_cov_penalty(_get(preds, "mu"), _get(preds, "cov")).detach().cpu().item())


class RunningMeanMetric:
    def __init__(self, metric_fn: Callable[[Any, torch.Tensor], float]):
        self.metric_fn = metric_fn
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, preds: Any, target: torch.Tensor) -> None:
        B = target.size(0)
        self.sum += self.metric_fn(preds, target) * B
        self.n += B

    def compute(self) -> float:
        return self.sum / max(self.n, 1)


class ActiveDimsRunningMetric:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.kl_sum: Optional[torch.Tensor] = None
        self.n = 0

    @torch.no_grad()
    def update(self, preds: Any, target: torch.Tensor) -> None:
        del target
        mu = _get(preds, "mu")
        logvar = _get(preds, "logvar")
        kl_mat = -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))
        kl_dim_sum = kl_mat.sum(dim=0).detach().cpu()
        self.kl_sum = kl_dim_sum if self.kl_sum is None else self.kl_sum + kl_dim_sum
        self.n += mu.size(0)

    def compute(self) -> int:
        if self.kl_sum is None or self.n == 0:
            return 0
        return int(((self.kl_sum / self.n) > self.threshold).sum().item())


def make_running_metrics(threshold: float = 0.01) -> Dict[str, Any]:
    return {
        "recon": RunningMeanMetric(recon_metric),
        "kl": RunningMeanMetric(kl_metric),
        "penalty": RunningMeanMetric(pen_metric),
        "active_dims": ActiveDimsRunningMetric(threshold=threshold),
    }


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    return x


def infer_batch_size(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        if "y" in batch and torch.is_tensor(batch["y"]):
            return int(batch["y"].shape[0])
        for value in batch.values():
            try:
                return infer_batch_size(value)
            except (TypeError, IndexError):
                pass
    if isinstance(batch, (tuple, list)):
        for value in batch:
            try:
                return infer_batch_size(value)
            except (TypeError, IndexError):
                pass
    raise TypeError("Could not infer batch size.")


def model_forward(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return model(batch["x"])


def compute_loss(loss_fn: Callable[[Any, torch.Tensor], torch.Tensor], output: Any, batch: Dict[str, Any]) -> torch.Tensor:
    return loss_fn(output, batch["y"])


def update_metrics(metric_objs: Dict[str, Any], output: Any, batch: Dict[str, Any]) -> None:
    target = batch["y"]
    for metric in metric_objs.values():
        metric.update(output, target)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[Any],
    loss_fn: Callable[..., Any],
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    metrics = make_running_metrics(threshold=0.01)

    for batch in data_loader:
        batch = move_to_device(batch, device)
        output = model_forward(model, batch)
        loss = compute_loss(loss_fn, output, batch)
        batch_n = infer_batch_size(batch)
        total_loss += float(loss.detach().cpu()) * batch_n
        total_n += batch_n
        update_metrics(metrics, output, batch)

    metric_values = {k: v.compute() for k, v in metrics.items()}
    return total_loss / max(total_n, 1), metric_values


def fit_one_seed_MLP_FiLM1(
    seed_id: int,
    cov_dim: int,
    train_dl: DataLoader,
    val_dl: DataLoader,
    latent_dim: int = 10,
    lambda_pen: float = 0.1,
    max_lr: float = 0.00538324276491867,
    epochs: int = 800,
    patience: int = 30,
    model_dir: Optional[os.PathLike[str] | str] = None,
    device: Optional[str | torch.device] = None,
) -> nn.Module:
    print("\n---------------------------------")
    print(
        "MLP_FiLM1 run:",
        "latent_dim =", latent_dim,
        "lambda_pen =", lambda_pen,
        "seed =", seed_id,
        "max_lr =", max_lr,
    )
    print("---------------------------------")

    set_seed(seed_id)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    loss_fn = make_vae_loss(lambda_pen)
    model = FUNGVA_MLP_FiLM1_Trainer(latent_dim=latent_dim, cov_dim=cov_dim).to(device)
    optimizer = Adam(model.parameters())
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_dl),
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        train_metrics = make_running_metrics(threshold=0.01)

        for batch in train_dl:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model_forward(model, batch)
            loss = compute_loss(loss_fn, output, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_n = infer_batch_size(batch)
            running_loss += float(loss.detach().cpu()) * batch_n
            running_n += batch_n
            with torch.no_grad():
                update_metrics(train_metrics, output, batch)

        train_loss = running_loss / max(running_n, 1)
        train_metric_values = {k: v.compute() for k, v in train_metrics.items()}
        val_loss, val_metric_values = evaluate(model, val_dl, loss_fn, device)

        print(
            f"Epoch {epoch:04d}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_recon={train_metric_values['recon']:.6f} | "
            f"train_kl={train_metric_values['kl']:.6f} | "
            f"train_pen={train_metric_values['penalty']:.6f} | "
            f"train_active_dims={train_metric_values['active_dims']} | "
            f"val_recon={val_metric_values['recon']:.6f} | "
            f"val_kl={val_metric_values['kl']:.6f} | "
            f"val_pen={val_metric_values['penalty']:.6f} | "
            f"val_active_dims={val_metric_values['active_dims']}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f"MLP_FiLM1_ld10_lam0.1_seed_{seed_id}.pt"
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
        print(f"Saved best model to: {model_file}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP_FiLM1 ld10 lambda=0.1 as one standalone Python script.")
    parser.add_argument("--workdir", default=".", help="Directory containing python_data/fc_data.h5 and python_data/covariates.parquet.")
    parser.add_argument("--h5-relpath", default="python_data/fc_data.h5")
    parser.add_argument("--covariates-relpath", default="python_data/covariates.parquet")
    parser.add_argument("--model-dir", default="MLP_FiLM1_ld10_lam0.1", help="Directory for saved .pt files.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303], help="Training seeds.")
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--lambda-pen", type=float, default=0.1)
    parser.add_argument("--max-lr", type=float, default=0.00538324276491867)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=15)
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu, cuda, cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dl, val_dl, _test_dl, cov_dim = build_dataloaders(
        workdir=args.workdir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        h5_relpath=args.h5_relpath,
        covariates_relpath=args.covariates_relpath,
    )

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    for sd in args.seeds:
        fit_one_seed_MLP_FiLM1(
            seed_id=sd,
            cov_dim=cov_dim,
            train_dl=train_dl,
            val_dl=val_dl,
            latent_dim=args.latent_dim,
            lambda_pen=args.lambda_pen,
            max_lr=args.max_lr,
            epochs=args.epochs,
            patience=args.patience,
            model_dir=args.model_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
