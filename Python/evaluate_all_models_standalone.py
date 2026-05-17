#!/usr/bin/env python3
"""
Standalone Python evaluation script for Graph-FUNGVA, MLP-FiLM1, and MLP-FiLM2.

This combines the R sourcing scripts plus both evaluation scripts:
  - libraries.R
  - datasets.R
  - loss_metrics.R
  - FUNGVA_model_definition.R
  - MLP_FiLM1_model_definition.R
  - MLP_FiLM2_model_definition.R
  - test_evaluation.R
  - test_evaluation_heatmaps.R

Expected data layout under --workdir:
  python_data/fc_data.h5
    X_train, X_val, X_test, train_ids, val_ids, test_ids
  python_data/covariates.parquet

Expected model directories under --workdir by default:
  FUNGVA_ld12_lam0.1/
  MLP_FiLM1_ld10_lam0.1/
  MLP_FiLM2_ld10_lam1/

The script loads PyTorch checkpoints saved by the standalone training scripts,
evaluates train/val/test reconstruction, computes group-difference recovery for
SEX and APOE4, saves CSV summaries, and generates PNG figures/heatmaps.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Use a non-interactive backend so this works on compute nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


P = (68 * 67) // 2
NUM_VARS = ["AGE", "EDUCATION", "pTau_ratio_log", "abeta_ratio_log"]
OTHER_VARS = ["SEX", "PARTNERED", "APOE4"]
REQUIRED_COVARS = ["RID", *NUM_VARS, *OTHER_VARS]


def set_seed(seed_id: int) -> None:
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_id)


def decode_ids(x: np.ndarray) -> np.ndarray:
    out = []
    for item in np.asarray(x).reshape(-1):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            s = str(item)
            if s.startswith("b'") and s.endswith("'"):
                s = s[2:-1]
            elif s.startswith('b"') and s.endswith('"'):
                s = s[2:-1]
            out.append(s)
    return np.asarray(out, dtype=str)


def rank_first(row: np.ndarray) -> np.ndarray:
    order = np.argsort(row, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(row) + 1, dtype=np.float32)
    return ranks


def normalize_fc_array(x: np.ndarray, name: str = "X") -> np.ndarray:
    """Return FC data as either (n, 2278) or (n, 68, 68), accepting common H5 layouts."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[1] == P:
            return x
        if x.shape[0] == P:
            return x.T
        raise ValueError(f"{name} has shape {x.shape}; expected (n, 2278) or (2278, n).")
    if x.ndim == 3:
        if x.shape[1:] == (68, 68):
            return x
        if x.shape[:2] == (68, 68):
            return np.transpose(x, (2, 0, 1))
        raise ValueError(f"{name} has shape {x.shape}; expected (n, 68, 68) or (68, 68, n).")
    raise ValueError(f"{name} has shape {x.shape}; expected 2D or 3D FC array.")


def lt_rowwise_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    out = np.empty(P, dtype=np.float32)
    k = 0
    for i in range(1, 68):
        out[k:k + i] = mat[i, :i]
        k += i
    return out


def fc_to_vectors(x: np.ndarray, name: str = "X") -> np.ndarray:
    x = normalize_fc_array(x, name=name)
    if x.ndim == 2:
        return x.astype(np.float32)
    return np.vstack([lt_rowwise_matrix(m) for m in x]).astype(np.float32)


def make_masks_from_train_x(x_train: np.ndarray) -> List[torch.Tensor]:
    x_train_norm = normalize_fc_array(x_train, name="X_train")
    if x_train_norm.ndim == 2:
        # If training data is already vectorized, reconstruct approximate matrices for mask ranking.
        mats = np.zeros((x_train_norm.shape[0], 68, 68), dtype=np.float32)
        for n, v in enumerate(x_train_norm):
            k = 0
            for i in range(1, 68):
                mats[n, i, :i] = v[k:k + i]
                mats[n, :i, i] = v[k:k + i]
                k += i
        x_train_norm = mats

    fc_train_array = 1.0 - x_train_norm.astype(np.float32)
    a_mat = np.mean(fc_train_array, axis=0)
    ranks = np.vstack([rank_first(row) for row in a_mat])
    n_size = 2
    masks_np = [
        (ranks < (n_size ** 1 + 1)).astype(np.float32),
        (ranks < (n_size ** 2 + 1)).astype(np.float32),
        (ranks < (n_size ** 3 + 1)).astype(np.float32),
        (ranks < (n_size ** 4 + 1)).astype(np.float32),
        (ranks < (n_size ** 5 + 1)).astype(np.float32),
        np.eye(P, dtype=np.float32),
    ]
    return [torch.tensor(m, dtype=torch.float32) for m in masks_np]


def fisher_atanh(x: np.ndarray) -> np.ndarray:
    eps = np.float32(1e-6)
    return np.arctanh(np.clip(x.astype(np.float32), -1 + eps, 1 - eps)).astype(np.float32)


def align_covariates(cov_df: pd.DataFrame, ids: np.ndarray, split: str) -> pd.DataFrame:
    missing_cols = [c for c in REQUIRED_COVARS if c not in cov_df.columns]
    if missing_cols:
        raise KeyError(f"covariates.parquet is missing required columns: {missing_cols}")

    ids = pd.Index(decode_ids(ids), name="RID")
    cov = cov_df.copy()
    cov["RID"] = cov["RID"].astype(str)
    available = set(cov["RID"])
    missing = [x for x in ids if x not in available]
    if missing:
        raise KeyError(
            f"{len(missing)} {split} IDs were not found in covariates.RID. "
            f"First missing IDs: {missing[:10]}"
        )
    return cov.set_index("RID").loc[ids].reset_index()


def scale_with_train(df: pd.DataFrame, means: pd.Series, sds: pd.Series) -> np.ndarray:
    x = df[NUM_VARS].to_numpy(dtype=np.float32)
    denom = sds.to_numpy(dtype=np.float32)
    if np.any(denom == 0):
        zero_vars = list(np.asarray(NUM_VARS)[denom == 0])
        raise ValueError(f"At least one training covariate SD is zero: {zero_vars}")
    return (x - means.to_numpy(dtype=np.float32)) / denom


class FungvaDataset(Dataset):
    def __init__(self, fc_mat: np.ndarray, cov_mat: np.ndarray):
        if fc_mat.shape[0] != cov_mat.shape[0]:
            raise ValueError(f"fc_mat rows {fc_mat.shape[0]} != cov_mat rows {cov_mat.shape[0]}")
        self.fc_mat = torch.tensor(fc_mat, dtype=torch.float32)
        self.cov_mat = torch.tensor(cov_mat, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.fc_mat.shape[0])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        fc = self.fc_mat[i]
        cov = self.cov_mat[i]
        return {"x": {"fc": fc, "cov": cov}, "y": fc, "indices": i}


def collate_fungva_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fc = torch.stack([b["x"]["fc"] for b in batch], dim=0)
    cov = torch.stack([b["x"]["cov"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    indices = torch.tensor([b["indices"] for b in batch], dtype=torch.long)
    return {"x": {"fc": fc, "cov": cov}, "y": y, "indices": indices}


def build_eval_data(workdir: Path, train_batch_size: int = 16, eval_batch_size: int = 15) -> Dict[str, Any]:
    data_dir = workdir / "python_data"
    h5_path = data_dir / "fc_data.h5"
    cov_path = data_dir / "covariates.parquet"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5 file: {h5_path}")
    if not cov_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {cov_path}")

    with h5py.File(h5_path, "r") as f:
        x_train_raw = f["X_train"][:]
        x_val_raw = f["X_val"][:]
        x_test_raw = f["X_test"][:]
        train_ids = decode_ids(f["train_ids"][:])
        val_ids = decode_ids(f["val_ids"][:])
        test_ids = decode_ids(f["test_ids"][:])

    covariates = pd.read_parquet(cov_path)
    masks = make_masks_from_train_x(x_train_raw)

    fc_train = fisher_atanh(fc_to_vectors(x_train_raw, "X_train"))
    fc_val = fisher_atanh(fc_to_vectors(x_val_raw, "X_val"))
    fc_test = fisher_atanh(fc_to_vectors(x_test_raw, "X_test"))

    cov_train = align_covariates(covariates, train_ids, "train")
    cov_val = align_covariates(covariates, val_ids, "val")
    cov_test = align_covariates(covariates, test_ids, "test")

    train_means = cov_train[NUM_VARS].mean(skipna=True)
    train_sds = cov_train[NUM_VARS].std(skipna=True, ddof=1)

    cov_train_mat = np.column_stack([
        scale_with_train(cov_train, train_means, train_sds),
        cov_train[OTHER_VARS].to_numpy(dtype=np.float32),
    ]).astype(np.float32)
    cov_val_mat = np.column_stack([
        scale_with_train(cov_val, train_means, train_sds),
        cov_val[OTHER_VARS].to_numpy(dtype=np.float32),
    ]).astype(np.float32)
    cov_test_mat = np.column_stack([
        scale_with_train(cov_test, train_means, train_sds),
        cov_test[OTHER_VARS].to_numpy(dtype=np.float32),
    ]).astype(np.float32)

    train_ds = FungvaDataset(fc_train, cov_train_mat)
    val_ds = FungvaDataset(fc_val, cov_val_mat)
    test_ds = FungvaDataset(fc_test, cov_test_mat)

    eval_train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fungva_batch)
    eval_val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fungva_batch)
    eval_test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fungva_batch)

    return {
        "cov_dim": int(cov_train_mat.shape[1]),
        "masks": masks,
        "splits": {
            "train": {"dl": eval_train_dl, "obs_z": fc_train, "cov_raw": cov_train, "ids": train_ids},
            "val": {"dl": eval_val_dl, "obs_z": fc_val, "cov_raw": cov_val, "ids": val_ids},
            "test": {"dl": eval_test_dl, "obs_z": fc_test, "cov_raw": cov_test, "ids": test_ids},
        },
    }


# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------

class GraphCNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mask: Optional[torch.Tensor] = None

    def set_mask(self, mask: torch.Tensor | np.ndarray) -> None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=self.linear.weight.dtype)
        self.mask = mask.detach().to(dtype=self.linear.weight.dtype, device=self.linear.weight.device)
        self.mask.requires_grad_(False)
        if self.mask.shape != self.linear.weight.shape:
            raise ValueError(f"Mask shape {tuple(self.mask.shape)} != weight shape {tuple(self.linear.weight.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            return F.linear(x, self.linear.weight * self.mask, self.linear.bias)
        return self.linear(x)


class FUNGVA(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        idx = torch.tril(torch.ones(68, 68, dtype=torch.bool), diagonal=-1).nonzero(as_tuple=False)
        self.register_buffer("lt_flat_idx", (idx[:, 0] * 68 + idx[:, 1]).long())

        self.en_mu_la1 = nn.Linear(P, 1024, bias=False)
        self.en_var_la1 = nn.Linear(P, 1024, bias=False)
        self.en_mu_la2 = nn.Linear(1024, 128, bias=False)
        self.en_var_la2 = nn.Linear(1024, 128, bias=False)
        self.en_mu_la3 = nn.Linear(128, latent_dim)
        self.en_var_la3 = nn.Linear(128, latent_dim)

        self.dc1_la1 = nn.Linear(latent_dim, 68)
        self.dc2_la1 = nn.Linear(latent_dim, 68)
        self.dc3_la1 = nn.Linear(latent_dim, 68)
        self.dc4_la1 = nn.Linear(latent_dim, 68)
        self.dc5_la1 = nn.Linear(latent_dim, 68)
        self.dc1_la2 = GraphCNN(68, 68)
        self.dc2_la2 = GraphCNN(68, 68)
        self.dc3_la2 = GraphCNN(68, 68)
        self.dc4_la2 = GraphCNN(68, 68)
        self.dc5_la2 = GraphCNN(68, 68)
        self.dcintercept = GraphCNN(P, P)

        film_hidden = 32
        self.film1 = nn.Sequential(nn.Linear(cov_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, 2 * 68))
        self.film2 = nn.Sequential(nn.Linear(cov_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, 2 * 68))
        self.film3 = nn.Sequential(nn.Linear(cov_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, 2 * 68))
        self.film4 = nn.Sequential(nn.Linear(cov_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, 2 * 68))
        self.film5 = nn.Sequential(nn.Linear(cov_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, 2 * 68))
        self.tanh_act = nn.Tanh()

    def set_mask(self, masks: Sequence[torch.Tensor | np.ndarray]) -> None:
        if len(masks) != 6:
            raise ValueError(f"Expected 6 masks; got {len(masks)}")
        self.dc1_la2.set_mask(masks[0])
        self.dc2_la2.set_mask(masks[1])
        self.dc3_la2.set_mask(masks[2])
        self.dc4_la2.set_mask(masks[3])
        self.dc5_la2.set_mask(masks[4])
        self.dcintercept.set_mask(masks[5])

    def encode(self, fc: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = self.en_mu_la3(F.relu(self.en_mu_la2(F.relu(self.en_mu_la1(fc)))))
        logvar = self.en_var_la3(F.relu(self.en_var_la2(F.relu(self.en_var_la1(fc)))))
        return {"mu": mu, "logvar": logvar}

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def film_modulate(h: torch.Tensor, cov: torch.Tensor, film_net: nn.Module) -> torch.Tensor:
        pars = film_net(cov)
        gamma = pars[:, :68]
        beta = pars[:, 68:136]
        return (1 + gamma) * h + beta

    def _decode_head(self, z: torch.Tensor, cov: torch.Tensor, first: nn.Module, second: nn.Module, film: nn.Module) -> torch.Tensor:
        h = F.relu(first(z))
        h = self.film_modulate(h, cov, film)
        h = second(h)
        outer = torch.bmm(h.unsqueeze(2), h.unsqueeze(1)).reshape(h.shape[0], 68 * 68)
        return outer.index_select(dim=1, index=self.lt_flat_idx)

    def decode(self, z: torch.Tensor, cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        dcl_out = (
            self._decode_head(z, cov, self.dc1_la1, self.dc1_la2, self.film1)
            + self._decode_head(z, cov, self.dc2_la1, self.dc2_la2, self.film2)
            + self._decode_head(z, cov, self.dc3_la1, self.dc3_la2, self.film3)
            + self._decode_head(z, cov, self.dc4_la1, self.dc4_la2, self.film4)
            + self._decode_head(z, cov, self.dc5_la1, self.dc5_la2, self.film5)
        )
        recon = self.dcintercept(dcl_out)
        return {"recon": recon, "fc_pred": self.tanh_act(recon)}

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fc = x["fc"]
        cov = x["cov"]
        enc = self.encode(fc)
        z = self.reparameterize(enc["mu"], enc["logvar"])
        dec = self.decode(z, cov)
        return {"recon": dec["recon"], "mu": enc["mu"], "logvar": enc["logvar"], "fc_pred": dec["fc_pred"], "z": z, "cov": cov}


class FUNGVA_Trainer(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int, masks: Optional[Sequence[torch.Tensor | np.ndarray]] = None):
        super().__init__()
        self.model = FUNGVA(latent_dim, cov_dim)
        if masks is not None:
            self.model.set_mask(masks)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(x)


class FUNGVA_MLP_FiLM1(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        self.en_mu_la1 = nn.Linear(P, 1024, bias=False)
        self.en_var_la1 = nn.Linear(P, 1024, bias=False)
        self.en_mu_la2 = nn.Linear(1024, 128, bias=False)
        self.en_var_la2 = nn.Linear(1024, 128, bias=False)
        self.en_mu_la3 = nn.Linear(128, latent_dim)
        self.en_var_la3 = nn.Linear(128, latent_dim)
        self.dc_la1 = nn.Linear(latent_dim, 128)
        self.dc_la2 = nn.Linear(128, 1024)
        self.dc_out = nn.Linear(1024, P)
        self.film1 = nn.Sequential(nn.Linear(cov_dim, 32), nn.ReLU(), nn.Linear(32, 2 * 128))
        self.tanh_act = nn.Tanh()

    def encode(self, fc: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = self.en_mu_la3(F.relu(self.en_mu_la2(F.relu(self.en_mu_la1(fc)))))
        logvar = self.en_var_la3(F.relu(self.en_var_la2(F.relu(self.en_var_la1(fc)))))
        return {"mu": mu, "logvar": logvar}

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def film_modulate(h: torch.Tensor, cov: torch.Tensor, film_net: nn.Module) -> torch.Tensor:
        pars = film_net(cov)
        out_dim = h.shape[1]
        gamma = pars[:, :out_dim]
        beta = pars[:, out_dim:2 * out_dim]
        return (1 + gamma) * h + beta

    def decode(self, z: torch.Tensor, cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        h1 = F.relu(self.dc_la1(z))
        h1 = self.film_modulate(h1, cov, self.film1)
        h2 = F.relu(self.dc_la2(h1))
        recon = self.dc_out(h2)
        return {"recon": recon, "fc_pred": self.tanh_act(recon)}

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encode(x["fc"])
        z = self.reparameterize(enc["mu"], enc["logvar"])
        dec = self.decode(z, x["cov"])
        return {"recon": dec["recon"], "mu": enc["mu"], "logvar": enc["logvar"], "fc_pred": dec["fc_pred"], "z": z, "cov": x["cov"]}


class FUNGVA_MLP_FiLM1_Trainer(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        self.model = FUNGVA_MLP_FiLM1(latent_dim, cov_dim)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(x)


class FUNGVA_MLP_FiLM2(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        self.en_mu_la1 = nn.Linear(P, 1024, bias=False)
        self.en_var_la1 = nn.Linear(P, 1024, bias=False)
        self.en_mu_la2 = nn.Linear(1024, 128, bias=False)
        self.en_var_la2 = nn.Linear(1024, 128, bias=False)
        self.en_mu_la3 = nn.Linear(128, latent_dim)
        self.en_var_la3 = nn.Linear(128, latent_dim)
        self.dc_la1 = nn.Linear(latent_dim, 128)
        self.dc_la2 = nn.Linear(128, 1024)
        self.dc_out = nn.Linear(1024, P)
        self.film1 = nn.Sequential(nn.Linear(cov_dim, 32), nn.ReLU(), nn.Linear(32, 2 * 128))
        self.film2 = nn.Sequential(nn.Linear(cov_dim, 32), nn.ReLU(), nn.Linear(32, 2 * 1024))
        self.tanh_act = nn.Tanh()

    def encode(self, fc: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = self.en_mu_la3(F.relu(self.en_mu_la2(F.relu(self.en_mu_la1(fc)))))
        logvar = self.en_var_la3(F.relu(self.en_var_la2(F.relu(self.en_var_la1(fc)))))
        return {"mu": mu, "logvar": logvar}

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def film_modulate(h: torch.Tensor, cov: torch.Tensor, film_net: nn.Module) -> torch.Tensor:
        pars = film_net(cov)
        out_dim = h.shape[1]
        gamma = pars[:, :out_dim]
        beta = pars[:, out_dim:2 * out_dim]
        return (1 + gamma) * h + beta

    def decode(self, z: torch.Tensor, cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        h1 = F.relu(self.dc_la1(z))
        h1 = self.film_modulate(h1, cov, self.film1)
        h2 = F.relu(self.dc_la2(h1))
        h2 = self.film_modulate(h2, cov, self.film2)
        recon = self.dc_out(h2)
        return {"recon": recon, "fc_pred": self.tanh_act(recon)}

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encode(x["fc"])
        z = self.reparameterize(enc["mu"], enc["logvar"])
        dec = self.decode(z, x["cov"])
        return {"recon": dec["recon"], "mu": enc["mu"], "logvar": enc["logvar"], "fc_pred": dec["fc_pred"], "z": z, "cov": x["cov"]}


class FUNGVA_MLP_FiLM2_Trainer(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int):
        super().__init__()
        self.model = FUNGVA_MLP_FiLM2(latent_dim, cov_dim)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(x)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def safe_cor(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return float("nan")
    return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))


def normalize_binary(x: Any) -> np.ndarray:
    s = pd.Series(x)
    if s.dtype.name == "category" or s.dtype == object:
        s = s.astype(str)
    return pd.to_numeric(s, errors="coerce").to_numpy()


@torch.no_grad()
def collect_preds(model: nn.Module, dl: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    pred_z, pred_r, obs_z = [], [], []
    for batch in dl:
        batch = move_to_device(batch, device)
        out = model(batch["x"])
        pred_z.append(out["recon"].detach().cpu().numpy())
        pred_r.append(out["fc_pred"].detach().cpu().numpy())
        obs_z.append(batch["y"].detach().cpu().numpy())
    return {"pred_z": np.vstack(pred_z), "pred_r": np.vstack(pred_r), "obs_z": np.vstack(obs_z)}


def compute_subject_metrics(obs_z: np.ndarray, pred_z: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    obs_r = np.tanh(obs_z)
    pred_r = np.tanh(pred_z)
    return pd.DataFrame({
        "subject_id": decode_ids(ids),
        "subject_corr": [safe_cor(obs_r[i, :], pred_r[i, :]) for i in range(obs_z.shape[0])],
        "subject_rmse": [safe_rmse(obs_z[i, :], pred_z[i, :]) for i in range(obs_z.shape[0])],
    })


def compute_group_metrics(obs_z: np.ndarray, pred_z: np.ndarray, cov_raw: pd.DataFrame, var: str) -> pd.DataFrame:
    g = normalize_binary(cov_raw[var])
    idx0 = np.where(g == 0)[0]
    idx1 = np.where(g == 1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return pd.DataFrame([{"covariate": var, "n_group0": len(idx0), "n_group1": len(idx1), "delta_corr": np.nan, "delta_rmse": np.nan}])
    obs_r = np.tanh(obs_z)
    pred_r = np.tanh(pred_z)
    delta_obs = obs_r[idx1, :].mean(axis=0) - obs_r[idx0, :].mean(axis=0)
    delta_pred = pred_r[idx1, :].mean(axis=0) - pred_r[idx0, :].mean(axis=0)
    return pd.DataFrame([{
        "covariate": var,
        "n_group0": len(idx0),
        "n_group1": len(idx1),
        "delta_corr": safe_cor(delta_obs, delta_pred),
        "delta_rmse": safe_rmse(delta_obs, delta_pred),
    }])


def vec_to_mat(v: np.ndarray, p: int = 68) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    m = np.zeros((p, p), dtype=np.float32)
    k = 0
    for i in range(1, p):
        m[i, :i] = v[k:k + i]
        m[:i, i] = v[k:k + i]
        k += i
    return m


def compute_delta(mat: np.ndarray, group_vec: Any) -> np.ndarray:
    g = normalize_binary(group_vec)
    idx0 = np.where(g == 0)[0]
    idx1 = np.where(g == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return np.full(mat.shape[1], np.nan, dtype=np.float32)
    return mat[idx1, :].mean(axis=0) - mat[idx0, :].mean(axis=0)


def make_model(model_key: str, cov_dim: int, masks: Sequence[torch.Tensor], device: torch.device) -> nn.Module:
    if model_key == "Graph_FUNGVA":
        model = FUNGVA_Trainer(latent_dim=12, cov_dim=cov_dim, masks=[m.to(device) for m in masks])
    elif model_key == "MLP_FiLM1":
        model = FUNGVA_MLP_FiLM1_Trainer(latent_dim=10, cov_dim=cov_dim)
    elif model_key == "MLP_FiLM2":
        model = FUNGVA_MLP_FiLM2_Trainer(latent_dim=10, cov_dim=cov_dim)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    return model.to(device)


def strip_prefix_if_present(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if all(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_torch_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    else:
        raise TypeError(
            f"Could not interpret checkpoint {path}. This script expects PyTorch .pt checkpoints "
            "saved by the standalone Python training scripts, not R luz_save objects."
        )

    clean = {k.replace("module.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(clean, strict=True)
        return
    except RuntimeError:
        pass

    # Some checkpoints may be saved from the inner model instead of the Trainer wrapper.
    if any(k.startswith("model.") for k in model.state_dict().keys()) and not any(k.startswith("model.") for k in clean.keys()):
        wrapped = {"model." + k: v for k, v in clean.items()}
        model.load_state_dict(wrapped, strict=True)
        return
    if not any(k.startswith("model.") for k in model.state_dict().keys()) and all(k.startswith("model.") for k in clean.keys()):
        unwrapped = strip_prefix_if_present(clean, "model.")
        model.load_state_dict(unwrapped, strict=True)
        return
    model.load_state_dict(clean, strict=True)


def find_checkpoint(workdir: Path, dir_path: str, prefixes: Sequence[str], seed: int) -> Path:
    d = workdir / dir_path
    candidates = []
    for prefix in prefixes:
        candidates.append(d / f"{prefix}{seed}.pt")
    candidates.extend(sorted(d.glob(f"*seed_{seed}.pt")))
    candidates.extend(sorted(d.glob(f"*{seed}.pt")))
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates[:12])
    raise FileNotFoundError(f"Could not find checkpoint for seed {seed} in {d}. Tried:\n  {tried}")


def save_boxplot_test(subject_df: pd.DataFrame, final_dir: Path) -> None:
    df = subject_df[subject_df["split"] == "test"].copy()
    labels = ["Graph-FUNGVA", "MLP-FiLM1", "MLP-FiLM2"]
    data = [df.loc[df["display_model"] == lab, "subject_corr"].dropna().to_numpy() for lab in labels]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.boxplot(data, labels=labels, showfliers=False)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            jitter = np.random.default_rng(123).normal(loc=i, scale=0.04, size=len(vals))
            ax.scatter(jitter, vals, alpha=0.45, s=12)
    ax.set_title("Subject-wise Correlation on Test Set", fontweight="bold")
    ax.set_ylabel("Observed vs Predicted Correlation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(final_dir / "Figure_Reconstruction_Test.png", dpi=300)
    plt.close(fig)


def save_generalization_plot(subject_summary_final: pd.DataFrame, final_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    splits = ["train", "val", "test"]
    split_labels = ["Train", "Validation", "Test"]
    x = np.arange(len(splits))
    for label, g in subject_summary_final.groupby("display_model"):
        vals = [g.loc[g["split"] == sp, "mean_all_subject_corr"].mean() for sp in splits]
        ax.plot(x, vals, marker="o", label=label)
    ax.set_xticks(x, split_labels)
    ax.set_title("Generalization Across Data Splits", fontweight="bold")
    ax.set_ylabel("Mean Subject-wise Correlation")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(final_dir / "Figure_Generalization_Gap.png", dpi=300)
    plt.close(fig)


def save_group_generalization(group_summary_final: pd.DataFrame, final_dir: Path) -> None:
    splits = ["train", "val", "test"]
    split_labels = ["Train", "Validation", "Test"]
    x = np.arange(len(splits))
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.8), sharex=True)
    for ax, cov in zip(axes, ["SEX", "APOE4"]):
        sub = group_summary_final[group_summary_final["covariate"] == cov]
        for label, g in sub.groupby("display_model"):
            vals = [g.loc[g["split"] == sp, "mean_delta_corr"].mean() for sp in splits]
            ax.plot(x, vals, marker="o", label=label)
        ax.set_xticks(x, split_labels)
        ax.set_title("Sex" if cov == "SEX" else "APOE4")
        ax.set_ylabel("Mean Delta Correlation")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(title="Model")
    fig.suptitle("Generalization of Group-Difference Recovery", fontweight="bold")
    fig.tight_layout()
    fig.savefig(final_dir / "Figure_Group_Generalization.png", dpi=300)
    plt.close(fig)


def save_combined_heatmap(delta_dict: Dict[str, np.ndarray], title: str, out_path: Path) -> None:
    panels = ["Observed", "Graph-FUNGVA", "MLP-FiLM1", "MLP-FiLM2"]
    mats = [vec_to_mat(delta_dict[p]) for p in panels]
    lim = np.nanmax(np.abs(np.stack(mats)))
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), constrained_layout=True)
    im = None
    for ax, label, mat in zip(axes, panels, mats):
        im = ax.imshow(mat, cmap="bwr", vmin=-lim, vmax=lim, origin="upper")
        ax.set_title(label, fontweight="bold")
        ticks = np.arange(0, 68, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks + 1, fontsize=6)
        ax.set_yticklabels(ticks + 1, fontsize=6)
    fig.suptitle(title, fontweight="bold")
    assert im is not None
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Delta FC")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Graph-FUNGVA, MLP-FiLM1, and MLP-FiLM2 checkpoints.")
    parser.add_argument("--workdir", default=".", help="Working directory containing python_data/ and model folders.")
    parser.add_argument("--output-dir", default="test_evaluation", help="Output directory for CSVs and figures.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0. Defaults to cuda if available else cpu.")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    output_dir = workdir / args.output_dir
    final_dir = output_dir / "final_tables_figures"
    heatmap_dir = output_dir / "heatmaps_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    data = build_eval_data(workdir, args.train_batch_size, args.eval_batch_size)
    cov_dim = data["cov_dim"]
    masks = data["masks"]
    splits = data["splits"]

    model_specs = [
        {
            "model": "Graph_FUNGVA",
            "display_model": "Graph-FUNGVA",
            "dir_path": "FUNGVA_ld12_lam0.1",
            "prefixes": ["graph_ld12_lam0.1_seed_", "FUNGVA_ld12_lam0.1_seed_"],
        },
        {
            "model": "MLP_FiLM1",
            "display_model": "MLP-FiLM1",
            "dir_path": "MLP_FiLM1_ld10_lam0.1",
            "prefixes": ["MLP_FiLM1_ld10_lam0.1_seed_"],
        },
        {
            "model": "MLP_FiLM2",
            "display_model": "MLP-FiLM2",
            "dir_path": "MLP_FiLM2_ld10_lam1",
            "prefixes": ["MLP_FiLM2_ld10_lam1_seed_"],
        },
    ]

    all_subject: List[pd.DataFrame] = []
    all_group: List[pd.DataFrame] = []
    pred_delta_apoe_by_model: Dict[str, List[np.ndarray]] = {s["display_model"]: [] for s in model_specs}
    pred_delta_sex_by_model: Dict[str, List[np.ndarray]] = {s["display_model"]: [] for s in model_specs}

    obs_test_r = np.tanh(splits["test"]["obs_z"])
    delta_obs_apoe = compute_delta(obs_test_r, splits["test"]["cov_raw"]["APOE4"])
    delta_obs_sex = compute_delta(obs_test_r, splits["test"]["cov_raw"]["SEX"])

    for spec in model_specs:
        for seed in args.seeds:
            ckpt = find_checkpoint(workdir, spec["dir_path"], spec["prefixes"], seed)
            print("\n====================================")
            print(f"Loading: {spec['display_model']} Seed: {seed}")
            print(f"Path: {ckpt}")
            print("====================================")

            set_seed(seed)
            model = make_model(spec["model"], cov_dim=cov_dim, masks=masks, device=device)
            load_torch_checkpoint(model, ckpt, device)
            model.eval()

            for split_name, split in splits.items():
                print(f"Evaluating: {split_name}")
                res = collect_preds(model, split["dl"], device)
                subj = compute_subject_metrics(split["obs_z"], res["pred_z"], split["ids"])
                subj["model"] = spec["model"]
                subj["display_model"] = spec["display_model"]
                subj["seed"] = seed
                subj["split"] = split_name
                all_subject.append(subj)

                grp = pd.concat([
                    compute_group_metrics(split["obs_z"], res["pred_z"], split["cov_raw"], "SEX"),
                    compute_group_metrics(split["obs_z"], res["pred_z"], split["cov_raw"], "APOE4"),
                ], ignore_index=True)
                grp["model"] = spec["model"]
                grp["display_model"] = spec["display_model"]
                grp["seed"] = seed
                grp["split"] = split_name
                all_group.append(grp)

                if split_name == "test":
                    pred_mat = res["pred_r"]
                    pred_delta_apoe_by_model[spec["display_model"]].append(compute_delta(pred_mat, split["cov_raw"]["APOE4"]))
                    pred_delta_sex_by_model[spec["display_model"]].append(compute_delta(pred_mat, split["cov_raw"]["SEX"]))

    subject_df = pd.concat(all_subject, ignore_index=True)
    group_df = pd.concat(all_group, ignore_index=True)

    subject_summary_by_seed = (
        subject_df.groupby(["model", "display_model", "seed", "split"], as_index=False)
        .agg(
            mean_subject_corr=("subject_corr", "mean"),
            sd_subject_corr=("subject_corr", "std"),
            mean_subject_rmse=("subject_rmse", "mean"),
            sd_subject_rmse=("subject_rmse", "std"),
        )
        .sort_values(["model", "seed", "split"])
    )

    subject_summary_final = (
        subject_summary_by_seed.groupby(["model", "display_model", "split"], as_index=False)
        .agg(
            mean_all_subject_corr=("mean_subject_corr", "mean"),
            sd_subject_corr=("mean_subject_corr", "std"),
            mean_all_subject_rmse=("mean_subject_rmse", "mean"),
            sd_subject_rmse=("mean_subject_rmse", "std"),
        )
        .sort_values(["split", "model"])
    )

    group_summary_by_seed = group_df.sort_values(["model", "seed", "split", "covariate"])
    group_summary_final = (
        group_summary_by_seed.groupby(["model", "display_model", "split", "covariate"], as_index=False)
        .agg(
            mean_delta_corr=("delta_corr", "mean"),
            sd_delta_corr=("delta_corr", "std"),
            mean_delta_rmse=("delta_rmse", "mean"),
            sd_delta_rmse=("delta_rmse", "std"),
            mean_n_group0=("n_group0", "mean"),
            mean_n_group1=("n_group1", "mean"),
        )
        .sort_values(["split", "covariate", "model"])
    )

    subject_df.to_csv(output_dir / "subject_metrics.csv", index=False)
    group_df.to_csv(output_dir / "group_metrics.csv", index=False)
    subject_summary_by_seed.to_csv(output_dir / "subject_summary_by_seed.csv", index=False)
    subject_summary_final.to_csv(output_dir / "subject_summary_final.csv", index=False)
    group_summary_by_seed.to_csv(output_dir / "group_summary_by_seed.csv", index=False)
    group_summary_final.to_csv(output_dir / "group_summary_final.csv", index=False)

    # Final compact tables.
    table_recon_test = subject_summary_final[subject_summary_final["split"] == "test"].copy()
    table_recon_test = pd.DataFrame({
        "Model": table_recon_test["display_model"],
        "Correlation (mean ± SD)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(table_recon_test["mean_all_subject_corr"], table_recon_test["sd_subject_corr"])],
        "RMSE (mean ± SD)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(table_recon_test["mean_all_subject_rmse"], table_recon_test["sd_subject_rmse"])],
    })
    table_recon_test.to_csv(final_dir / "Table_Reconstruction_Test.csv", index=False)

    table_group_test_src = group_summary_final[group_summary_final["split"] == "test"].copy()
    table_group_test_src["Covariate"] = table_group_test_src["covariate"].replace({"SEX": "Sex", "APOE4": "APOE4"})
    table_group_test = pd.DataFrame({
        "Covariate": table_group_test_src["Covariate"],
        "Model": table_group_test_src["display_model"],
        "Delta Correlation (mean ± SD)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(table_group_test_src["mean_delta_corr"], table_group_test_src["sd_delta_corr"])],
        "Delta RMSE (mean ± SD)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(table_group_test_src["mean_delta_rmse"], table_group_test_src["sd_delta_rmse"])],
    }).sort_values(["Covariate", "Model"])
    table_group_test.to_csv(final_dir / "Table_GroupDiff_Test.csv", index=False)

    print("\nReconstruction test table:")
    print(table_recon_test.to_string(index=False))
    print("\nGroup-difference test table:")
    print(table_group_test.to_string(index=False))

    save_boxplot_test(subject_df, final_dir)
    save_generalization_plot(subject_summary_final, final_dir)
    save_group_generalization(group_summary_final, final_dir)

    avg_apoe = {"Observed": delta_obs_apoe}
    avg_sex = {"Observed": delta_obs_sex}
    for label in ["Graph-FUNGVA", "MLP-FiLM1", "MLP-FiLM2"]:
        avg_apoe[label] = np.nanmean(np.vstack(pred_delta_apoe_by_model[label]), axis=0)
        avg_sex[label] = np.nanmean(np.vstack(pred_delta_sex_by_model[label]), axis=0)

    save_combined_heatmap(
        avg_apoe,
        "APOE4 Group Difference on Test Set (Carrier - Non-carrier)",
        heatmap_dir / "Combined_Heatmap_APOE4.png",
    )
    save_combined_heatmap(
        avg_sex,
        "SEX Group Difference on Test Set (Male - Female)",
        heatmap_dir / "Combined_Heatmap_SEX.png",
    )

    print(f"\nDONE. Results saved to: {output_dir}")
    print(f"Final tables and figures saved to: {final_dir}")
    print(f"Combined heatmaps saved to: {heatmap_dir}")


if __name__ == "__main__":
    main()
