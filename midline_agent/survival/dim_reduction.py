#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pool slice features and compress patient-level vectors with a regularized autoencoder."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 8
AE_SEED = 50
HIDDEN_DIM = 4096
T1CE_MODALITY = "T1CE"
MODE_ALL_SLICES = "all_slices"
MODE_T1CE_KEYFRAME = "t1ce_keyframe"
AGGREGATION_MODES = (MODE_ALL_SLICES, MODE_T1CE_KEYFRAME)


class SliceSelectionError(ValueError):
    """The selected slices cannot be pooled into one patient-level hidden vector."""


def _as_hidden(patient_id: str, hidden: object) -> np.ndarray:
    if isinstance(hidden, torch.Tensor):
        array = hidden.detach().cpu().numpy()
    else:
        array = np.asarray(hidden)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != HIDDEN_DIM:
        raise SliceSelectionError(
            f"patient {patient_id} hidden must have shape (n_slices, {HIDDEN_DIM})"
        )
    if not bool(np.isfinite(array).all()):
        raise SliceSelectionError(f"patient {patient_id} hidden must be finite")
    return array


def _keyframe_label(metadata_row: object) -> int | None:
    if not isinstance(metadata_row, Mapping) or "keyframe_label" not in metadata_row:
        return None
    value = metadata_row["keyframe_label"]
    if type(value) is not int or value not in (0, 1):
        raise SliceSelectionError("keyframe_label must be integer 0 or 1")
    return value


def _selected_rows(patient_id: str, patient: Mapping, metadata: Mapping, mode: str) -> np.ndarray:
    if mode not in AGGREGATION_MODES:
        raise SliceSelectionError(f"unsupported aggregation mode: {mode}")
    hidden = _as_hidden(patient_id, patient.get("hidden"))
    n_slices = hidden.shape[0]
    modalities = patient.get("modalities")
    valid_mask = patient.get("valid_mask")
    slice_metadata = patient.get("slice_metadata")
    if not all(isinstance(value, Sequence) for value in (modalities, valid_mask, slice_metadata)):
        raise SliceSelectionError(f"patient {patient_id} is missing aligned slice fields")
    if len({n_slices, len(modalities), len(valid_mask), len(slice_metadata)}) != 1:
        raise SliceSelectionError(f"patient {patient_id} slice fields have inconsistent lengths")
    if any(type(flag) is not bool for flag in valid_mask):
        raise SliceSelectionError(f"patient {patient_id} valid_mask must contain booleans")

    keep = [bool(flag) for flag in valid_mask]
    if mode == MODE_T1CE_KEYFRAME:
        modality_map = metadata.get("modality_map") if isinstance(metadata, Mapping) else None
        if not isinstance(modality_map, Mapping) or T1CE_MODALITY not in modality_map:
            raise SliceSelectionError("modality_map must define T1CE")
        t1ce = modality_map[T1CE_MODALITY]
        if type(t1ce) is not int:
            raise SliceSelectionError("T1CE modality index must be an integer")
        # Keep valid T1CE slices first, then read keyframe_label. Other modalities need not have that label.
        for index, (flag, modality, row) in enumerate(zip(keep, modalities, slice_metadata)):
            if not flag or modality != t1ce:
                keep[index] = False
                continue
            label = _keyframe_label(row)
            if label is None:
                raise SliceSelectionError(
                    f"patient {patient_id} T1CE slice {index} is missing keyframe_label"
                )
            keep[index] = label == 1
    chosen = hidden[np.asarray(keep, dtype=bool)]
    if chosen.shape[0] == 0:
        raise SliceSelectionError(f"patient {patient_id} mode {mode} selected no slices")
    return chosen.mean(axis=0).astype(np.float32)


def aggregate_patient_hidden(
    patient: Mapping,
    metadata: Mapping,
    mode: str,
    patient_id: str = "patient",
) -> np.ndarray:
    """Pool one patient's selected slice hidden features into a single 4096-dimensional vector."""
    return _selected_rows(patient_id, patient, metadata, mode)


def aggregate_hidden(
    store: Mapping,
    mode: str,
    patient_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Return ``(n_patients, 4096)`` in the given order. The default order is the patient order in the store."""
    patients = store.get("patients")
    metadata = store.get("metadata")
    if not isinstance(patients, Mapping) or not patients:
        raise SliceSelectionError("feature store must contain patients")
    if not isinstance(metadata, Mapping):
        raise SliceSelectionError("feature store must contain metadata")
    order = tuple(patient_ids) if patient_ids is not None else tuple(patients)
    if not order:
        raise SliceSelectionError("patient_ids must be non-empty")
    rows = []
    for patient_id in order:
        if patient_id not in patients:
            raise SliceSelectionError(f"patient {patient_id} is not in the feature store")
        rows.append(aggregate_patient_hidden(patients[patient_id], metadata, mode, patient_id))
    return np.stack(rows).astype(np.float32)


class RegAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, h1: int = 512, h2: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_reg_ae(
    x_train: np.ndarray,
    x_val: np.ndarray,
    latent_dim: int,
    seed: int,
    noise: float = 0.0,
    max_epochs: int = 150,
    early_stop: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> RegAutoEncoder:
    """Train the regularized autoencoder and return the weights with the lowest validation loss."""
    if len(x_train) < 2:
        raise ValueError("train_reg_ae needs at least 2 training rows for BatchNorm")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = RegAutoEncoder(x_train.shape[1], latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(x_train) > batch_size,
    )
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val, patience = float("inf"), 0

    for _ in range(max_epochs):
        model.train()
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            bx_noisy = bx + noise * torch.randn_like(bx)
            loss = crit(model(bx_noisy), bx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(x_val_t), x_val_t).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def encode_reg_ae(model: RegAutoEncoder, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            model.encode(torch.tensor(x, dtype=torch.float32, device=DEVICE))
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def _scale_clip(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return np.clip(
        np.nan_to_num(scaler.transform(x), nan=0.0, posinf=0.0, neginf=0.0),
        -10,
        10,
    ).astype(np.float32)


def fit_encode(
    x_train: np.ndarray,
    x_apply: np.ndarray,
    latent_dim: int = LATENT_DIM,
    seed: int = AE_SEED,
    val_frac: float = 0.2,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the scaler on x_train and return the training and application embeddings. The final val_frac of x_train is reserved for autoencoder early stopping."""
    scaler = StandardScaler()
    x_fit = _scale_clip(x_train, scaler.fit(x_train))
    x_new = _scale_clip(x_apply, scaler)

    n_val = max(2, int(round(x_fit.shape[0] * val_frac)))
    n_train = x_fit.shape[0] - n_val
    if n_train < 2:
        raise ValueError(
            "fit_encode needs at least 2 training rows after holding out the tail "
            f"for early stopping; got {x_fit.shape[0]} rows and {n_val} validation rows"
        )
    model = train_reg_ae(
        x_fit[:-n_val],
        x_fit[-n_val:],
        latent_dim,
        seed=seed,
        noise=noise,
    )
    return encode_reg_ae(model, x_fit), encode_reg_ae(model, x_new)
