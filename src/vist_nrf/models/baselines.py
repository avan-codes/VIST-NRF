"""VIST-NRF models: all share fit / predict_proba / save / load interface.

Specification sections 8-9: Stage A baselines first (logistic regression, XGBoost
values-only, XGBoost values+timing), Stage B robustness baselines (ERM, mask
augmentation, IRM/GroupDRO/V-REx), Stage C VIST-NRF method only if justified.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


class BaseModel(ABC):
    """Common interface for all VIST-NRF models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class, shape (n_samples, 2)."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


def _save_sklearn_model(model: BaseModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def _load_sklearn_model(path: str | Path) -> BaseModel:
    return joblib.load(path)


def _proba_2col(scores_1d: np.ndarray) -> np.ndarray:
    """Convert 1-D positive-class scores to (n, 2) probability matrix."""
    p = np.asarray(scores_1d, dtype=float).reshape(-1)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.stack([1.0 - p, p], axis=1)


class LogisticRegressionBaseline(BaseModel):
    """Logistic regression on values-only causal features (Stage A)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LogisticRegressionBaseline":
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return _proba_2col(self._model.predict_proba(X)[:, 1])

    def save(self, path: str | Path) -> None:
        _save_sklearn_model(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionBaseline":
        return _load_sklearn_model(path)

    @property
    def name(self) -> str:
        return "logistic_regression"


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


class XGBoostBaseline(BaseModel):
    """XGBoost on values-only or values+timing features (Stage A, optional dep)."""

    def __init__(self, max_depth: int = 6, n_estimators: int = 400,
                 learning_rate: float = 0.05, seed: int = 0,
                 use_timing: bool = False):
        if not _xgboost_available():
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost>=2.0"
            )
        import xgboost as xgb
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self.use_timing = use_timing
        self._model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=seed,
            eval_metric="auc",
            n_jobs=1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostBaseline":
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return _proba_2col(self._model.predict_proba(X)[:, 1])

    def save(self, path: str | Path) -> None:
        _save_sklearn_model(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostBaseline":
        return _load_sklearn_model(path)

    @property
    def name(self) -> str:
        return "xgboost_timing" if self.use_timing else "xgboost_values"


class GRUDModel(BaseModel):
    """GRU-D time-aware recurrent baseline (Stage A, PyTorch).

    Implements the GRU-D decay mechanism: missing values decay toward a learnable
    / data-driven mean using time gaps between observations.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 1,
                 dropout: float = 0.1, lr: float = 1e-3, epochs: int = 20,
                 batch_size: int = 64, seed: int = 0, device: str = "cpu"):
        import torch
        import torch.nn as nn
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self._model = None
        self._fitted = False
        self._mean = None

    def _build(self):
        import torch
        import torch.nn as nn

        class GRUD(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_layers, dropout):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                                  batch_first=True, dropout=dropout if n_layers > 1 else 0)
                self.decay_w = nn.Parameter(torch.Tensor(input_dim))
                self.decay_b = nn.Parameter(torch.Tensor(input_dim))
                nn.init.uniform_(self.decay_w, 0, 1)
                nn.init.constant_(self.decay_b, 0)
                self.fc = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, mask, delta):
                # x: (B, T, F), mask: (B, T, F), delta: (B, T, F) time gaps
                gamma = torch.exp(-torch.relu(self.decay_w * delta + self.decay_b))
                x = x * mask + self.mean_buffer * (1 - mask) * gamma
                # GRU-D standard: also decay hidden state; simplified here
                out, h_n = self.gru(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out).squeeze(-1)

        model = GRUD(self.input_dim, self.hidden_dim, self.n_layers, self.dropout).to(self.device)
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "GRUDModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._mean = np.nanmean(X, axis=0)
        self._mean = np.nan_to_num(self._mean, nan=0.0)

        X_t = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        # Build mask and delta as all-ones / zeros for feature-matrix input
        # (feature matrix already encodes mask as separate columns; here we treat X as values)
        mask = torch.ones_like(X_t)
        delta = torch.zeros_like(X_t)
        ds = TensorDataset(X_t, mask, delta, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._model = self._build()
        self._model.mean_buffer = torch.tensor(self._mean, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self._model.train()
        for _ in range(self.epochs):
            for xb, mb, db, yb in dl:
                opt.zero_grad()
                logits = self._model(xb, mb, db)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self._model is None:
            raise RuntimeError("Model not fitted")
        self._model.eval()
        X_t = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32, device=self.device)
        mask = torch.ones_like(X_t)
        delta = torch.zeros_like(X_t)
        with torch.no_grad():
            logits = self._model(X_t, mask, delta)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return _proba_2col(p)

    def save(self, path: str | Path) -> None:
        import torch
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self._model.state_dict(), "mean": self._mean,
                    "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers, "dropout": self.dropout}, p)

    @classmethod
    def load(cls, path: str | Path) -> "GRUDModel":
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(input_dim=ck["input_dim"], hidden_dim=ck["hidden_dim"],
                    n_layers=ck["n_layers"], dropout=ck["dropout"])
        model._mean = ck["mean"]
        model._model = model._build()
        model._model.load_state_dict(ck["state"])
        model._model.mean_buffer = torch.tensor(model._mean, dtype=torch.float32)
        model._fitted = True
        return model

    @property
    def name(self) -> str:
        return "grud"


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionBaseline,
    "xgboost": XGBoostBaseline,
    "xgboost_values": lambda **kw: XGBoostBaseline(**kw, use_timing=False),
    "xgboost_timing": lambda **kw: XGBoostBaseline(**kw, use_timing=True),
    "grud": GRUDModel,
}


def build_model(model_name: str, **kwargs) -> BaseModel:
    if model_name == "vist_nrf":
        from vist_nrf.models.vist_nrf import VISTNRFModel  # lazy: avoids circular import
        return VISTNRFModel(**kwargs)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[model_name]
    if callable(entry) and not isinstance(entry, type):
        return entry(**kwargs)
    return entry(**kwargs)
