"""VIST-NRF method (Stage C).

Architecture per specification section 9:

  State encoder  : time-aware GRU/GRU-D over values, masks, time gaps  -> z_state
  Policy encoder : summarizes mask/timing patterns                     -> z_policy
  Prediction head: primarily z_state; ablated variant uses
                   z_state + z_policy to test whether timing/policy
                   information adds predictive value beyond state.

Separation objective (ablated individually):
  - gradient-reversal adversarial environment head on z_state (DANN-style;
    encoder learns to make z_state environment-unpredictable)
  - cross-environment risk penalty (InvarianceLoss: irm / vrex / groupdro)

Spec cautions (§9):
  - an adversarial hospital classifier does NOT automatically identify patient
    state and may remove clinically useful hospital-correlated information.
  - ablate individually; test environment leakage, representation collapse,
    predictive performance, and component ablations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vist_nrf.models.baselines import BaseModel, _proba_2col
from vist_nrf.losses.invariance import InvarianceLoss


class _GradientReversal(torch.autograd.Function):
    """Gradient-reversal layer: forward identity, backward negates the gradient.

    Used for DANN-style adversarial environment confusion: the environment
    classifier is trained to predict env from z_state (normal gradient), while
    the state encoder receives the negated gradient (learns to make z_state
    environment-unpredictable).
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output, None


class _StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim * 3, hidden_dim, n_layers,
                          batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask, delta):
        x = torch.cat([values, mask, delta], dim=-1)
        _, h_n = self.gru(x)
        return self.dropout(h_n[-1])


class _PolicyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, mask, delta):
        obs_rate = mask.mean(dim=1)
        mean_gap = delta.mean(dim=1)
        return self.mlp(torch.cat([obs_rate, mean_gap], dim=-1))


class _PredictionHead(nn.Module):
    def __init__(self, state_dim, policy_dim, use_policy=False):
        super().__init__()
        self.use_policy = use_policy
        in_dim = state_dim + (policy_dim if use_policy else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, z_state, z_policy=None):
        z = torch.cat([z_state, z_policy], dim=-1) if self.use_policy else z_state
        return self.mlp(z).squeeze(-1)


class VISTNRFModel(nn.Module, BaseModel):
    """VIST-NRF: state-focused prediction with optional policy separation."""

    def __init__(self, input_dim, n_envs=2, hidden_dim=64, policy_dim=32,
                 use_policy=False, use_adversarial=False,
                 invariance_method="erm", invariance_weight=1.0,
                 lr=1e-3, epochs=20, batch_size=64, seed=0, device="cpu"):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.n_envs = n_envs
        self.hidden_dim = hidden_dim
        self.policy_dim = policy_dim
        self.use_policy = use_policy
        self.use_adversarial = use_adversarial
        self.invariance_method = invariance_method
        self.invariance_weight = invariance_weight
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

        self.state_encoder = _StateEncoder(input_dim, hidden_dim)
        self.policy_encoder = _PolicyEncoder(input_dim, policy_dim)
        self.head = _PredictionHead(hidden_dim, policy_dim, use_policy=use_policy)
        self.env_head = nn.Linear(hidden_dim, n_envs) if use_adversarial else None
        self.invariance = InvarianceLoss(invariance_method, invariance_weight)

        self._fitted = False
        self._mean = None

    # -- data prep ---------------------------------------------------------
    def _to_sequences(self, X):
        X = np.asarray(X, dtype=np.float32)
        values = np.nan_to_num(X, nan=0.0)
        mask = (~np.isnan(X)).astype(np.float32)
        delta = np.ones_like(values)  # flat feature matrix: 1-hour step
        # reshape to (B, T=1, F)
        return (values[:, None, :], mask[:, None, :], delta[:, None, :])

    # -- fit / predict -----------------------------------------------------
    def fit(self, X, y, envs=None, **kwargs):
        torch.manual_seed(self.seed)
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._mean = np.nan_to_num(np.nanmean(X, axis=0), nan=0.0)

        values, mask, delta = self._to_sequences(X)
        v = torch.tensor(values, device=self.device)
        m = torch.tensor(mask, device=self.device)
        d = torch.tensor(delta, device=self.device)
        y_t = torch.tensor(y, device=self.device)
        env_t = torch.tensor(np.asarray(envs), device=self.device) if envs is not None else None

        ds = torch.utils.data.TensorDataset(v, m, d, y_t)
        if env_t is not None:
            ds = torch.utils.data.TensorDataset(v, m, d, y_t, env_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()

        self.train()
        for _ in range(self.epochs):
            for batch in dl:
                if env_t is not None:
                    vb, mb, db, yb, eb = batch
                else:
                    vb, mb, db, yb = batch
                    eb = None
                opt.zero_grad()
                z = self.state_encoder(vb, mb, db)
                logits = self.head(z)
                loss = bce(logits, yb)

                # cross-environment risk penalty on the predictor
                if eb is not None and self.invariance.method != "erm":
                    loss = loss + self.invariance(logits, yb, eb)

                # adversarial environment head (DANN gradient reversal)
                if self.use_adversarial and self.env_head is not None and eb is not None:
                    z_rev = _GradientReversal.apply(z)
                    env_logits = self.env_head(z_rev)
                    env_loss = F.cross_entropy(env_logits, eb.long())
                    loss = loss + self.invariance_weight * env_loss

                loss.backward()
                opt.step()
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        self.eval()
        values, mask, delta = self._to_sequences(np.asarray(X, dtype=np.float32))
        v = torch.tensor(values, device=self.device)
        m = torch.tensor(mask, device=self.device)
        d = torch.tensor(delta, device=self.device)
        with torch.no_grad():
            z = self.state_encoder(v, m, d)
            logits = self.head(z)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return _proba_2col(p)

    # -- save / load -------------------------------------------------------
    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim, "n_envs": self.n_envs,
            "hidden_dim": self.hidden_dim, "policy_dim": self.policy_dim,
            "use_policy": self.use_policy, "use_adversarial": self.use_adversarial,
            "invariance_method": self.invariance_method,
            "invariance_weight": self.invariance_weight,
            "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size,
            "seed": self.seed, "device": self.device,
            "mean": self._mean,
        }, p)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(input_dim=ck["input_dim"], n_envs=ck["n_envs"],
                    hidden_dim=ck["hidden_dim"], policy_dim=ck["policy_dim"],
                    use_policy=ck["use_policy"],
                    use_adversarial=ck["use_adversarial"],
                    invariance_method=ck["invariance_method"],
                    invariance_weight=ck["invariance_weight"],
                    lr=ck["lr"], epochs=ck["epochs"],
                    batch_size=ck["batch_size"], seed=ck["seed"],
                    device=ck["device"])
        model.load_state_dict(ck["state_dict"])
        model._mean = ck["mean"]
        model._fitted = True
        return model

    @property
    def name(self):
        parts = ["vist_nrf"]
        if self.use_policy:
            parts.append("policy")
        if self.use_adversarial:
            parts.append("adv")
        parts.append(self.invariance_method)
        return "_".join(parts)
