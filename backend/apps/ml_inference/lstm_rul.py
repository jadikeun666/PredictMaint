"""
apps/ml_inference/lstm_rul.py  (versi Giliran 3)

Model LSTM-RUL (PyTorch, CPU) — ml-pipeline.md §2: jendela PANJANG TETAP 6
vektor fitur -> regresi sequence-to-one -> RUL (jam).

JANGAN diimpor dari tasks.py: torch berat dan autodiscover Celery mengimpor
tasks.py di SEMUA worker. Hanya diimpor management command training/registrasi
dan test; worker inference (Sesi C) mengimpornya secara LAZY.

Target dilatih di ruang log: y = log1p(RUL jam), distandardisasi dengan
statistik data latih; prediksi dikembalikan ke jam (expm1, dipotong >= 0).
Hyperparameter DIKUNCI a priori (tidak di-tuning pada fold evaluasi).

Perubahan dari versi Giliran 2 (perilaku numerik IDENTIK; run evaluasi
Giliran 2 tidak perlu diulang): fit dan predict dipisah, loss dicatat dengan
.item() (menghilangkan UserWarning), ditambah save/load artefak.
"""
import os

import numpy as np
import torch
from torch import nn

DEFAULT_HP = {
    "hidden": 32,
    "layers": 1,
    "epochs": 40,
    "batch": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_lstm(X_train, y_train_hours, seed: int, hp=None):
    """Latih dari nol. Return (model, tstats {'mu','sd'}, final_train_loss)."""
    hp = {**DEFAULT_HP, **(hp or {})}
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    y_log = np.log1p(np.asarray(y_train_hours, dtype=np.float64))
    mu = float(y_log.mean())
    sd = float(max(y_log.std(), 1e-6))

    Xtr = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    ytr = torch.as_tensor(((y_log - mu) / sd).astype(np.float32))

    model = LSTMRegressor(Xtr.shape[2], hp["hidden"], hp["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    n = Xtr.shape[0]
    last_loss = float("nan")
    for _ in range(hp["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for s in range(0, n, hp["batch"]):
            b = perm[s:s + hp["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        last_loss = total / n
    return model, {"mu": mu, "sd": sd}, last_loss


def predict_lstm(model, tstats, X) -> np.ndarray:
    """X: (M, seq_len, F) fitur SUDAH ternormalisasi. Return RUL jam float64 (M,), >= 0."""
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)
    return np.clip(np.expm1(pred_norm * tstats["sd"] + tstats["mu"]), 0.0, None)


def fit_predict_lstm(X_train, y_train_hours, X_test, seed: int, hp=None):
    model, tstats, loss = fit_lstm(X_train, y_train_hours, seed, hp)
    return predict_lstm(model, tstats, X_test), loss


# --- artefak ----------------------------------------------------------------
def save_artifact(path, model, tstats, scaler_mean, scaler_std, feature_order, seq_len, hp, meta=None):
    """Hanya tipe polos + tensor (aman untuk torch.load(weights_only=True))."""
    hp = {**DEFAULT_HP, **(hp or {})}
    payload = {
        "state_dict": model.state_dict(),
        "n_features": int(len(feature_order)),
        "hidden": int(hp["hidden"]),
        "layers": int(hp["layers"]),
        "target_mu": float(tstats["mu"]),
        "target_sd": float(tstats["sd"]),
        "scaler_mean": [float(v) for v in scaler_mean],
        "scaler_std": [float(v) for v in scaler_std],
        "feature_order": list(feature_order),
        "seq_len": int(seq_len),
        "target_transform": "log1p(RUL jam), distandardisasi; inverse: expm1",
        "feature_transform": "signed_log1p lalu (x-mean)/std",
        "meta": dict(meta or {}),
    }
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_artifact(path):
    """Return (model[eval], tstats, scaler_mean ndarray, scaler_std ndarray, info dict)."""
    p = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMRegressor(p["n_features"], p["hidden"], p["layers"])
    model.load_state_dict(p["state_dict"])
    model.eval()
    tstats = {"mu": p["target_mu"], "sd": p["target_sd"]}
    info = {k: p[k] for k in ("feature_order", "seq_len", "target_transform", "feature_transform", "meta")}
    return model, tstats, np.array(p["scaler_mean"]), np.array(p["scaler_std"]), info
