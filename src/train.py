"""Eğitim döngüsü: Adam + MSE (dB), learning rate düşürme ve early stopping."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE, patience=8, verbose=True):
    """Modeli eğitir, en iyi validation loss'taki ağırlıkları geri yükler.

    Dönüş: history sözlüğü (train_loss, val_loss, lr; her epoch için)
    """
    device = get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    tr = _loader(X_train, y_train, batch_size, shuffle=True)
    va = _loader(X_val, y_val, batch_size * 4, shuffle=False)

    hist = {"train_loss": [], "val_loss": [], "lr": []}
    best, best_state, wait = np.inf, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        train_loss = tot / len(tr.dataset)

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                tot += loss_fn(model(xb), yb).item() * len(xb)
        val_loss = tot / len(va.dataset)

        sched.step(val_loss)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["lr"].append(opt.param_groups[0]["lr"])

        if verbose:
            print(f"epoch {ep:3d} | train RMSE {np.sqrt(train_loss):6.3f} dB | "
                  f"val RMSE {np.sqrt(val_loss):6.3f} dB | lr {hist['lr'][-1]:.1e} | "
                  f"{time.time() - t0:4.1f}s")

        if val_loss < best - 1e-4:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping: {patience} epoch boyunca iyileşme yok.")
                break

    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict(model, X, batch_size=4096):
    """Tahmin (dB). Ayrıca frame başına ortalama inference süresini (ms) döndürür."""
    device = get_device()
    model.to(device).eval()
    out = []
    t0 = time.time()
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_frame = (time.time() - t0) * 1000 / len(X)
    return np.concatenate(out), ms_per_frame
