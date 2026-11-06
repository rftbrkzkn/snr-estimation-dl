"""Bildiri kalitesinde figürler.

Tüm figürler hem PNG (300 dpi) hem PDF (vektör, LaTeX için) olarak kaydedilir.
Renkler sabit sırayla atanır: aynı yöntem her figürde aynı renkte görünür.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src import config

# Yöntem -> renk (her figürde sabit)
METHOD_COLORS = {
    "M2M4": "#2a78d6",
    "Feature + MLP": "#eb6834",
    "MLP": "#1baf7a",
    "1D CNN": "#eda100",
}
METHOD_MARKERS = {"M2M4": "o", "Feature + MLP": "s", "MLP": "^", "1D CNN": "D"}

INK = "#33332f"
GRID = "#e4e3dd"


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def save(fig, name, out_dir=config.FIGURES_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)


def plot_constellations(frames, snrs_db, name="constellation"):
    """Farklı SNR'larda constellation diyagramları, yan yana paneller.

    frames  : her SNR için bir complex frame listesi
    snrs_db : panel başlıkları için SNR değerleri
    """
    set_style()
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4), sharex=True, sharey=True)
    ideal = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    for ax, r, snr in zip(axes, frames, snrs_db):
        ax.scatter(r.real, r.imag, s=2, alpha=0.35, color=METHOD_COLORS["M2M4"], linewidths=0)
        ax.scatter(ideal.real, ideal.imag, s=30, marker="x", color=INK, linewidths=1.2)
        ax.set_title(f"SNR = {snr:g} dB", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("I")
    axes[0].set_ylabel("Q")
    fig.tight_layout()
    save(fig, name)
    return fig


def plot_metric_vs_snr(results, metric="rmse", name=None, ylabel=None):
    """Yöntemlerin metrik eğrileri (RMSE veya bias) — SNR'a göre.

    results : {yöntem_adı: per_snr_metrics DataFrame}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for method, df in results.items():
        ax.plot(df["snr_db"], df[metric], linewidth=2,
                marker=METHOD_MARKERS.get(method, "o"), markersize=5,
                color=METHOD_COLORS.get(method), label=method)
    if metric == "bias":
        ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Gerçek SNR (dB)")
    ax.set_ylabel(ylabel or {"rmse": "RMSE (dB)", "bias": "Bias (dB)"}[metric])
    if metric == "rmse":
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    save(fig, name or f"{metric}_vs_snr")
    return fig
