"""Değerlendirme: SNR başına RMSE ve bias.

    hata  = tahmin_dB − gerçek_dB
    RMSE  = sqrt( mean( hata² ) )
    bias  = mean( hata )
"""

import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def per_snr_metrics(y_true, y_pred, valid=None):
    """Test grid'indeki her SNR noktası için RMSE, bias ve geçersiz oranı.

    Dönüş: pandas DataFrame, sütunlar: snr_db, rmse, bias, invalid_frac
    """
    rows = []
    for snr in np.unique(y_true):
        m = y_true == snr
        err = y_pred[m] - y_true[m]
        rows.append({
            "snr_db": snr,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "invalid_frac": float(1 - valid[m].mean()) if valid is not None else 0.0,
        })
    return pd.DataFrame(rows)
