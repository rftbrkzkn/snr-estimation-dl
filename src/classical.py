"""Klasik SNR estimator: M2M4 (moment tabanlı, data-aided değil).

Sabit genlikli sinyaller (QPSK gibi) ve karmaşık Gauss gürültü için:

    M2 = mean( |r|² )        ->  S + N
    M4 = mean( |r|⁴ )        ->  S² + 4·S·N + 2·N²

Buradan:

    S_hat = sqrt( 2·M2² − M4 )
    N_hat = M2 − S_hat
    SNR_hat = 10 · log10( S_hat / N_hat )

Düşük SNR'da ve kısa frame'lerde (2·M2² − M4) veya N_hat tahmin hatası
yüzünden sıfırın altına düşebilir. Bu durumlar "geçersiz" sayılır ve
tahmin, sonlu kalması için sınırlanır (clip).
"""

import numpy as np

# Geçersiz durumlarda tahminin sıkıştırılacağı aralık (dB)
CLIP_DB = (-30.0, 50.0)


def m2m4(r, return_valid=False):
    """Her frame için M2M4 SNR tahmini (dB).

    r : (n_frames, frame_len) complex
    Dönüş: (n_frames,) tahmin dB; return_valid=True ise ayrıca geçerlilik maskesi.
    """
    p = np.abs(r) ** 2
    m2 = p.mean(axis=1)
    m4 = (p ** 2).mean(axis=1)

    s_sq = 2.0 * m2 ** 2 - m4
    valid = s_sq > 0

    s_hat = np.sqrt(np.clip(s_sq, 0.0, None))
    n_hat = m2 - s_hat
    valid &= n_hat > 0

    eps = 1e-12
    snr_lin = np.clip(s_hat, eps, None) / np.clip(n_hat, eps, None)
    snr_db = np.clip(10.0 * np.log10(snr_lin), *CLIP_DB)

    if return_valid:
        return snr_db, valid
    return snr_db
