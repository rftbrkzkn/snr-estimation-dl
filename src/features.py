"""Feature + MLP modeli için öznitelik çıkarımı.

Her frame güce göre normalize edildikten sonra (|r|² ortalaması = 1)
ölçekten bağımsız istatistikler hesaplanır:

    p = |r|²
    M4 / M2²,  M6 / M2³          ->  genlik dağılımının şekli (SNR ile değişir)
    std(|r|), std(p)             ->  genliğin ne kadar saçıldığı
    mean(|I|), mean(|Q|)         ->  QPSK'da yüksek SNR'da 1/sqrt(2)'ye yaklaşır
    mean(|r|⁴ · cos(4θ))         ->  4. derece faz istatistiği (QPSK'ya özgü)
    M2M4 tahmini (dB)            ->  klasik estimator'ın kendisi de bir öznitelik
"""

import numpy as np

from src.classical import m2m4
from src.data_gen import normalize_power

FEATURE_NAMES = [
    "m4_over_m2sq", "m6_over_m2cube", "std_abs", "std_pow",
    "mean_abs_i", "mean_abs_q", "fourth_order_phase", "m2m4_db",
]


def extract_features(r):
    """r: (n_frames, frame_len) complex  ->  (n_frames, 8) float32"""
    r = normalize_power(r)
    a = np.abs(r)
    p = a ** 2
    m2 = p.mean(axis=1)
    m4 = (p ** 2).mean(axis=1)
    m6 = (p ** 3).mean(axis=1)

    feats = np.stack([
        m4 / m2 ** 2,
        m6 / m2 ** 3,
        a.std(axis=1),
        p.std(axis=1),
        np.abs(r.real).mean(axis=1),
        np.abs(r.imag).mean(axis=1),
        np.real(np.mean(r ** 4, axis=1)) * -1.0,   # QPSK için s⁴ = -1, işareti düzeltiyoruz
        m2m4(r) / 10.0,                            # dB / 10: diğerleriyle benzer ölçek
    ], axis=1)
    return feats.astype(np.float32)


class Standardizer:
    """Öznitelikleri eğitim setinin ortalama/std'si ile ölçekler (val/test'e de aynısı uygulanır)."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)
