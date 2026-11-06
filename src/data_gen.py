"""QPSK + AWGN frame üretimi.

Sinyal modeli:
    r[k] = s[k] + n[k]

    s[k] : QPSK sembolü, ortalama gücü 1  ->  (±1 ± j) / sqrt(2)
    n[k] : karmaşık Gauss gürültü, gücü sigma² = 10^(-SNR_dB / 10)

Sinyal gücü 1 olduğu için:
    SNR = 1 / sigma²
"""

import numpy as np

from src import config


def qpsk_symbols(n_frames, frame_len, rng):
    """Rastgele QPSK sembolleri üretir. Çıktı: (n_frames, frame_len) complex."""
    i = rng.choice([-1.0, 1.0], size=(n_frames, frame_len))
    q = rng.choice([-1.0, 1.0], size=(n_frames, frame_len))
    return (i + 1j * q) / np.sqrt(2)


def add_awgn(s, snr_db, rng):
    """Her frame'e kendi SNR değerine göre AWGN ekler.

    s      : (n_frames, frame_len) complex, ortalama gücü 1
    snr_db : (n_frames,) her frame'in SNR değeri (dB)
    """
    noise_power = 10.0 ** (-snr_db / 10.0)               # sigma²
    sigma = np.sqrt(noise_power / 2.0)[:, None]          # I ve Q'ya eşit bölünür
    noise = sigma * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
    return s + noise


def generate_frames(n_frames, frame_len=config.FRAME_LEN, snr_db=None, rng=None):
    """Etiketli frame'ler üretir.

    snr_db = None  -> her frame için SNR, [SNR_MIN_DB, SNR_MAX_DB] aralığında uniform rastgele
    snr_db = sayı  -> tüm frame'ler aynı SNR'da

    Dönüş:
        r      : (n_frames, frame_len) complex, alınan I/Q örnekleri
        snr_db : (n_frames,) float, etiketler (dB)
    """
    rng = rng if rng is not None else np.random.default_rng(config.SEED)

    if snr_db is None:
        snr_db = rng.uniform(config.SNR_MIN_DB, config.SNR_MAX_DB, size=n_frames)
    else:
        snr_db = np.full(n_frames, float(snr_db))

    s = qpsk_symbols(n_frames, frame_len, rng)
    r = add_awgn(s, snr_db, rng)
    return r, snr_db


def snr_test_grid():
    """Test SNR noktaları: -10, -8, ..., 20 dB."""
    return np.arange(config.SNR_MIN_DB, config.SNR_MAX_DB + 1e-9, config.SNR_TEST_STEP_DB)


def generate_test_set(frame_len=config.FRAME_LEN, n_per_snr=config.N_TEST_PER_SNR, rng=None):
    """Sabit SNR grid'inde test seti. Her SNR noktasında n_per_snr frame."""
    rng = rng if rng is not None else np.random.default_rng(config.SEED + 1)
    rs, ys = [], []
    for snr in snr_test_grid():
        r, y = generate_frames(n_per_snr, frame_len, snr_db=snr, rng=rng)
        rs.append(r)
        ys.append(y)
    return np.concatenate(rs), np.concatenate(ys)


def normalize_power(r):
    """Her frame'i birim ortalama güce ölçekler (alıcıdaki AGC'nin karşılığı).

    Bu adım olmadan model, toplam gücün (1 + sigma²) büyüklüğünden SNR'ı
    "kopya çekerek" öğrenebilir. Gerçek alıcıda sinyal gücü bilinmediği için
    modele sadece frame'in şekli verilir. M2M4 zaten ölçekten bağımsızdır.
    """
    p = np.mean(np.abs(r) ** 2, axis=1, keepdims=True)
    return r / np.sqrt(p)


def to_iq(r, normalize=True):
    """Complex frame'leri model girdisine çevirir: (n_frames, 2, frame_len) float32.

    Kanal 0 = I (gerçel kısım), kanal 1 = Q (sanal kısım).
    """
    if normalize:
        r = normalize_power(r)
    return np.stack([r.real, r.imag], axis=1).astype(np.float32)
