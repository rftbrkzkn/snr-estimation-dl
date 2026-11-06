"""Tüm deney ayarları tek yerde. Başka dosyalarda sabit sayı yazma, buradan oku."""

import os

# Hızlı test modu: SNR_QUICK=1 ortam değişkeniyle veri boyutları ve epoch küçülür.
# Sadece kodun hatasız çalıştığını kontrol etmek için. Gerçek sonuçlar için kullanma.
QUICK = os.environ.get("SNR_QUICK") == "1"

# Tekrarlanabilirlik
SEED = 42

# Sinyal
MODULATION = "QPSK"
FRAME_LEN = 1024                      # N: frame başına I/Q örnek sayısı
FRAME_LENS_SWEEP = [128, 256, 512, 1024]

# SNR (dB)
SNR_MIN_DB = -10.0
SNR_MAX_DB = 20.0
SNR_TEST_STEP_DB = 2.0                # test grid: -10, -8, ..., 20

# Veri seti boyutları (frame sayısı)
N_TRAIN = 50_000                      # 50k x 2 x 1024 float32 ≈ 400 MB
N_VAL = 10_000
N_TEST_PER_SNR = 1000

# Eğitim
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3

if QUICK:
    N_TRAIN, N_VAL, N_TEST_PER_SNR, EPOCHS = 2000, 500, 50, 2

# Yollar
DATA_DIR = "data"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"
