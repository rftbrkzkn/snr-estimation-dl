# Çalışma Planı

---

## Adım 1 — Veri üreteci

- QPSK, birim ortalama sembol gücü
- AWGN, SNR −10 … +20 dB
- Etiket: SNR (dB cinsinden)
- Eğitim / doğrulama: SNR uniform rastgele
- Test: −10:2:20 dB sabit grid, her noktada 1000 frame

Kontrol: 20 dB'de constellation'da 4 net küme görülmeli.

---

## Adım 2 — M2M4 baseline

M2 = mean( |r|² )

M4 = mean( |r|⁴ )

S_hat = sqrt( 2·M2² − M4 )

N_hat = M2 − S_hat

SNR_hat = 10 · log10( S_hat / N_hat )

Not: (2·M2² − M4) < 0 durumları clamp edilecek ve sayısı kaydedilecek.

---

## Adım 3 — Modeller

| Model | Girdi | Not |
|---|---|---|
| Feature + MLP | M2, M4, M6, genlik istatistikleri | En hafif |
| MLP | Ham I/Q (2N) | Basit deep baseline |
| 1D CNN | 2 kanal × N | Ana model |

Ortak kurallar: aynı veri, aynı split, aynı loss (MSE, dB), aynı seed.

---

## Adım 4 — Deneyler

1. AWGN, N = 1024, tüm yöntemler
2. Frame uzunluğu taraması: N = 128 / 256 / 512 / 1024
3. Robustness: faz kayması, frekans kayması
4. (Opsiyonel) Constellation 2D histogram + 2D CNN

---

## Figür Listesi (bildiri)

1. Constellation diyagramları — 20 / 10 / 0 / −5 dB
2. RMSE vs SNR — tüm yöntemler (ana figür)
3. RMSE vs frame uzunluğu (N)
4. Tahmin vs gerçek SNR scatter — CNN ve M2M4
5. Bias vs SNR
6. Robustness: bozucu etkiler altında RMSE
7. Eğitim eğrileri (training / validation loss)

## Tablolar

- Parametre sayısı, inference süresi, ortalama RMSE
