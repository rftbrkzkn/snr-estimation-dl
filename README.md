# Deep Learning ile SNR Tahmini

Alınan I/Q örneklerinden **SNR (Signal-to-Noise Ratio)** değerini derin öğrenme modelleriyle tahmin eden ve sonuçları klasik **M2M4** estimator ile **RMSE** üzerinden karşılaştıran çalışma.

Hedef: ders ödevi + konferans bildirisi.

---

## Problem

Bir haberleşme alıcısı, gelen sinyalin ne kadar "temiz" olduğunu (SNR) bilmek zorundadır. Adaptive modulation, güç kontrolü ve demodülasyon kararları bu bilgiye dayanır.

Klasik yöntemler (ör. M2M4) basit ve hızlıdır, ancak **düşük SNR**'da ve **kısa frame**'lerde hataları büyür. Bu çalışmada, aynı I/Q verisinden SNR'ı doğrudan öğrenen modellerin bu bölgelerde daha başarılı olup olmadığı incelenir.

---

## Yöntemler

| Yöntem | Girdi | Tür |
|---|---|---|
| M2M4 | Ham I/Q | Klasik (baseline) |
| Feature + MLP | Moment tabanlı öznitelikler | Hafif ML |
| MLP | Ham I/Q (düzleştirilmiş, 2N) | Deep learning |
| 1D CNN | I/Q, 2 kanal × N | Deep learning (ana model) |

---

## Deney Kurulumu

- **Modülasyon:** QPSK, birim ortalama güç
- **Kanal:** AWGN (sonraki aşamada faz / frekans kayması)
- **SNR aralığı:** −10 dB … +20 dB
- **Frame uzunluğu (N):** 128, 256, 512, 1024
- **Metrik:** RMSE (dB), bias (dB)

Ayrıntılı plan: [`docs/plan.md`](docs/plan.md)
Akış şemaları: [`docs/flowcharts.md`](docs/flowcharts.md)

---

## Klasör Yapısı

```
snr-estimation-dl/
├── src/
│   ├── config.py        # Tüm ayarlar tek yerde (SNR aralığı, N, seed…)
│   ├── data_gen.py      # QPSK + AWGN frame üretimi
│   ├── classical.py     # M2M4 estimator
│   ├── features.py      # Feature + MLP için öznitelik çıkarımı
│   ├── models.py        # MLP, 1D CNN
│   ├── train.py         # Eğitim döngüsü
│   ├── evaluate.py      # RMSE, bias, SNR bazlı metrikler
│   └── plots.py         # Bildiri kalitesinde figürler
├── docs/                # Plan, flowchart'lar, notlar
├── notebooks/           # Keşif ve hızlı denemeler
├── results/
│   ├── figures/         # Üretilen grafikler
│   └── tables/          # Sonuç tabloları (csv)
├── paper/               # Bildiri taslağı
└── data/                # Üretilen veri (git'e eklenmez)
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Durum

- [x] Repo iskeleti
- [ ] Veri üreteci + constellation grafiği
- [ ] M2M4 baseline + ilk RMSE vs SNR eğrisi
- [ ] Feature + MLP
- [ ] MLP
- [ ] 1D CNN
- [ ] Frame uzunluğu deneyi
- [ ] Robustness deneyi (faz / frekans kayması)
- [ ] Bildiri taslağı
