"""Deep learning modelleri. Hepsi tek bir çıktı verir: SNR tahmini (dB)."""

import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """Elle çıkarılmış öznitelikler -> SNR. En hafif model."""

    def __init__(self, n_features, hidden=(64, 64)):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                 # x: (B, n_features)
        return self.net(x).squeeze(-1)


class RawMLP(nn.Module):
    """Ham I/Q (düzleştirilmiş, 2N) -> SNR."""

    def __init__(self, frame_len, hidden=(512, 256, 64), dropout=0.2):
        super().__init__()
        layers, d = [nn.Flatten()], 2 * frame_len
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                 # x: (B, 2, N)
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    """I/Q (2 kanal x N) -> SNR.

    Conv blokları yerel örüntüleri öğrenir; global average pooling sayesinde
    model frame uzunluğundan (N) bağımsızdır. Aynı model 128 ve 1024 örnekte
    çalışabilir.
    """

    def __init__(self, channels=(32, 64, 128), kernel_size=7, dropout=0.2):
        super().__init__()
        blocks, c_in = [], 2
        for c in channels:
            blocks += [
                nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # x: (B, 2, N)
        return self.head(self.features(x)).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
