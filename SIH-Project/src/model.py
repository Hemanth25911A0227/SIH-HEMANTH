"""Phase 4b — ForecastLSTM: 14-day multivariate window in, 3 sensor forecasts out."""
import torch.nn as nn

class ForecastLSTM(nn.Module):
    def __init__(self, n_features: int, n_sensors: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, n_sensors)

    def forward(self, x):                    # x: (B, lookback, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])         # forecast from the past window
