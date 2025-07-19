import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_window, output_class_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_forecast = nn.Linear(hidden_size, 34)  # Forecast per timestep
        self.fc_class = nn.Linear(hidden_size, output_class_size)  # Final classification
        self.forecast_window = forecast_window

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]              # (B, hidden)
        forecast = self.fc_forecast(out)         # (B, T, forecast_window)
        classify = self.fc_class(last_hidden)    # (B, output_class_size)
        return forecast, classify
