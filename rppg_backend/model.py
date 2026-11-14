import torch
import torch.nn as nn


class HeartRateCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)


if __name__ == "__main__":
    model = HeartRateCNN()

    # Simulate one 30-second RGB recording
    # 9 channels, 600 samples at 20 Hz
    sample_input = torch.randn(1, 9, 600)

    output = model(sample_input)

    print("Model created successfully!")
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Predicted value (untrained):", output.item())
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )