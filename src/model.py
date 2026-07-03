import warnings

warnings.filterwarnings("ignore")

# This file builds our CNN + LSTM hybrid
import torch
import torch.nn as nn


# Step 1 -> defining the model class
# Every PyTorch model is a class that inherits from nn.Module. Think of it as the template:
class ElderCareModel(nn.Module):
    def __init__(self):
        super(ElderCareModel, self).__init__()

        # Step 2 -> Set the CNN Layer
        # This reads the spectrogram like an image, looking for textures:
        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, padding=1
            ),  # 1 channel in, 16 feature maps out
            nn.ReLU(),  # kill negative values
            nn.MaxPool2d(kernel_size=2),  # shrink the image by half
        )
        # kernel_size=3 means a 3×3 window slides across your spectrogram looking for patterns.
        # 16 means it learns 16 different pattern detectors.

        # Step 3 -> Set the LSTM Layer
        self.lstm = nn.LSTM(
            input_size=16 * 6,  # 16 feature maps × 6 frequency bins (after pooling)
            hidden_size=64,  # memory capacity of the LSTM
            batch_first=True,  # expects (batch, time, features)
        )

        # Step 4 -> Set the Decision Layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # 64 LSTM outputs → 1 single value
            # nn.Sigmoid(),  # squash to 0.0–1.0 probability
        )

    def forward(self, x):
        # x shape coming in: (batch, 1, 13, 173)

        x = self.cnn(x)
        # after CNN + pooling: (batch, 16, 6, 86)

        x = x.permute(0, 3, 1, 2)
        # rearrange for LSTM: (batch, 86, 16, 6)
        # time frames become the sequence LSTM reads

        x = x.reshape(x.shape[0], x.shape[1], -1)
        # flatten last two dims: (batch, 86, 96)

        x, _ = self.lstm(x)
        # LSTM reads the sequence: (batch, 86, 64)
        # we only need the LAST time step's output
        x = x[:, -1, :]
        # (batch, 64)

        x = self.classifier(x)
        # (batch, 1) → probability of distress

        return x


if __name__ == "__main__":
    model = ElderCareModel()

    # fake one audio clip — same shape as your real data
    dummy = torch.zeros(1, 1, 13, 173)
    output = model(dummy)

    print(output.shape)  # (1, 1) — one probability value
    print(output)
