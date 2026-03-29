import torch
import torch.nn as nn
import torch.nn.functional as F


class RadioMLCNN(nn.Module):
    """RadioML 2016.10a classifier (matches cnn.ipynb)."""

    def __init__(self, num_classes: int = 11, dropout_rate: float = 0.5):
        super().__init__()
        self.pad1 = nn.ZeroPad2d((1, 1, 0, 0))
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), padding=0)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), padding=0)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.flatten_size = 80 * 1 * 130
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad1(x)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = self.pad2(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
