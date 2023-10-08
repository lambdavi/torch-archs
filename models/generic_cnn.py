import torch
from torch import nn

class Gen_CNN(nn.Module):
    def __init__(self, n_classes, *args, **kwargs) -> None:
        super(Gen_CNN, self).__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0), # W1 =(W - K + 2P)/S + 1
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1) # W2 = (W1 - K)/S + 1
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=5, kernel_size=5, stride=1, padding=0), # W1 =(W - K + 2P)/S + 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1) # W2 = (W1 - K)/S + 1
        )
        self.fc = nn.Linear(1620, n_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out