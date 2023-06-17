import torch
from torch import nn, optim
from torchvision.models import mobilenet_v2, resnet50

class ResNet50(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.rnet = resnet50(pretrained=True)
        self.freeze()
        self.rnet.fc = nn.Sequential(    
            nn.Linear(2048, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.rnet(x)

    def freeze(self):
        for param in self.rnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.rnet.parameters():
            param.requires_grad = True