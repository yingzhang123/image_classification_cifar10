import torch.nn as nn
from torchvision import models    # 一些预训练好的模型在torchvision的models中


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 10)

    def forward(self, x):
        out = self.model(x)
        return out


def pytorch_resnet18():
    return resnet18()