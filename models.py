import torch
from torch import nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self,
                 input_shape,
                 blocks=18,
                 output_classes=10):
        super().__init__()
        attr_name = 'resnet%d'%(blocks)
        model_constructor = getattr(models, attr_name)
        model = model_constructor()
        num_features = list(model.children())[-1].in_features
        model.fc = nn.Linear(num_features, output_classes)
        self.model = model
        eg_input = torch.rand(input_shape)
        _ = self.model(eg_input)
        
    def forward(self, x):
        logits = self.model(x)
        return logits

class LeNet5(nn.Module):
    def __init__(self,
                 input_shape,
                 output_classes=10):
        super().__init__()
        layers = [nn.Conv2d(input_shape[1], 6, kernel_size=5, stride=1, padding=2, bias=True),
                  nn.BatchNorm2d(6),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),
                  nn.BatchNorm2d(16),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Flatten(),
                  nn.Linear(16*5*5, 120),
                  nn.ReLU(),
                  nn.Linear(120, 84),
                  nn.ReLU(),
                  nn.Linear(84, output_classes)]
        self.model = nn.Sequential(*layers)
        eg_input = torch.rand(input_shape)
        _ = self.model(eg_input)
    
    def forward(self, x):
        logits = self.model(x)
        return logits