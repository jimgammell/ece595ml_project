import torch
from torch import nn
from torchvision import models

# From https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/AlexNet.py
class AlexNet(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

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
    
    def finetune(self, enable):
        for layer in list(self.model.children())[:-1]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        logits = self.model(x)
        return logits