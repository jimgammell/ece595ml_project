import numpy as np
import torch
from torch import nn

class MultilayerPerceptron(nn.Module):
    def __init__(self, 
                 layers, 
                 hidden_activation, 
                 output_activation,
                 input_shape,
                 output_shape):
        super(MultilayerPerceptron, self).__init__()
        layers = [np.prod(input_shape)] + layers + [np.prod(output_shape)]
        nn_layers = []
        for li, lo in zip(layers[:-2], layers[1:-1]):
            nn_layers.append(nn.Linear(li, lo))
            nn_layers.append(hidden_activation)
        nn_layers.append(nn.Linear(layers[-2], layers[-1]))
        if output_activation != None:
            nn_layers.append(output_activation)
        
        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        self.model = nn.Sequential(*nn_layers)
    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], np.prod(self.input_shape[1:])))
        x = self.model(x)
        x = torch.reshape(x, [x.shape[0]]+self.output_shape)
        return x

