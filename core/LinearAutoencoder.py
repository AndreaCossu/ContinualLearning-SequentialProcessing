import torch
import torch.nn as nn
import numpy as np


class LinearAutoencoder(nn.Module):

    def __init__(self, dim_input, dim_hidden, device):
        super(LinearAutoencoder, self).__init__()

        self.dim_hidden = dim_hidden
        self.device = device

        self.A = nn.Linear(dim_input, dim_hidden, bias=False)
        self.B = nn.Linear(dim_hidden, dim_hidden ,bias=False)
        self.C = nn.Linear(dim_hidden, dim_input+dim_hidden, bias=False)


    def forward(self, x):
        
        x_decoded = torch.zeros_like(x).to(self.device)

        y = torch.zeros(x.size(0), self.dim_hidden).to(self.device)
        
        # encoding
        for i in range(x.size(1)):
            y = self.A(x[:, i, :]) + self.B(y)

        # decoding
        for i in range(x.size(1)-1, -1, -1):
            out = self.C(y)
            x_decoded[:, i, :] , y = out[:, :3], out[:, 3:] 

        return x_decoded
