import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv


class Fusion(nn.Module):
    def __init__(self, HiddenSize, Window):
        super(Fusion, self).__init__()
        self.SpatialLin1 = nn.Linear(HiddenSize * Window * Window, 32)
        self.SpatialLin2 = nn.Linear(32, HiddenSize)

        self.SpectralLin1 = nn.Linear(16, HiddenSize)
        self.FusionLin = nn.Linear(HiddenSize * 2, 2)

    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2] * x1.shape[3])
        x1 = F.relu(self.SpatialLin1(x1))
        x1 = F.dropout(x1, 0.2)
        x1 = F.relu(self.SpatialLin2(x1))

        x2 = F.relu(self.SpectralLin1(x2))
        x = self.FusionLin(torch.cat([x1, x2], dim=-1))
        return x


class Discriminator(nn.Module):
    def __init__(self, InputSize, DropOut=0.2):
        super(Discriminator, self).__init__()
        self.RNN1 = nn.RNN(1, 8, num_layers=1, batch_first=True, bidirectional=False)
        self.RNN2 = nn.RNN(8, 16, num_layers=1, batch_first=True, bidirectional=False)
        self.RNN3 = nn.RNN(16, 8, num_layers=1, batch_first=True, bidirectional=False)
        self.RNN4 = nn.RNN(8, 1, num_layers=1, batch_first=True, bidirectional=False)

        self.Conv1 = nn.Conv2d(InputSize, 32, 3)
        self.Conv2 = nn.Conv2d(32, 16, 3)
        self.Conv3 = nn.Conv2d(16, 8, 3)
        self.dropout = DropOut
        self.Fusion = Fusion(8, 1)

    def forward(self, SpatialX, SpectrumX):
        x = SpectrumX.reshape(SpectrumX.shape[0], SpectrumX.shape[1], 1)
        x = F.relu(self.RNN1(x)[0])
        x = F.dropout(x, 0.2)
        x = F.relu(self.RNN2(x)[0])
        x = F.dropout(x, 0.2)
        x = F.relu(self.RNN3(x)[0])
        x = F.dropout(x, 0.2)
        x = self.RNN4(x)[0]
        x1 = x.reshape(x.shape[0], x.shape[1])

        x2 = F.relu(self.Conv1(SpatialX))
        x2 = F.relu(self.Conv2(x2))
        x2 = self.Conv3(x2)
        x3 = self.Fusion(x2, x1)
        return x3


class SpatialGenerator(nn.Module):
    def __init__(self, InputSize, DropOut=0.3):
        super(SpatialGenerator, self).__init__()
        self.Conv1 = nn.Conv2d(InputSize, 32, 3)
        self.Conv2 = nn.Conv2d(32, 64, 3)
        self.Conv3 = nn.Conv2d(64, 128, 3)
        self.Conv4 = nn.Conv2d(128, 256, 1)
        self.TransConv1 = nn.Conv2d(256, 128, 1)
        self.TransConv2 = nn.ConvTranspose2d(128, 32, 3)
        self.TransConv3 = nn.ConvTranspose2d(32, 64, 3)
        self.TransConv4 = nn.ConvTranspose2d(64, InputSize, 3)
        self.dropout = DropOut

    def forward(self, x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.Conv2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.Conv3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.Conv4(x)


        x = self.TransConv1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.TransConv2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.TransConv3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.TransConv4(x)
        return x


class SpectrumGenerator(nn.Module):
    def __init__(self):
        super(SpectrumGenerator, self).__init__()
        self.RNN1 = nn.RNN(1, 8, num_layers=1, batch_first=True, bidirectional=False)
        self.RNN2 = nn.RNN(8, 16, num_layers=1, batch_first=True, bidirectional=False)
        self.RNN3 = nn.RNN(16, 8, num_layers=1, batch_first=True, bidirectional=False)
        self.RNN4 = nn.RNN(8, 1, num_layers=1, batch_first=True, bidirectional=False)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1)
        x = F.relu(self.RNN1(x)[0])
        x = F.dropout(x, 0.2)
        x = F.relu(self.RNN2(x)[0])

        x = F.dropout(x, 0.2)
        x = F.relu(self.RNN3(x)[0])
        x = F.dropout(x, 0.2)
        x = self.RNN4(x)[0]
        x = x.reshape(x.shape[0], x.shape[1])
        return x


