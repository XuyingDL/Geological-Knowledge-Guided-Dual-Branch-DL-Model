# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import var
from osgeo import gdal
import torch.nn.functional as F
import torch
import Model
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Seeting random seed
seed = 4
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
device = "cuda"



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# Write the predicted results as tif files
def Predict(Hidden, TifName):

    Hidden = normalization(Hidden)

    XY = np.load(r"PointXY.npy").T
    Projection = str(np.load(r"Projection.npy"))

    XDistance = abs(XY[0, 0] - XY[:, 0])
    YDistance = abs(XY[0, 1] - XY[:, 1])
    ColumGap = np.min(XDistance[np.where(XDistance != 0)])
    IndexGap = np.min(YDistance[np.where(YDistance != 0)])
    Column = np.arange(XY[:, 0].min(), XY[:, 0].max() + ColumGap, ColumGap)
    Index = np.arange(XY[:, 1].max(), XY[:, 1].min() - IndexGap, -IndexGap)
    result = np.zeros([len(Index), len(Column)])
    result[:, :] = -99
    for i in range(len(Hidden)):
        result[np.where(Index == XY[i, 1])[0], np.where(Column == XY[i, 0])[0]] = Hidden[i]


    newdata = pd.DataFrame(result, index=Index, columns=Column)
    var_lon = newdata.columns.map(float)
    var_lon = var_lon.astype(np.float64)
    var_lat = newdata.index
    data_arr = np.asarray(newdata)
    LonMin, LatMax, LonMax, LatMin = [var_lon.min(), var_lat.max(), var_lon.max(), var_lat.min()]
    N_Lat = len(var_lat)
    N_Lon = len(var_lon)
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = TifName
    out_tif_name = out_tif_name.format(var)
    out_tif = driver.Create(out_tif_name, N_Lon, N_Lat, 1, gdal.GDT_Float32)  # 创建框架
    geotransform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
    out_tif.SetGeoTransform(geotransform)

    out_tif.SetProjection(Projection)
    out_tif.GetRasterBand(1).SetNoDataValue(-99)
    out_tif.GetRasterBand(1).WriteArray(data_arr)
    out_tif.FlushCache()
    out_tif = None


# 设置参数
InputSize = 16
HiddenSize = 8
epoch = 3000


SpatialInputNumpy = np.load(r"SpatialInput.npy")
SpectralInputNumpy = np.load(r"SpectrumInput.npy")
Noisy = np.load(r"Noisy.npy").reshape(15450, 16, 1)
NoisyCNN = np.load(r"NoisyCNN.npy")

SpatialInput = torch.FloatTensor(SpatialInputNumpy).to(device)
SpectralInput = torch.FloatTensor(SpectralInputNumpy).to(device)

Noisy = SpectralInputNumpy + Noisy
NoisyCNN = SpatialInputNumpy + NoisyCNN
isnan = np.isnan(NoisyCNN)
NoisyCNN[np.where(isnan)] = 0

NoisyInput = torch.FloatTensor(Noisy).to(device)
NoisyCNNInput = torch.FloatTensor(NoisyCNN).to(device)


SpatialGenerator = Model.SpatialGenerator(InputSize).to(device)
SpectralGenerator = Model.SpectrumGenerator().to(device)
Discriminator = Model.Discriminator(InputSize).to(device)
print(SpatialGenerator)
print(SpectralGenerator)
print(Discriminator)


Optim = optim.Adam([{'params': SpatialGenerator.parameters()},
                    {'params': SpectralGenerator.parameters()}],
                   lr=0.005, weight_decay=0.005)
DiscriminatorOptim = optim.Adam([{'params': Discriminator.parameters()}],
                                lr=0.00005, weight_decay=0.005)


Loss = []

# Model training
for i in range(epoch):

    Optim.zero_grad()
    DiscriminatorOptim.zero_grad()

    SpatialOut = SpatialGenerator(NoisyCNNInput)  # 15450*16*9*9
    SpectralOut = SpectralGenerator(NoisyInput)  # 15450*16
    FalseDiscriminatorOut = Discriminator(SpatialOut, SpectralOut)
    # label1
    NegativeLabel = torch.ones(FalseDiscriminatorOut.shape[0]).to(torch.long).to(device)
    NegativeLoss = F.cross_entropy(FalseDiscriminatorOut, NegativeLabel)
    Loss1 = NegativeLoss * 2


    DiscriminatorOptim.zero_grad()

    TrueDiscriminatorOut = Discriminator(SpatialInput, SpectralInput)

    FalseDiscriminatorOut = Discriminator(SpatialOut.detach(), SpectralOut.detach())
    # label2
    PositiveLabel = torch.ones(TrueDiscriminatorOut.shape[0]).to(torch.long).to(device)
    NegativeLabel = torch.zeros(FalseDiscriminatorOut.shape[0]).to(torch.long).to(device)
    Loss3 = (F.cross_entropy(TrueDiscriminatorOut, PositiveLabel) +
             F.cross_entropy(FalseDiscriminatorOut, NegativeLabel))


    ############### geological constrained term #############
    # SpatialError = F.l1_loss(SpatialOut, SpatialInput, reduction="none")
    # SpectralError = F.l1_loss(SpectralOut.reshape(15450, 16, 1), SpectralInput, reduction="none")
    # SpatialError = torch.mean(SpatialError, [1, 2, 3])
    # SpectralError = torch.mean(SpectralError, [1, 2])
    # Error = SpatialError + SpectralError
    # # Error = (Error - Error.min()) / (Error.max() - Error.min())
    # Loss4 = F.l1_loss(Error, fault)
    # loss = [Loss1.to('cpu').detach().numpy(), Loss3.to('cpu').detach().numpy(), Loss4.to('cpu').detach().numpy()]
    # Loss1 += Loss4
    # Loss3.backward(retain_graph=True)
    # Loss1.backward(retain_graph=True)
    #
    # DiscriminatorOptim.step()
    # Optim.step()
    # Loss.append(loss)
    # print("epoch: ", i, "\tGenerator Loss:", loss[0], "\tDiscriminator Loss", loss[1], "\tConstrain Loss", loss[2])
    ###########################################
    ######## no geological constrained term #######
    Loss3.backward(retain_graph=True)
    Loss1.backward(retain_graph=True)
    loss = [Loss1.to('cpu').detach().numpy(), Loss3.to('cpu').detach().numpy()]
    DiscriminatorOptim.step()
    Optim.step()
    Loss.append(loss)
    print("epoch: ", i, "\tGenerator Loss:", loss[0], "\tDiscriminator Loss", loss[1])
    ###########################################
Loss = np.array(Loss)
epochs = range(Loss.shape[0])
plt.plot(epochs, Loss[:, 0], label='Generator Loss')
plt.plot(epochs, Loss[:, 1], label='Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()

# Prediction
SpatialOut = SpatialGenerator(SpatialInput)  # 15450*16*9*9
SpectralOut = SpectralGenerator(SpectralInput)  # 15450*16
SpatialError = F.l1_loss(SpatialOut, SpatialInput, reduction="none")
SpectralError = F.l1_loss(SpectralOut.reshape(15450, 16, 1), SpectralInput, reduction="none")
SpatialError = torch.mean(SpatialError, [1, 2, 3])
SpectralError = torch.mean(SpectralError, [1, 2])
Error = SpatialError + SpectralError
Predict(Error.to("cpu").detach().numpy(), r"data5.tif")

