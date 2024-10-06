import glob
import math
import os

import pandas as pd
import shapely as sp
from osgeo import gdal
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import numpy as np
import sys
import torch


def read_tif(path):
    gdal.AllRegister()
    filePath = path
    dataset = gdal.Open(filePath)
    Nodata = dataset.GetRasterBand(1).GetNoDataValue()
    adfGeoTransform = dataset.GetGeoTransform()
    nXSize = dataset.RasterXSize
    nYSize = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, nXSize, nYSize)
    index = []
    columns = []
    for j in range(nYSize):
        lat = adfGeoTransform[3] + j * adfGeoTransform[5]
        index.append(lat)
    for i in range(nXSize):
        lon = adfGeoTransform[0] + i * adfGeoTransform[1]
        columns.append(lon)
    data = pd.DataFrame(im_data, index=index, columns=columns)
    return data.values, Nodata, index, columns


def MakeCNNTrainData(PointXY, Feature2d, WindowSize):
    XDistance = abs(PointXY[0, :] - PointXY[0, 0])
    YDistance = abs(PointXY[1, :] - PointXY[1, 0])
    ColumGap = np.min(XDistance[np.where(XDistance != 0)])
    IndexGap = np.min(YDistance[np.where(YDistance != 0)])
    Column = np.arange(PointXY[0, :].min(), PointXY[0, :].max() + ColumGap, ColumGap)
    Index = np.arange(PointXY[1, :].max(), PointXY[1, :].min() - IndexGap, -IndexGap)
    PointXYIndex = np.array([(PointXY[0, :] - Column[0]) / ColumGap, (PointXY[1, :] - Index[0]) / (-IndexGap)])
    CNNPaddingData = []
    NoDataPadding = np.zeros(Feature2d.shape[-1])
    NoDataPadding[:] = -99
    for i in range(len(PointXYIndex[0, :])):
        WindowData = []
        for j in range(-math.floor(WindowSize / 2), math.floor(WindowSize / 2) + 1):
            Data = []
            for k in range(-math.floor(WindowSize / 2), math.floor(WindowSize / 2) + 1):
                XIndex = int(PointXYIndex[0, i] + k)
                YIndex = int(PointXYIndex[1, i] + j)
                if (XIndex < 0 or XIndex >= Feature2d.shape[0]) or (YIndex < 0 or YIndex >= Feature2d.shape[1]):
                    Data.append(NoDataPadding)
                else:
                    Data.append(Feature2d[XIndex, YIndex, :])
            WindowData.append(Data)
        CNNPaddingData.append(WindowData)
    CNNPaddingData = np.array(CNNPaddingData)
    CNNTrainData = []
    for i in range(len(CNNPaddingData)):
        ChanelData = []
        for j in range(len(CNNPaddingData[0, 0, 0])):
            OneChanelData = CNNPaddingData[i, :, :, j]
            if len(np.where(OneChanelData == -99)[0]) != 0:
                AverageData = np.mean(OneChanelData[np.where(OneChanelData != -99)])
                OneChanelData[np.where(OneChanelData == -99)] = AverageData
            ChanelData.append(OneChanelData)
        CNNTrainData.append(ChanelData)
    CNNTrainData = np.array(CNNTrainData)
    return CNNTrainData


def get_tif_path(directory_path):
    # 使用glob模块查找所有.tif文件的路径
    tif_files = glob.glob(os.path.join(directory_path, '*.tif'))
    # 打印出所有找到的.tif文件路径
    return tif_files


Path = ".\\data\\"
AllTifPath = get_tif_path(Path)
OutputPath = ""
WindowSize = 7
tempData, Nodata, Index, Column = read_tif(AllTifPath[0])
TifData = []
for TifPath in AllTifPath:
    tempData, tempNoData, _, _ = read_tif(TifPath)
    TifData.append(tempData)
TifData = np.array(TifData)
HTPointX = []
HTPointY = []
HTPointFeature = []
for i in range(TifData.shape[1]):
    for j in range(TifData.shape[2]):
        HTPointFeature.append(TifData[:, i, j].tolist())
        HTPointX.append(Column[j])
        HTPointY.append(Index[i])
# Produce 2 dim data
HTPointFeature = np.array(HTPointFeature)
DataIndex = np.where(HTPointFeature[:, -1] != tempNoData)[0]
HTPointFeature = HTPointFeature[DataIndex]
for i in range(HTPointFeature.shape[-1]):
    data = HTPointFeature[:, i]
    HTPointFeature[:, i] = (data - np.mean(data)) / np.std(data)
HTPointXY = np.array([HTPointX, HTPointY])
XDistance = abs(HTPointXY[0, :] - HTPointXY[0, 0])
YDistance = abs(HTPointXY[1, :] - HTPointXY[1, 0])
ColumGap = np.min(YDistance[np.where(YDistance != 0)])
IndexGap = np.min(XDistance[np.where(XDistance != 0)])
HTPointXY = HTPointXY[:, DataIndex]
BackGround = np.zeros((len(Column), len(Index), HTPointFeature.shape[-1]))
BackGround[:, :, :] = -99
for i in range(len(HTPointXY[0])):
    BackGround[np.where(Column == HTPointXY[0, i])[0], np.where(Index == HTPointXY[1, i])[0]] = HTPointFeature[i]

SpectrumInput = torch.tensor(HTPointFeature).to(torch.float32)
SpectrumInput = SpectrumInput.reshape(SpectrumInput.size(0), SpectrumInput.size(1), 1)

CNNTrainData = MakeCNNTrainData(HTPointXY, BackGround, WindowSize)
SpatialInput = torch.tensor(CNNTrainData).to(torch.float32)

np.save("PointXY.npy", HTPointXY)
np.save("Feature.npy", HTPointFeature)
np.save("Feature2D.npy", BackGround)
np.save('SpatialInput.npy', np.array(SpatialInput))
np.save('SpectrumInput.npy', np.array(SpectrumInput))

dataset = gdal.Open(AllTifPath[0])
proj = dataset.GetProjection()
np.save(OutputPath + "\\Projection.npy", proj)

Noisy = np.random.rand(HTPointFeature.shape[0], HTPointFeature.shape[1])
HTPointFeature = Noisy
for i in range(HTPointFeature.shape[-1]):
    data = HTPointFeature[:, i]
    HTPointFeature[:, i] = (data - np.mean(data)) / np.std(data)
XDistance = abs(HTPointXY[0, :] - HTPointXY[0, 0])
YDistance = abs(HTPointXY[1, :] - HTPointXY[1, 0])
ColumGap = np.min(YDistance[np.where(YDistance != 0)])
IndexGap = np.min(XDistance[np.where(XDistance != 0)])
Column = np.arange(HTPointXY[0, :].min(), HTPointXY[0, :].max() + ColumGap, ColumGap)
Index = np.arange(HTPointXY[1, :].max(), HTPointXY[1, :].min() - IndexGap, -IndexGap)
BackGround = np.zeros((len(Column), len(Index), HTPointFeature.shape[-1]))
BackGround[:, :, :] = -99
for i in range(len(HTPointXY[0])):
    BackGround[np.where(Column == HTPointXY[0, i])[0], np.where(Index == HTPointXY[1, i])[0]] = HTPointFeature[i]

NoisyCNN = MakeCNNTrainData(HTPointXY, BackGround, WindowSize=7)
np.save(r"NoisyCNN.npy", NoisyCNN)
np.save(r"Noisy.npy", Noisy)
