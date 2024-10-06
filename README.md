# Geological-Knowledge-Guided-Dual-Branch-DL-Model-
Geological-Knowledge-Guided Dual-Branch Deep Learning Model for Identification of Geochemical Anomalies Related to Mineralization

# System-specific notes
Matlab R2016a, Tensorflow1.14 version, python3.7
# How to use it?
Firstly, run 1DCNN_DataPreparation.m and miniGCN_DataPreparation.m to generate data  
1DCNN_DataPreparation:Generate training, validation, and testing datastes for the one-dimensional convolutional neural network  
miniGCN_DataPreparation:Generate training, validation, and testing datastes for the one-dimensional mini-batch graph convolutional network  
Then run 1DCNN+miniGCN_loss.py
# REFERENCE
## CNN+miniGCN
[Hong, D., Gao, L., Yao, J., Zhang, B., Plaza, A., Chanussot, J., 2021. Graph Convolutional Networks for Hyperspectral Image Classification. IEEE Transactions on Geoscience and Remote Sensing 59(7), 5966-5978.](https://ieeexplore.ieee.org/document/9170817)
If you have any question, you can contact me via email xy1021989464@163.com
