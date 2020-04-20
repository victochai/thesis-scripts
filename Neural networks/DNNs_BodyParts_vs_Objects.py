#%% LOAD

import os
import scipy.io
import matplotlib.pyplot as plt

#%% LOAD cos_small

# ALEX
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv small, different averaging")
alexnet = scipy.io.loadmat("cos_small")["cos_small"]

# VGG19
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv small, different averaging")
vgg19 = scipy.io.loadmat("cos_small")["cos_small"]

#INCEPTIONV3
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Experimental images\Conv small, different averaging")
inceptionv3 = scipy.io.loadmat("cos_small")["cos_small"]

#ResNet-50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv small, different averaging")
resnet50 = scipy.io.loadmat("cos_small")["cos_small"]

#ResNet-101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv small, different averaging")
resnet101 = scipy.io.loadmat("cos_small")["cos_small"]

#InceptionResNetV2
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Experimental images\Conv small, different averaging")
inception_resnetv2 = scipy.io.loadmat("cos_small")["cos_small"]

#DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Experimental images\Conv small, different averaging")
densenet201 = scipy.io.loadmat("cos_small")["cos_small"]

os.chdir(r"D:\thesis-scripts\Neural networks")

#%% SAVE ALL

allnets = {
        "alexnet" : alexnet,
        "vgg19" : vgg19,
        "inceptionv3" : inceptionv3,
        "resnet50" : resnet50,
        "resnet101" : resnet101,
        "inception_resnetv2" : inception_resnetv2,
        "densenet201" : densenet201
        }

scipy.io.savemat("allnets.mat", allnets)

#%% Normalization

import numpy as np
alexnet_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in alexnet]
vgg19_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in vgg19]
inceptionv3_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in inceptionv3]
resnet50_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in resnet50]
resnet101_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in resnet101]
inception_resnetv2_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in inception_resnetv2]
densenet201_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in densenet201]

allnets_norm = {
        "alexnet_norm" : alexnet_norm,
        "vgg19_norm" : vgg19_norm,
        "inceptionv3_norm" : inceptionv3_norm,
        "resnet50_norm" : resnet50_norm,
        "resnet101_norm" : resnet101_norm,
        "inception_resnetv2_norm" : inception_resnetv2_norm,
        "densenet201_norm" : densenet201_norm
        }    
    
scipy.io.savemat("allnets_norm.mat", allnets_norm)

#%% Body parts vs. Objects

