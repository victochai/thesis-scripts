#%% Impo

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#%% Vizualize all

cos = []
# AlexNet
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Big")
cos.append(scipy.io.loadmat("cos_alex.mat")["cos"][7][0])

# VGG19
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv big")
cos.append(scipy.io.loadmat("co_fc8.mat")["co"])

# ResNet50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv big")
cos.append(scipy.io.loadmat("co_orig_50_fc1000.mat")["co"])

# ResNet101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv big")
cos.append(scipy.io.loadmat("co_orig_101_fc1000.mat")["co"])

## InceptionV3
#os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Conv big")
#cos.append(scipy.io.loadmat("co_conv2d_48_pred.mat")["co"])

## InceptionResNetV2
#os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Conv big\last 2")
#cos.append(scipy.io.loadmat("co_conv2d_176_pred.mat")["co"])

# DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Experimental images\Conv big")
cos.append(scipy.io.loadmat("co_201_fc1000.mat")["co"])

cos_small = []
for co in cos:
    x_ind = -48
    y_ind = -48
    small = np.zeros((7, 7))
    for x in range(0, 7):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 7):
            y_ind += 48
            small[x, y] = np.mean(co[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    cos_small.append(small)    
del co, small, x, x_ind, y, y_ind

#nets = ["AlexNet", "VGG19", "ResNet50", "ResNet101", "InceptionV3", "InceptionResNetV2", "DenseNet-201"]
nets = ["AlexNet", "VGG19", "ResNet50", "ResNet101", "DenseNet-201"]

#%% Visualize all

fig = plt.figure()
fig.suptitle("All DNNs | original images: last layer")
for _ in range(0, len(nets)):
    plt.subplot(2,4,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos[_])
    plt.colorbar()
    plt.clim([-0.4, 1])
    plt.axis("off")
    plt.title(nets[_], fontsize=9)
plt.show() 

os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Experimental images\Conv small, different averaging")
cos_small.append(scipy.io.loadmat("cos_small.mat")["cos_small"][-1])

os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Experimental images\Conv small, different averaging")
cos_small.append(scipy.io.loadmat("cos_small.mat")["cos_small"][-1])

nets = ["AlexNet", "VGG19", "ResNet50", "ResNet101", "DenseNet-201", "InceptionV3", "InceptionResNetV2"]

fig = plt.figure()
fig.suptitle("All DNNs (every condition is averaged) | original images: last layer")
for _ in range(0, len(nets)):
    plt.subplot(2,4,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.clim([0, .75])
    plt.axis("off")
    plt.title(nets[_], fontsize=9)
plt.show() 

#%% Save the results

os.chdir(r"D:\thesis-scripts\Neural networks")
COS_LAST_ALL = {"cos_small" : cos_small}

scipy.io.savemat("COS_LAST_ALL.mat", COS_LAST_ALL)

#%% Visualize all + the brain

os.chdir(r"D:\thesis-scripts\Brain\Brain representations\RDMs and other")
ant = scipy.io.loadmat("ANT.mat")["ant_av"]
ant_left = scipy.io.loadmat("ANT_LEFT.mat")["ant_left_av"]
ant_right = scipy.io.loadmat("ANT_RIGHT.mat")["ant_right_av"]
# new = scipy.io.loadmat("new.mat")["new"]

#%% Visualize brain VS. DNNs

fig = plt.figure()
fig.suptitle("Brain vs. Last layer of Neural Networks (same scale)")

plt.subplot(3,4,1)
plt.imshow(ant)
plt.colorbar()
plt.axis("off")
plt.clim([-0.45, .75])
plt.title("ITG + Anterior IOG", color='red')

plt.subplot(3,4,2)
plt.imshow(ant_left)
plt.colorbar()
plt.axis("off")
plt.clim([-0.45, .75])
plt.title("ITG + Anterior IOG (left)", color='red')

plt.subplot(3,4,3)
plt.imshow(ant_right)
plt.colorbar()
plt.axis("off")
plt.clim([-0.45, .75])
plt.title("ITG + Anterior IOG (right)", color='red')

for _ in range(0, len(nets)):
    plt.subplot(3,4,_+5)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.axis("off")
    plt.clim([-0.45, .75])
    plt.title(nets[_], fontsize=9)
    
plt.show()

#%% ORIGINAL IMAGES

cos = []
# AlexNet
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Original images")
cos.append(scipy.io.loadmat("cos_ALEX")["cos"][0][7])
# VGG19
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Original images")
cos.append(scipy.io.loadmat("cos_VGG")["cos"][0][18])
# ResNet50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Original images\Conv big, original images")
cos.append(scipy.io.loadmat("co_orig_50_fc1000.mat")["co"])
# ResNet101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Original images\Conv big, original")
cos.append(scipy.io.loadmat("co_orig_101_fc1000.mat")["co"])
# DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Original images\Conv big, original images")
cos.append(scipy.io.loadmat("co_orig_201_fc1000.mat")["co"])
# InceptionV3
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Original images")
cos.append(scipy.io.loadmat("cos_INC")["cos"][0][47])
# InceptionResNetV2
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Original images\Conv big, original images")
cos.append(scipy.io.loadmat("co_orig_176_PRED.mat")["co"])

import numpy as np
cos_small = []
for co in cos:
    x_ind = -48
    y_ind = -48
    small = np.zeros((7, 7))
    for x in range(0, 7):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 7):
            y_ind += 48
            small[x, y] = np.mean(co[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    cos_small.append(small)    
del co, small, x, x_ind, y, y_ind

nets = ["AlexNet", "VGG19", "ResNet50", "ResNet101", "DenseNet-201", "InceptionV3", "InceptionResNetV2"]

#%% CREATE RDMs

import os
import scipy.io
import numpy as np

# Big RDM / RESNET-50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv big\l1")
l1 = os.listdir()
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv big\l2")
l2 = os.listdir()
l = l1 + l2
del l1, l2
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv big")
mat = [scipy.io.loadmat(m)["co"] for m in l]
rdms_resnet50 = np.zeros((50, 336, 336))
for _ in range(0, 50):
    rdms_resnet50[_, :, :] = 1 - mat[_]
rdms_resnet50 = {"rdms_resnet50" : rdms_resnet50}
scipy.io.savemat("rdms_resnet50.mat", rdms_resnet50)
    
# Small RDM / RESNET-50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv small\l1")
l1 = os.listdir()
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv small\l2")
l2 = os.listdir()
l = l1 + l2
del l1, l2
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv small")
mat = [scipy.io.loadmat(m)["co_small"] for m in l]
rdms_resnet50_small = np.zeros((50, 7, 7))
for _ in range(0, 50):
    rdms_resnet50_small[_, :, :] = 1 - mat[_]   
rdms_resnet50_small = {"rdms_resnet50_small" : rdms_resnet50_small}
scipy.io.savemat("rdms_resnet50_small.mat", rdms_resnet50_small)
    
# Big RDM / RESNET-101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv big\l1")
l1 = os.listdir()
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv big\l2")
l2 = os.listdir()
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv big\l3")
l3 = os.listdir()
l = l1 + l2 + l3
del l1, l2, l3
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv big")
mat = [scipy.io.loadmat(m)["co"] for m in l]
rdms_resnet101 = np.zeros((101, 336, 336))
for _ in range(0, 101):
    rdms_resnet101[_, :, :] = 1 - mat[_]
rdms_resnet101 = {"rdms_resnet101" : rdms_resnet101}
scipy.io.savemat("rdms_resnet101.mat", rdms_resnet101)

# Big RDM / RESNET-101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv small\l1")
l1 = os.listdir()
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv small\l2")
l2 = os.listdir()
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv small\l3")
l3 = os.listdir()
l = l1 + l2 + l3
del l1, l2, l3
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv small")
mat = [scipy.io.loadmat(m)["co_small"] for m in l]
rdms_resnet101_small = np.zeros((101, 7, 7))
for _ in range(0, 101):
    rdms_resnet101_small[_, :, :] = 1 - mat[_]
rdms_resnet101_small = {"rdms_resnet101_small" : rdms_resnet101_small}
scipy.io.savemat("rdms_resnet101_small.mat", rdms_resnet101_small)    
