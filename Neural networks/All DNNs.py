#%% Impo

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#%% Vizualize all

cos = []
# AlexNet
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Conv Big")
cos.append(a = scipy.io.loadmat("cos_alex.mat")["cos"])
# VGG19
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Conv big")
cos.append(scipy.io.loadmat("fc8.mat")["fc8"])
# ResNet50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Conv big")
cos.append(scipy.io.loadmat("co_conv2d_50_fc1000.mat")["co"])
# ResNet101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Conv big")
cos.append(scipy.io.loadmat("co_conv2d_101_fc1000.mat")["co"])
# InceptionV3
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Conv big")
cos.append(scipy.io.loadmat("co_conv2d_48_pred.mat")["co"])
# InceptionResNetV2
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Conv big\last 2")
cos.append(scipy.io.loadmat("co_conv2d_176_pred.mat")["co"])
# DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Conv big")
cos.append(scipy.io.loadmat("co_conv2d_201_fc1000.mat")["co"])

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

nets = ["AlexNet", "VGG19", "ResNet50", "ResNet101", "InceptionV3", "InceptionResNetV2", "DenseNet-201"]

#%% Visualize all

fig = plt.figure()
fig.suptitle("All DNNs | ORIGINAL IMAGES: last layer")
for _ in range(0, len(nets)):
    plt.subplot(2,4,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos[_])
    plt.colorbar()
    plt.axis("off")
    plt.title(nets[_], fontsize=9)
plt.show() 

fig = plt.figure()
fig.suptitle("All DNNs (every condition is averaged) | ORIGINAL IMAGES: last layer")
for _ in range(0, len(nets)):
    plt.subplot(2,4,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.axis("off")
    plt.title(nets[_], fontsize=9)
plt.show() 

#%% Save the results

os.chdir(r"D:\thesis-scripts\Neural networks")
cos = {"cos" : cos}
cos_small = {"cos_small_original" : cos_small}

scipy.io.savemat("cos.mat", cos)
scipy.io.savemat("cos_small_original.mat", cos_small)

#%% Visualize all + the brain

os.chdir(r"D:\thesis-scripts\Brain\Brain representations")
ant = scipy.io.loadmat("anterior_big_MATRIX.mat")["anterior_big_MATRIX"]
ant_left = scipy.io.loadmat("anterior_left.mat")["anterior_left"]
ant_right = scipy.io.loadmat("anterior_right.mat")["anterior_right"]
new = scipy.io.loadmat("new.mat")["new"]

#%% Visualize brain VS. DNNs

fig = plt.figure()
fig.suptitle("Brain vs. Last layer of Neural Networks")

plt.subplot(3,4,1)
plt.imshow(np.mean(ant, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG", color='red')

plt.subplot(3,4,2)
plt.imshow(np.mean(ant_left, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (left)", color='red')

plt.subplot(3,4,3)
plt.imshow(np.mean(ant_right, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (right)", color='red')

plt.subplot(3,4,4)
plt.imshow(np.mean(new, 2))
plt.colorbar()
plt.axis("off")
plt.title("NEW ROI (??)", color='red')

for _ in range(0, len(nets)):
    plt.subplot(3,4,_+5)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.axis("off")
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
# InceptionV3
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Original images")
cos.append(scipy.io.loadmat("cos_INC")["cos"][0][47])
# InceptionResNetV2
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Original images\Conv big, original images")
cos.append(scipy.io.loadmat("co_orig_176_PRED.mat")["co"])
# DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Original images\Conv big, original images")
cos.append(scipy.io.loadmat("co_orig_201_fc1000.mat")["co"])

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

nets = ["AlexNet", "VGG19", "ResNet50", "ResNet101", "InceptionV3", "InceptionResNetV2", "DenseNet-201"]
