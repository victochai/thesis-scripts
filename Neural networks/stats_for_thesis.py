#%% 1.) Libraries

import numpy as np
import os
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

#%% 2.) Loading DNNs (first + last layers)

# 1. Alexnet
# 2. VGG-19
# 3. Inception-V3
# 4. Resnet-50
# 5. Densenet-201


cos_last = []
cos_first = []

os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Big")
cos_last.append(loadmat("cos_alex.mat")["cos"][7][0])
cos_first.append(loadmat("cos_alex.mat")["cos"][0][0])

os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv big")
cos_last.append(loadmat("co_fc8.mat")["co"])
cos_first.append(loadmat("co_conv1_1.mat")["co"])

os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv big")
cos_last.append(loadmat("co_orig_50_fc1000.mat")["co"])
cos_first.append(loadmat("co_orig_1_conv1.mat")["co"])

os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Experimental images\First and last layers")
cos_last.append(loadmat("co_predictions.mat")["co"])
cos_first.append(loadmat("co_conv2d_1.mat")["co"])

os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Experimental images\Conv big")
cos_last.append(loadmat("co_201_fc1000.mat")["co"])
cos_first.append(loadmat("co_1_.mat")["co"])

nets = ["AlexNet", "VGG-19", "ResNet-50", "Inception-V3", "DenseNet-201"]

#%% 3.) Loading brain representations

os.chdir(r"D:\thesis-scripts\Brain\Brain representations\RDMs and other")
ant = loadmat("ANT.mat")["ant_av"]
calc = loadmat("CALC.mat")["calc_av"]

#%% 4.) Deleting chair condition

cos_last_ = []
for _ in cos_last:
    cos_last_.append(_[0:48*6, 0:48*6])
cos_last = cos_last_
del cos_last_

cos_first_ = []
for _ in cos_first:
    cos_first_.append(_[0:48*6, 0:48*6])
cos_first = cos_first_
del cos_first_

ant = ant[0:6, 0:6]
calc = calc[0:6, 0:6]

cos_small_last = []
for co in cos_last:
    x_ind = -48
    y_ind = -48
    small = np.zeros((6, 6))
    for x in range(0, 6):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 6):
            y_ind += 48
            small[x, y] = np.mean(co[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    cos_small_last.append(small)    
del co, small, x, x_ind, y, y_ind 

cos_small_first = []
for co in cos_first:
    x_ind = -48
    y_ind = -48
    small = np.zeros((6, 6))
    for x in range(0, 6):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 6):
            y_ind += 48
            small[x, y] = np.mean(co[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    cos_small_first.append(small)    
del co, small, x, x_ind, y, y_ind 

#%% 5.) Visualization co's big | last layers

min_ = []
max_ = []
for _ in cos_last:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig = plt.figure()
# fig.subtitle()
for _ in range(0, 5):
    plt.subplot(2,3,_+1)
    plt.imshow(cos_last[_])
    plt.title(nets[_])
    plt.axis("Off")
    plt.clim([np.min(min_), np.max(max_)])
    plt.colorbar()
plt.show()

del min_, max_

#%% 6.) Visualization co's big | first layers

min_ = []
max_ = []
for _ in cos_small_first:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig = plt.figure()

plt.subplot(2,5,1)
plt.imshow(calc)
plt.title("Calcarine cortex")
plt.axis("Off")
#plt.clim([np.min(min_), np.max(max_)])
plt.colorbar(fraction=0.046, pad=0.04)

for _ in range(0, 5):
    plt.subplot(2,5,_+6)
    plt.imshow(cos_small_first[_])
    plt.title(nets[_])
    plt.axis("Off")
    plt.clim([np.min(min_), np.max(max_)])
    plt.colorbar(fraction=0.046, pad=0.04)
    
plt.show()
del min_, max_

#%% 7.) Visualizing co's small + the brain

min_ = []
max_ = []
for _ in cos_first:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig = plt.figure()
# fig.subtitle()
for _ in range(0, 5):
    plt.subplot(2,3,_+1)
    plt.imshow(cos_first[_])
    plt.title(nets[_])
    plt.axis("Off")
    plt.clim([np.min(min_), np.max(max_)])
    plt.colorbar()
plt.show()

del min_, max_


#%% Visualization | Last layer

fig = plt.figure()
# fig.subtitle()

plt.subplot(2,5,1)
plt.imshow(ant)
plt.colorbar()
plt.axis("off")
#plt.clim([-0.45, .75])
plt.title("ITG + Anterior IOG")

for _ in range(1, 5):
    plt.subplot(2,5,_+5)
    plt.imshow(cos_small_last[_])
    plt.title(nets[_])
    plt.axis("Off")
    plt.colorbar()
    
plt.show()

#%% Visualization | First layer

fig = plt.figure()
# fig.subtitle()

plt.subplot(2,5,1)
plt.imshow(calc)
plt.colorbar()
plt.axis("off")
#plt.clim([-0.45, .75])
plt.title("Calcarine cortex")

for _ in range(1, 5):
    plt.subplot(2,5,_+5)
    plt.imshow(cos_small_first[_])
    plt.title(nets[_])
    plt.axis("Off")
    plt.colorbar()
    
plt.show()

#%% Visualization co's big

cos_last_ = []
for _ in cos_last:
    cos_last_.append(_[0:288, 0:288])
cos_last = cos_last_
del cos_last_

min_ = []
max_ = []
for _ in cos_last:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig = plt.figure()
# fig.subtitle()
for _ in range(0, 5):
    plt.subplot(2,3,_+1)
    plt.imshow(cos_last[_])
    plt.title(nets[_])
    plt.axis("Off")
    plt.clim([np.min(min_), np.max(max_)])
    plt.colorbar()
plt.show()

