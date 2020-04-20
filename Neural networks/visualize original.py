#%% Original images visualization

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# Load DNN
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Original images")
alex = scipy.io.loadmat("cos_INC.mat")["cos"][0]
cos = []
for _ in alex:
    cos.append(_)
del alex

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

#%% Visualization

# Big
fig = plt.figure()
fig.suptitle("RESNET-101\nORIGINAL IMAGES\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(cos)):
    plt.subplot(8,13,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos[_])
#    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=5)
plt.show() 

# Small
fig = plt.figure()
fig.suptitle("RESNET-101\nORIGINAL IMAGES\nEvery condition is averaged\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(cos_small)):
    plt.subplot(8,13,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
#    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=5)
plt.show() 

#%% SAVE

cos_small_original = {"cos_small" : cos_small}
scipy.io.savemat("cos_small_original.mat", cos_small_original)
