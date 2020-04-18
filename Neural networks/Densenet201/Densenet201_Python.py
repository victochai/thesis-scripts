#%% LOAD

import os
import scipy.io
import matplotlib.pyplot as plt

#%% CONV BIG

os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Conv big")
cos = []
l = os.listdir()
for _ in l:
    cos.append(scipy.io.loadmat(_)["co"]) 
    
fig = plt.figure()
fig.suptitle("DenseNet-201\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(l)):
    plt.subplot(13,16,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos[_])
#    plt.colorbar()
    plt.axis("off")
#    plt.title(str(_+1), fontsize=5)
plt.show() 

#%% CONV SMALL

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

fig = plt.figure()
fig.suptitle("DenseNet-201\nEvery condition is averaged\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(l)):
    plt.subplot(13,16,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
#    plt.colorbar()
    plt.axis("off")
#    plt.title(str(_+1), fontsize=9)
plt.show() 

# Saving stuff
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Conv small, different averaging")
cos_small = {"cos_small" : cos_small}
scipy.io.savemat("cos_small.mat", cos_small)
