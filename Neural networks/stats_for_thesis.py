#%% Libraries

import numpy as np
import os
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% Visuazations|Transformations I need

"""
1. Fisher transform all the correlations
2. Plot correlations: One scale... ALL TOGETHER.
3. Visualization: first layers + calc. cortex
4. Visualization: last layers + ROI.
5. MDS
6. Visualize big co's * optional

"""

#%% Loading DNNs (first + last layers)

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

#%% Loading brain representations

os.chdir(r"D:\thesis-scripts\Brain\Brain representations\RDMs and other")
ant = loadmat("ANT.mat")["ant_av"]
calc = loadmat("CALC.mat")["calc_av"]

#%% Deleting chair condition

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

#%% Choosing style

print(mpl.style.available)
mpl.style.use("fivethirtyeight")
mpl.style.use("ggplot")
mpl.style.use("seaborn-dark")

mpl.rcdefaults()
 
#%% Visualization co's big | last layers

min_ = []
max_ = []
for _ in cos_last:
    min_.append(np.min(_))
    max_.append(np.max(_))

fig, axs = plt.subplots(nrows=2,ncols=5, sharex=True, sharey=True)
labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=range(0,288,48) 
ticks_minor=range(24,288+24,48)

im=[]
for _ in range(0, 5):
    im.append(axs[_].imshow(cos_last[_], vmin=min(min_), vmax=max(max_)))
    axs[_].set_title(nets[_])
    axs[_].grid(False)
    
    # X ticks
    axs[_].set_xticks(ticks=ticks_major)
    axs[_].set_xticklabels(labels="")
    axs[_].set_xticks(ticks=ticks_minor, minor=True)
    axs[_].set_xticklabels(labels=labels, minor=True, rotation=90)
    axs[_].tick_params(axis="x", which="major", width=1)
    axs[_].tick_params(axis="x", which="minor", color ="w")
    
    if _ == 0:
        # Y ticks
        axs[_].set_yticks(ticks=ticks_major)
        axs[_].set_yticklabels(labels="")
        axs[_].set_yticks(ticks=ticks_minor, minor=True)
        axs[_].set_yticklabels(labels=labels, minor=True)
        axs[_].tick_params(axis="y", which="major", width=1)
        axs[_].tick_params(axis="y", which="minor", color ="w")

    if _ > 0:
        # Y ticks
        axs[_].tick_params(axis="y", which="major", width=1)
        axs[_].tick_params(axis="y", which="minor", color ="w")
    
#    axs[_].grid("On")

#plt.grid("Off")    
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5]) # left bottom width height
fig.colorbar(im[_], cax=cbar_ax)

#fig.tight_layout()
plt.show()

#%% Visualization co's big | first layers

min_ = []
max_ = []
for _ in cos_first:
    min_.append(np.min(_))
    max_.append(np.max(_))

fig, axs = plt.subplots(nrows=1,ncols=5, sharex=True, sharey=True)
labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=range(0,288,48)
ticks_minor=range(24,288+24,48)

im=[]
for _ in range(0, 5):
    im.append(axs[_].imshow(cos_first[_], vmin=min(min_), vmax=max(max_)))
    axs[_].set_title(nets[_])
    axs[_].grid(False)
    
    # X ticks
    axs[_].set_xticks(ticks=ticks_major)
    axs[_].set_xticklabels(labels="")
    axs[_].set_xticks(ticks=ticks_minor, minor=True)
    axs[_].set_xticklabels(labels=labels, minor=True, rotation=90)
    axs[_].tick_params(axis="x", which="major", width=1)
    axs[_].tick_params(axis="x", which="minor", color ="w")
    
    if _ == 0:
        # Y ticks
        axs[_].set_yticks(ticks=ticks_major)
        axs[_].set_yticklabels(labels="")
        axs[_].set_yticks(ticks=ticks_minor, minor=True)
        axs[_].set_yticklabels(labels=labels, minor=True)
        axs[_].tick_params(axis="y", which="major", width=1)
        axs[_].tick_params(axis="y", which="minor", color ="w")

    if _ > 0:
        # Y ticks
        axs[_].tick_params(axis="y", which="major", width=1)
        axs[_].tick_params(axis="y", which="minor", color ="w")
    
#    axs[_].grid("On")

#plt.grid("Off")    
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5]) # left bottom width height
fig.colorbar(im[_], cax=cbar_ax)

#fig.tight_layout()
plt.show()

#%% Visualize co_small + the brain | FIRST LAYERS

# Min | Max of DNNs
min_ = []
max_ = []
for _ in cos_small_first:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig, axs = plt.subplots(nrows=3,ncols=3, sharex=True, sharey=True)

labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=[0.5,1.5,2.5,3.5,4.5,5.5]
ticks_minor=range(0,6,1)
im = []

# Brain
im.append(axs[0,0].imshow(calc, cmap="Greens"))
axs[0,0].set_title("Calcarine \ncortex", size=12)
#axs[0,0].grid(False)
axs[0,0].set_xticks(ticks=ticks_major)
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].set_yticks(ticks=ticks_major)
axs[0,0].set_yticklabels(labels="")
axs[0,0].set_yticks(ticks=ticks_minor, minor=True)
axs[0,0].set_yticklabels(labels=labels, minor=True)
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")

axs[0,1].axis("Off")
axs[0,2].axis("Off")

for _ in range(0,3):
    im.append(axs[1,_].imshow(cos_small_first[_], cmap="Greens", vmin=np.min(min_), vmax=np.max(max_)))
    axs[1,_].set_title(nets[_],size=12)
#    axs[1,_].grid(False)  
    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].set_yticks(ticks=ticks_major)
    axs[1,_].set_yticklabels(labels="")
    axs[1,_].set_yticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_yticklabels(labels=labels, minor=True)
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")


im.append(axs[2,0].imshow(cos_small_first[3], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,0].set_title(nets[3],size=12)
#axs[2,0].grid(False)  
axs[2,0].set_xticks(ticks=ticks_major)
axs[2,0].set_xticklabels(labels="")
axs[2,0].set_xticks(ticks=ticks_minor, minor=True)
axs[2,0].set_xticklabels(labels=labels, minor=True, rotation=90)
axs[2,0].tick_params(axis="x", which="major", color ="w")
axs[2,0].tick_params(axis="x", which="minor", color ="w")
axs[2,0].set_yticks(ticks=ticks_major)
axs[2,0].set_yticklabels(labels="")
axs[2,0].set_yticks(ticks=ticks_minor, minor=True)
axs[2,0].set_yticklabels(labels=labels, minor=True)
axs[2,0].tick_params(axis="y", which="major", color ="w")
axs[2,0].tick_params(axis="y", which="minor", color ="w")

im.append(axs[2,1].imshow(cos_small_first[4], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,1].set_title(nets[4],size=12)
#axs[2,1].grid(False)  
axs[2,1].set_xticks(ticks=ticks_major)
axs[2,1].set_xticklabels(labels="")
axs[2,1].set_xticks(ticks=ticks_minor, minor=True)
axs[2,1].set_xticklabels(labels=labels, minor=True, rotation=90)
axs[2,1].tick_params(axis="x", which="major", color ="w")
axs[2,1].tick_params(axis="x", which="minor", color ="w")
axs[2,1].tick_params(axis="y", which="major", color ="w")
axs[2,1].tick_params(axis="y", which="minor", color ="w")

axs[2,2].axis("Off")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.488, 0.65, 0.01, 0.195]) # left bottom width height
v = np.linspace(np.min(calc), np.max(calc), 5, endpoint=True)
cbar1 = fig.colorbar(im[0], cax=cbar_ax, ticks=v)
cbar1.ax.tick_params(labelsize=8) 

fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.488, 0.2, 0.01, 0.325]) # left bottom width height
cbar_ax = fig.add_axes([0.488, 0.18, 0.01, 0.382]) # left bottom width height
v = np.linspace(np.min(min_), np.max(max_), 8, endpoint=True)
cbar2 = fig.colorbar(im[_], cax=cbar_ax, ticks=v)
cbar2.ax.tick_params(labelsize=8) 

fig.subplots_adjust(hspace=0.270, wspace=0.0, top=0.840, bottom=0.160, left=0.160, right=0.475)
#fig.tight_layout()
plt.show()

#%% ############### Visualize co_small + the brain | LAST LAYERS ##############
###############################################################################

# Min | Max of DNNs
min_ = []
max_ = []
for _ in cos_small_last:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig, axs = plt.subplots(nrows=3,ncols=3, sharex=True, sharey=True)

labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=[0.5,1.5,2.5,3.5,4.5,5.5]
ticks_minor=range(0,6,1)
im = []

# Brain
im.append(axs[0,0].imshow(ant, cmap="Greens"))
axs[0,0].set_title("ITG + \nPosterior IOG", size=12)
#axs[0,0].grid(False)
axs[0,0].set_xticks(ticks=ticks_major)
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].set_yticks(ticks=ticks_major)
axs[0,0].set_yticklabels(labels="")
axs[0,0].set_yticks(ticks=ticks_minor, minor=True)
axs[0,0].set_yticklabels(labels=labels, minor=True, size=10, color="dimgray")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")

axs[0,1].axis("Off")
axs[0,2].axis("Off")

for _ in range(0,3):
    im.append(axs[1,_].imshow(cos_small_last[_], cmap="Greens", vmin=np.min(min_), vmax=np.max(max_)))
    axs[1,_].set_title(nets[_], size=12)
#    axs[1,_].grid(False)  
    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].set_yticks(ticks=ticks_major)
    axs[1,_].set_yticklabels(labels="")
    axs[1,_].set_yticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_yticklabels(labels=labels, minor=True, size=10, color="dimgray")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")


im.append(axs[2,0].imshow(cos_small_last[3], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,0].set_title(nets[3], size=12)
#axs[2,0].grid(False)  
axs[2,0].set_xticks(ticks=ticks_major)
axs[2,0].set_xticklabels(labels="")
axs[2,0].set_xticks(ticks=ticks_minor, minor=True)
axs[2,0].set_xticklabels(labels=labels, minor=True, rotation=90, size=10, color="dimgray")
axs[2,0].tick_params(axis="x", which="major", color ="w")
axs[2,0].tick_params(axis="x", which="minor", color ="w")
axs[2,0].set_yticks(ticks=ticks_major)
axs[2,0].set_yticklabels(labels="")
axs[2,0].set_yticks(ticks=ticks_minor, minor=True)
axs[2,0].set_yticklabels(labels=labels, minor=True, size=10, color="dimgray")
axs[2,0].tick_params(axis="y", which="major", color ="w")
axs[2,0].tick_params(axis="y", which="minor", color ="w")

im.append(axs[2,1].imshow(cos_small_last[4], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,1].set_title(nets[4], size=12)
#axs[2,1].grid(False)  
axs[2,1].set_xticks(ticks=ticks_major)
axs[2,1].set_xticklabels(labels="")
axs[2,1].set_xticks(ticks=ticks_minor, minor=True)
axs[2,1].set_xticklabels(labels=labels, minor=True, rotation=90, size=10, color="dimgray")
axs[2,1].tick_params(axis="x", which="major", color ="w")
axs[2,1].tick_params(axis="x", which="minor", color ="w")
axs[2,1].tick_params(axis="y", which="major", color ="w")
axs[2,1].tick_params(axis="y", which="minor", color ="w")

axs[2,2].axis("Off")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.488, 0.65, 0.01, 0.195]) # left bottom width height
v = np.linspace(np.min(ant), np.max(ant), 5, endpoint=True)
cbar1 = fig.colorbar(im[0], cax=cbar_ax, ticks=v)
cbar1.ax.tick_params(labelsize=8) 

fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.488, 0.2, 0.01, 0.325]) # left bottom width height
cbar_ax = fig.add_axes([0.488, 0.18, 0.01, 0.382]) # left bottom width height
v = np.linspace(np.min(min_), np.max(max_), 8, endpoint=True)
cbar2 = fig.colorbar(im[_], cax=cbar_ax, ticks=v)
cbar2.ax.tick_params(labelsize=8) 

fig.subplots_adjust(hspace=0.270, wspace=0.0, top=0.840, bottom=0.160, left=0.160, right=0.475)
#fig.tight_layout()
plt.show()

#%% Plot correlations... Body parts VS. Objects









