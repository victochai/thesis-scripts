#%% Libraries

import numpy as np
import os
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

size_ticks=9
size_title=9
size_cbar=7.5
 
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
axs[0,0].grid(False)
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
    axs[1,_].grid(False)  
    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].set_yticks(ticks=ticks_major)
    axs[1,_].set_yticklabels(labels="")
    axs[1,_].set_yticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_yticklabels(labels=labels, minor=True,size=10, color="dimgray")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")


im.append(axs[2,0].imshow(cos_small_first[3], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,0].set_title(nets[3],size=12)
axs[2,0].grid(False)  
axs[2,0].set_xticks(ticks=ticks_major)
axs[2,0].set_xticklabels(labels="")
axs[2,0].set_xticks(ticks=ticks_minor, minor=True)
axs[2,0].set_xticklabels(labels=labels, minor=True, rotation=90,size=10, color="dimgray")
axs[2,0].tick_params(axis="x", which="major", color ="w")
axs[2,0].tick_params(axis="x", which="minor", color ="w")
axs[2,0].set_yticks(ticks=ticks_major)
axs[2,0].set_yticklabels(labels="")
axs[2,0].set_yticks(ticks=ticks_minor, minor=True)
axs[2,0].set_yticklabels(labels=labels, minor=True,size=10, color="dimgray")
axs[2,0].tick_params(axis="y", which="major", color ="w")
axs[2,0].tick_params(axis="y", which="minor", color ="w")

im.append(axs[2,1].imshow(cos_small_first[4], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,1].set_title(nets[4],size=12)
axs[2,1].grid(False)  
axs[2,1].set_xticks(ticks=ticks_major)
axs[2,1].set_xticklabels(labels="")
axs[2,1].set_xticks(ticks=ticks_minor, minor=True)
axs[2,1].set_xticklabels(labels=labels, minor=True, rotation=90,size=10, color="dimgray")
axs[2,1].tick_params(axis="x", which="major", color ="w")
axs[2,1].tick_params(axis="x", which="minor", color ="w")
axs[2,1].tick_params(axis="y", which="major", color ="w")
axs[2,1].tick_params(axis="y", which="minor", color ="w")

axs[2,2].axis("Off")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.2690, 0.6475, 0.01, 0.193]) # left bottom width height
#v = np.linspace(np.min(calc), np.max(calc), 5, endpoint=True)
#cbar1 = fig.colorbar(im[0], cax=cbar_ax, ticks=v)
cbar1 = fig.colorbar(im[0], cax=cbar_ax, format="%.1f", 
                     boundaries=np.linspace(-.2, .4, 1000, endpoint=True), 
                     ticks=np.linspace(-.2, .4, 5, endpoint=True))
cbar1.ax.tick_params(labelsize=8) 

fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.488, 0.2, 0.01, 0.325]) # left bottom width height
cbar_ax = fig.add_axes([0.3743, 0.159, 0.01, 0.1925]) # left bottom width height
#v = np.linspace(np.min(min_), np.max(max_), 8, endpoint=True)
#cbar2 = fig.colorbar(im[_], cax=cbar_ax, ticks=v)
cbar2 = fig.colorbar(im[_], cax=cbar_ax, 
                     boundaries=np.linspace(.6, 1, 1000, endpoint=True), 
                     ticks=np.linspace(.6, 1, 5, endpoint=True),
                     format="%.1f")

cbar2.ax.tick_params(labelsize=8) 

fig.text(.12, .27, "DNN data (first layer)", fontsize=15, rotation=90)
fig.text(.12, .605, "b", fontsize=16)
fig.text(.12, .69, "Brain data", fontsize=15, rotation=90)
fig.text(.12, .86, "a", fontsize=16)

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
axs[0,0].grid(False)
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
    axs[1,_].grid(False)  
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
axs[2,0].grid(False)  
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
axs[2,1].grid(False)  
axs[2,1].set_xticks(ticks=ticks_major)
axs[2,1].set_xticklabels(labels="")
axs[2,1].set_xticks(ticks=ticks_minor, minor=True)
axs[2,1].set_xticklabels(labels=labels, minor=True, rotation=90, size=10, color="dimgray")
axs[2,1].tick_params(axis="x", which="major", color ="w")
axs[2,1].tick_params(axis="x", which="minor", color ="w")
axs[2,1].tick_params(axis="y", which="major", color ="w")
axs[2,1].tick_params(axis="y", which="minor", color ="w")

axs[2,2].

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.2690, 0.6475, 0.01, 0.193]) # left bottom width height
#v = np.linspace(np.min(ant), np.max(ant), 5, endpoint=True)
#v = np.linspace(-.5, .8, 5, endpoint=True)
cbar1 = fig.colorbar(im[0], cax=cbar_ax, format="%.1f",
                     boundaries=np.linspace(-.5, .8, 1000, endpoint=True),
                     ticks=np.linspace(-.5, .8, 5, endpoint=True))
cbar1.ax.tick_params(labelsize=8) 

fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.488, 0.2, 0.01, 0.325]) # left bottom width height
cbar_ax = fig.add_axes([0.3743, 0.159, 0.01, 0.1925]) # left bottom width height
#v = np.linspace(np.min(min_), np.max(max_), 8, endpoint=True)
#cbar2 = fig.colorbar(im[_], cax=cbar_ax, ticks=v)
cbar2 = fig.colorbar(im[_], cax=cbar_ax, 
                     boundaries=np.linspace(0, .8, 1000, endpoint=True), 
                     ticks=np.linspace(0, .8, 5, endpoint=True),
                     format="%.1f")
cbar2.ax.tick_params(labelsize=8) 

fig.text(.12, .27, "DNN data (last layer)", fontsize=15, rotation=90)
fig.text(.12, .605, "b", fontsize=16)
fig.text(.12, .69, "Brain data", fontsize=15, rotation=90)
fig.text(.12, .86, "a", fontsize=16)

fig.subplots_adjust(hspace=0.270, wspace=0.0, top=0.840, bottom=0.160, left=0.160, right=0.475)
#fig.tight_layout()
plt.show()

#%% DIFFERENT SCALING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#######%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#######%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#######%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#######%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% lAST LAYER

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
axs[0,0].set_title("ITG + \nPosterior IOG", size=9,fontweight="bold",color="dimgray")
axs[0,0].grid(False)
axs[0,0].set_xticks(ticks=ticks_major)
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].set_yticks(ticks=ticks_major)
axs[0,0].set_yticklabels(labels="")
axs[0,0].set_yticks(ticks=ticks_minor, minor=True)
axs[0,0].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")

axs[0,1].axis("Off")
axs[0,2].axis("Off")

for _ in range(0,2):
    im.append(axs[1,_].imshow(cos_small_last[_], cmap="Greens", vmin=np.min(min_), vmax=np.max(max_)))
    axs[1,_].set_title(nets[_], size=9,fontweight="bold",color="dimgray")
    axs[1,_].grid(False)  
    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].set_yticks(ticks=ticks_major)
    axs[1,_].set_yticklabels(labels="")
    axs[1,_].set_yticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")

axs[1,2].axis("Off")

im.append(axs[2,0].imshow(cos_small_last[2], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,0].set_title(nets[2], size=9, fontweight="bold",color="dimgray")
axs[2,0].grid(False)  
axs[2,0].set_xticks(ticks=ticks_major)
axs[2,0].set_xticklabels(labels="")
axs[2,0].set_xticks(ticks=ticks_minor, minor=True)
axs[2,0].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")
axs[2,0].tick_params(axis="x", which="major", color ="w")
axs[2,0].tick_params(axis="x", which="minor", color ="w")
axs[2,0].set_yticks(ticks=ticks_major)
axs[2,0].set_yticklabels(labels="")
axs[2,0].set_yticks(ticks=ticks_minor, minor=True)
axs[2,0].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")
axs[2,0].tick_params(axis="y", which="major", color ="w")
axs[2,0].tick_params(axis="y", which="minor", color ="w")

im.append(axs[2,1].imshow(cos_small_last[3], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,1].set_title(nets[3], size=9,fontweight="bold",color="dimgray")
axs[2,1].grid(False)  
axs[2,1].set_xticks(ticks=ticks_major)
axs[2,1].set_xticklabels(labels="")
axs[2,1].set_xticks(ticks=ticks_minor, minor=True)
axs[2,1].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")
axs[2,1].tick_params(axis="x", which="major", color ="w")
axs[2,1].tick_params(axis="x", which="minor", color ="w")
axs[2,1].set_yticks(ticks=ticks_major)
axs[2,1].set_yticklabels(labels="")
axs[2,1].set_yticks(ticks=ticks_minor, minor=True)
axs[2,1].set_yticklabels(labels=labels, minor=True, size=9, color="dimgray")
axs[2,1].tick_params(axis="y", which="major", color ="w")
axs[2,1].tick_params(axis="y", which="minor", color ="w")

im.append(axs[2,2].imshow(cos_small_last[4], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,2].set_title(nets[4], size=9,fontweight="bold",color="dimgray")
axs[2,2].grid(False)  
axs[2,2].set_xticks(ticks=ticks_major)
axs[2,2].set_xticklabels(labels="")
axs[2,2].set_xticks(ticks=ticks_minor, minor=True)
axs[2,2].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks,color="dimgray")
axs[2,2].tick_params(axis="x", which="major", color ="w")
axs[2,2].tick_params(axis="x", which="minor", color ="w")
axs[2,2].tick_params(axis="y", which="major", color ="w")
axs[2,2].tick_params(axis="y", which="minor", color ="w")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.425, 0.672, 0.01, 0.212]) # left bottom width height
#cbar_ax = fig.add_axes([0.422, 0.67, 0.01, 0.212]) # left bottom width height
#v = np.linspace(np.min(ant), np.max(ant), 5, endpoint=True)
#v = np.linspace(-.5, .8, 5, endpoint=True)
cbar1 = fig.colorbar(im[0], cax=cbar_ax, format="%.1f",
                     boundaries=np.linspace(-.5, .8, 1000, endpoint=True),
                     ticks=np.linspace(-.5, .8, 5, endpoint=True))
cbar1.ax.tick_params(labelsize=size_cbar) 

fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.488, 0.2, 0.01, 0.325]) # left bottom width height
#cbar_ax = fig.add_axes([0.6, 0.124, 0.01, 0.212]) # left bottom width height
#cbar_ax = fig.add_axes([0.7772, 0.125, 0.01, 0.212]) # left bottom width height
cbar_ax = fig.add_axes([0.7772, 0.125, 0.01, 0.4835]) # left bottom width height
#v = np.linspace(np.min(min_), np.max(max_), 8, endpoint=True)
#cbar2 = fig.colorbar(im[_], cax=cbar_ax, ticks=v)
cbar2 = fig.colorbar(im[_], cax=cbar_ax, 
                     boundaries=np.linspace(0, .8, 1000, endpoint=True), 
                     ticks=np.linspace(0, .8, 7, endpoint=True),
                     format="%.1f")
cbar2.ax.tick_params(labelsize=size_cbar) 

fig.text(.130, .20, "DNN data (last layer)", fontsize=10, rotation=90,fontweight="bold",color="dimgray")
fig.text(.130, .625, "b.", fontsize=10,fontweight="bold",color="dimgray")
fig.text(.130, .715, "Brain data", fontsize=10, rotation=90, fontweight="bold",color="dimgray")
fig.text(.130, .925, "a.", fontsize=10,fontweight="bold",color="dimgray")

fig.subplots_adjust(top=0.885,
bottom=0.125,
left=0.24,
right=0.77,
hspace=0.28,
wspace=0.0)
#fig.tight_layout()
plt.show()

#%% FIRST LAYER

# Min | Max of DNNs
min_ = []
max_ = []
for _ in cos_small_first:
    min_.append(np.min(_))
    max_.append(np.max(_))
    
fig, axs = plt.subplots(nrows=3,ncols=3, sharex=True, sharey=True)

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=[0.5,1.5,2.5,3.5,4.5,5.5]
ticks_minor=range(0,6,1)
im = []

# Brain
im.append(axs[0,0].imshow(calc, cmap="Greens"))
axs[0,0].set_title("Calcarine \nCortex", size=9,fontweight="bold",color="dimgray")
axs[0,0].grid(False)
axs[0,0].set_xticks(ticks=ticks_major)
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].set_yticks(ticks=ticks_major)
axs[0,0].set_yticklabels(labels="")
axs[0,0].set_yticks(ticks=ticks_minor, minor=True)
axs[0,0].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")

axs[0,1].axis("Off")
axs[0,2].axis("Off")

for _ in range(0,2):
    im.append(axs[1,_].imshow(cos_small_first[_], cmap="Greens", vmin=np.min(min_), vmax=np.max(max_)))
    axs[1,_].set_title(nets[_], size=9,fontweight="bold",color="dimgray")
    axs[1,_].grid(False)  
    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].set_yticks(ticks=ticks_major)
    axs[1,_].set_yticklabels(labels="")
    axs[1,_].set_yticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")

axs[1,2].axis("Off")

im.append(axs[2,0].imshow(cos_small_first[2], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,0].set_title(nets[2], size=9,fontweight="bold",color="dimgray")
axs[2,0].grid(False)  
axs[2,0].set_xticks(ticks=ticks_major)
axs[2,0].set_xticklabels(labels="")
axs[2,0].set_xticks(ticks=ticks_minor, minor=True)
axs[2,0].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")
axs[2,0].tick_params(axis="x", which="major", color ="w")
axs[2,0].tick_params(axis="x", which="minor", color ="w")
axs[2,0].set_yticks(ticks=ticks_major)
axs[2,0].set_yticklabels(labels="")
axs[2,0].set_yticks(ticks=ticks_minor, minor=True)
axs[2,0].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")
axs[2,0].tick_params(axis="y", which="major", color ="w")
axs[2,0].tick_params(axis="y", which="minor", color ="w")


im.append(axs[2,1].imshow(cos_small_first[3], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,1].set_title(nets[3], size=9,fontweight="bold",color="dimgray")
axs[2,1].grid(False)  
axs[2,1].set_xticks(ticks=ticks_major)
axs[2,1].set_xticklabels(labels="")
axs[2,1].set_xticks(ticks=ticks_minor, minor=True)
axs[2,1].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")
axs[2,1].tick_params(axis="x", which="major", color ="w")
axs[2,1].tick_params(axis="x", which="minor", color ="w")
axs[2,1].tick_params(axis="y", which="major", color ="w")
axs[2,1].tick_params(axis="y", which="minor", color ="w")

im.append(axs[2,2].imshow(cos_small_first[4], cmap="Greens", vmin=min(min_), vmax=max(max_)))
axs[2,2].set_title(nets[4], size=9,fontweight="bold",color="dimgray")
axs[2,2].grid(False)  
axs[2,2].set_xticks(ticks=ticks_major)
axs[2,2].set_xticklabels(labels="")
axs[2,2].set_xticks(ticks=ticks_minor, minor=True)
axs[2,2].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")
axs[2,2].tick_params(axis="x", which="major", color ="w")
axs[2,2].tick_params(axis="x", which="minor", color ="w")
axs[2,2].tick_params(axis="y", which="major", color ="w")
axs[2,2].tick_params(axis="y", which="minor", color ="w")

a = axs[0,0]
b = axs[-1,-1]
print(a.dataLim)
print(b.dataLim)
print(a.viewLim)
print(b.viewLim)
print(fig.patch)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.425, 0.672, 0.01, 0.212]) # left bottom width height
#v = np.linspace(np.min(ant), np.max(ant), 5, endpoint=True)
#v = np.linspace(-.5, .8, 5, endpoint=True)
cbar1 = fig.colorbar(im[0], cax=cbar_ax, format="%.1f", 
                     boundaries=np.linspace(-.2, .4, 1000, endpoint=True), 
                     ticks=np.linspace(-.2, .4, 5, endpoint=True))
cbar1.ax.tick_params(labelsize=size_cbar) 

fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.488, 0.2, 0.01, 0.325]) # left bottom width height
#cbar_ax = fig.add_axes([0.7772, 0.125, 0.01, 0.212]) # left bottom width height
cbar_ax = fig.add_axes([0.7772, 0.125, 0.01, 0.4835]) # left bottom width height
#cbar_ax = fig.add_axes([0.7772, 0.125+0.05, 0.01, 0.4835-0.1]) # left bottom width height
#v = np.linspace(np.min(min_), np.max(max_), 8, endpoint=True)
#cbar2 = fig.colorbar(im[_], cax=cbar_ax, ticks=v)
cbar2 = fig.colorbar(im[_], cax=cbar_ax, 
                     boundaries=np.linspace(.6, 1, 1000, endpoint=True), 
                     ticks=np.linspace(.6, 1, 7, endpoint=True),
                     format="%.1f")
cbar2.ax.tick_params(labelsize=size_cbar) 

fig.text(.130, .20, "DNN data (first layer)", fontsize=10, rotation=90,fontweight="bold",color="dimgray")
fig.text(.130, .625, "b.", fontsize=10,fontweight="bold",color="dimgray")
fig.text(.130, .715, "Brain data", fontsize=10, rotation=90, fontweight="bold",color="dimgray")
fig.text(.130, .925, "a.", fontsize=10,fontweight="bold",color="dimgray")

fig.subplots_adjust(top=0.885,
bottom=0.125,
left=0.24,
right=0.77,
hspace=0.28,
wspace=0.0)
#fig.tight_layout()
plt.show()

#%% Images

os.chdir(r"D:\THESIS\IMAGES EXPERIMENT, 336")
img = loadmat("img1.mat")
img = img["img"]
img = img[:,:,:,0]

#%% Show

fig, axs = plt.subplots(nrows=4,ncols=7,figsize=(13,7))
label=["body","hand", "face", "tool", "mani", "nman", "chair"]

axs[0,0].imshow(img[0,:,:],cmap="gray")
axs[1,0].imshow(img[2,:,:],cmap="gray")
axs[2,0].imshow(img[17,:,:],cmap="gray")
axs[3,0].imshow(img[19,:,:],cmap="gray")

axs[0,1].imshow(img[49,:,:],cmap="gray")
axs[1,1].imshow(img[52,:,:],cmap="gray")
axs[2,1].imshow(img[66,:,:],cmap="gray")
axs[3,1].imshow(img[68,:,:],cmap="gray")

axs[0,2].imshow(img[97,:,:],cmap="gray")
axs[1,2].imshow(img[99,:,:],cmap="gray")
axs[2,2].imshow(img[113,:,:],cmap="gray")
axs[3,2].imshow(img[115,:,:],cmap="gray")

axs[0,3].imshow(img[144,:,:],cmap="gray")
axs[1,3].imshow(img[146,:,:],cmap="gray")
axs[2,3].imshow(img[156,:,:],cmap="gray")
axs[3,3].imshow(img[164,:,:],cmap="gray")

axs[0,4].imshow(img[192,:,:],cmap="gray")
axs[1,4].imshow(img[194,:,:],cmap="gray")
axs[2,4].imshow(img[204,:,:],cmap="gray")
axs[3,4].imshow(img[212,:,:],cmap="gray")

axs[0,5].imshow(img[240,:,:],cmap="gray")
axs[1,5].imshow(img[242,:,:],cmap="gray")
axs[2,5].imshow(img[252,:,:],cmap="gray")
axs[3,5].imshow(img[260,:,:],cmap="gray")

axs[0,6].imshow(img[335,:,:],cmap="gray")
axs[1,6].imshow(img[300,:,:],cmap="gray")
axs[2,6].imshow(img[331,:,:],cmap="gray")
axs[3,6].imshow(img[320,:,:],cmap="gray")

for i in range(0,4):
    for j in range(0,7):
        axs[i,j].axis("Off")
        lbl=label[j]
#        axs[0,j].set_title(lbl,size=11,fontweight="bold",color="dimgray")
        axs[0,j].set_title(lbl,size=11,fontweight="bold",color="k")
        
#plt.tight_layout()
plt.show()

#%% Correlations

norm_ant = (ant-np.mean(ant))/np.std(ant)
norm_calc = (calc-np.mean(calc))/np.std(calc)
cos_small_last_norm = [(co-np.mean(co))/np.std(co) for co in cos_small_last]
cos_small_first_norm = [(co-np.mean(co))/np.std(co) for co in cos_small_first]

body=0
hand=1
face=2
tool=3
mani=4
nman=5
chair=6

fig, axs = plt.subplots(1,3,figsize=(13,3))

axs[0].bar(1,ant[hand,tool])
axs[0].bar(2,ant[hand,mani])
axs[0].bar(3,ant[hand,nman])
axs[0].set_title("ITG + \nPosterior IOG", size=9,fontweight="bold",color="dimgray")




