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

#%% Save

os.chdir(r"D:\thesis-scripts\Visualizations and stats for the paper")
ant = {"ant" : ant}
calc = {"calc" : calc}
cos_small_first = {"cos_small_first" : cos_small_first}
cos_small_last = {"cos_small_last" : cos_small_last}

savemat("ant.mat", ant)
savemat("calc.mat", calc)
savemat("cos_small_first.mat", cos_small_first)
savemat("cos_small_last.mat", cos_small_last)

#%% Choosing style

print(mpl.style.available)
mpl.style.use("fivethirtyeight")
mpl.style.use("ggplot")
mpl.style.use("seaborn-dark")

mpl.rcdefaults()
 
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

#%% Body part vs. correlation | LAST LAYER

norm_ant = (ant-np.mean(ant))/np.std(ant)
norm_calc = (calc-np.mean(calc))/np.std(calc)
cos_small_last_norm = [(co-np.mean(co))/np.std(co) for co in cos_small_last]
cos_small_first_norm = [(co-np.mean(co))/np.std(co) for co in cos_small_first]

color_tool="#E69F00"
color_mani="#009E73"
color_nman="#D55E00"

body=0
hand=1
face=2
tool=3
mani=4
nman=5
chair=6

labels=["body", "hand", "face"]
size_ticks=10
size_title=10
color_title="k"
color_ticks="dimgray"

fig, axs = plt.subplots(2,5,figsize=(13,5),sharex=True)
width = 0.25
axs[0,0].bar(np.array([1,2,3])-2*width, [ant[body,tool], ant[hand,tool], ant[face,tool]], width=width,label="tool",color=color_tool)
axs[0,0].bar(np.array([1,2,3])-1*width, [ant[body,mani], ant[hand,mani], ant[face,mani]], width=width,label="mani",color=color_mani)
axs[0,0].bar(np.array([1,2,3])-0*width, [ant[body,nman], ant[hand,nman], ant[face,nman]], width=width,label="nman",color=color_nman)
#axs[0,0].legend()
#axs[0].set_title("ITG + \nPosterior IOG", size=9,fontweight="bold",color="dimgray")

axs[0,0].set_xticks(ticks=[1,2,3])
axs[0,0].set_xticklabels(labels="")
axs[0,0].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
axs[0,0].set_xticklabels(labels=labels, minor=True, size=size_ticks, color=color_ticks)
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")
axs[0,0].set_title("ITG + \nPosterior IOG", size=size_title,fontweight="bold",color=color_title)
#axs[0,0].legend()
axs[0,0].set_yticks([-.5, -.25, 0])

for _ in range(1,5):
    axs[0,_].axis("Off")

for _ in range(0,5):
    axs[1,_].bar(np.array([1,2,3])-2*width, [cos_small_last[_][body,tool], cos_small_last[_][hand,tool], cos_small_last[_][face,tool]], width=width,label="tool",color=color_tool)
    axs[1,_].bar(np.array([1,2,3])-1*width, [cos_small_last[_][body,mani], cos_small_last[_][hand,mani], cos_small_last[_][face,mani]], width=width,label="mani",color=color_mani)
    axs[1,_].bar(np.array([1,2,3])-0*width, [cos_small_last[_][body,nman], cos_small_last[_][hand,nman], cos_small_last[_][face,nman]], width=width,label="nman",color=color_nman)
    
    axs[1,_].set_xticks(ticks=[1,2,3])
    axs[1,_].set_xticklabels(labels="")
    axs[1,_].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
    axs[1,_].set_xticklabels(labels=labels, minor=True, size=size_ticks, color=color_ticks)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")
    axs[1,_].set_title(nets[_], size=size_title,fontweight="bold",color=color_title)
#    axs[1,_].legend()
    
    if _ == 0:
        axs[1,_].set_yticks([0, .25, .5])
    else:
        axs[1,_].set_yticks([0, .25, .5])
        axs[1,_].set_yticklabels(labels="")
        
fig.text(.05, .10, "DNN data (last layer)", fontsize=size_title, rotation=90,fontweight="bold",color=color_title)
fig.text(.05, .475, "b.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.05, .690, "Brain data", fontsize=size_title, rotation=90, fontweight="bold",color=color_title)
fig.text(.05, .925, "a.", fontsize=size_title,fontweight="bold",color=color_title)

plt.figlegend(
    labels=('tool', 'mani', 'nman'),
    loc='upper center',bbox_to_anchor=(0.5, 0.4, 0.735, 0.5))

plt.show()

#%% Body part vs. correlation | FIRST LAYER

body=0
hand=1
face=2
tool=3
mani=4
nman=5
chair=6

color_tool="#E69F00"
color_mani="#009E73"
color_nman="#D55E00"

labels=["body", "hand", "face"]
size_ticks=10
size_title=10
color_title="k"
color_ticks="dimgray"
sizescatter=15

fig, axs = plt.subplots(2,5,figsize=(13,5),sharex=True)
width = 0.25
axs[0,0].bar(np.array([1,2,3])-2*width, [calc[body,tool], calc[hand,tool], calc[face,tool]], width=width,label="tool",color=color_tool)
axs[0,0].bar(np.array([1,2,3])-1*width, [calc[body,mani], calc[hand,mani], calc[face,mani]], width=width,label="mani",color=color_mani)
axs[0,0].bar(np.array([1,2,3])-0*width, [calc[body,nman], calc[hand,nman], calc[face,nman]], width=width,label="nman",color=color_nman)
#axs[0,0].legend()
#axs[0].set_title("ITG + \nPosterior IOG", size=9,fontweight="bold",color="dimgray")

axs[0,0].set_xticks(ticks=[1,2,3])
axs[0,0].set_xticklabels(labels="")
axs[0,0].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
axs[0,0].set_xticklabels(labels=labels, minor=True, size=size_ticks, color=color_ticks)
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")
axs[0,0].set_title("Calcarine\nCortex", size=size_title,fontweight="bold",color=color_title)
#axs[0,0].legend()
axs[0,0].set_yticks(np.linspace(-.1, 0.05,4))


for _ in range(1,5):
    axs[0,_].axis("Off")

for _ in range(0,5):
    axs[1,_].bar(np.array([1,2,3])-2*width, [cos_small_first[_][body,tool], cos_small_first[_][hand,tool], cos_small_first[_][face,tool]], width=width,label="tool",color=color_tool)
    axs[1,_].bar(np.array([1,2,3])-1*width, [cos_small_first[_][body,mani], cos_small_first[_][hand,mani], cos_small_first[_][face,mani]], width=width,label="mani",color=color_mani)
    axs[1,_].bar(np.array([1,2,3])-0*width, [cos_small_first[_][body,nman], cos_small_first[_][hand,nman], cos_small_first[_][face,nman]], width=width,label="nman",color=color_nman)
    
    axs[1,_].set_xticks(ticks=[1,2,3])
    axs[1,_].set_xticklabels(labels="")
    axs[1,_].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
    axs[1,_].set_xticklabels(labels=labels, minor=True, size=size_ticks, color=color_ticks)
    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")
    axs[1,_].set_title(nets[_], size=size_title,fontweight="bold",color=color_title)
#    axs[1,_].legend()
    
    if _ == 0:
        axs[1,_].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    else:
        axs[1,_].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1])
        axs[1,_].set_yticklabels(labels="")
        
fig.text(.05, .10, "DNN data (first layer)", fontsize=size_title, rotation=90,fontweight="bold",color=color_title)
fig.text(.05, .475, "b.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.05, .690, "Brain data", fontsize=size_title, rotation=90, fontweight="bold",color=color_title)
fig.text(.05, .925, "a.", fontsize=size_title,fontweight="bold",color=color_title)


plt.figlegend(
    labels=('tool', 'mani', 'nman'),
    loc='upper center',bbox_to_anchor=(0.5, 0.4, 0.735, 0.5))

plt.show()

#%% VISUALISE SPLIT-HALF | FIRST

size_ticks=10
size_title=10
color_title="k"
color_ticks="dimgray"
cmap="Greens"
ticks=np.array([0,1,2,3,4,5])
labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=[0.5,1.5,2.5,3.5,4.5,5.5]
ticks_minor=range(0,6,1)

min_ = []
max_ = []
for _ in cos_small_first:
    min_.append(np.min(_))
    max_.append(np.max(_))

# First plot
im=[]
fig, axs = plt.subplots(2,5,figsize=(13,5),sharex=True,sharey=True)
im.append(axs[0,0].imshow(calc, cmap=cmap))
axs[0,0].grid(False)

# 2
axs[0,0].set_yticks(ticks=ticks_major)
axs[0,0].set_yticklabels(labels="")
axs[0,0].set_yticks(ticks=ticks_minor, minor=True)
axs[0,0].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")

axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")

axs[0,0].set_title("Calcarine\nCortex", size=size_title,fontweight="bold",color=color_title)

for _ in range(1,5):
    axs[0,_].axis("Off")

for _ in range(0,5):
    im.append(axs[1,_].imshow(cos_small_first[_],cmap=cmap,vmin=np.min(min_),vmax=np.max(max_)))
    axs[1,_].set_title(nets[_], size=size_title,fontweight="bold",color=color_title)
    axs[1,_].grid(False)

    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].set_xticklabels(labels="")
    axs[1,_].set_xticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")

    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")
     
fig.text(.05, .10, "DNN data (first layer)", fontsize=size_title, rotation=90,fontweight="bold",color=color_title)
fig.text(.05, .475, "b.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.05, .690, "Brain data", fontsize=size_title, rotation=90, fontweight="bold",color=color_title)
fig.text(.05, .925, "a.", fontsize=size_title,fontweight="bold",color=color_title)

#plt.figlegend(
#    labels=('tool', 'mani', 'nman'),
#    loc='upper center',bbox_to_anchor=(0.5, 0.4, 0.735, 0.5))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.265, 0.55, 0.005, 0.305]) # left bottom width height
cbar1 = fig.colorbar(im[0], cax=cbar_ax, 
                     boundaries=np.linspace(-0.2, 0.4, 1000, endpoint=True), 
                     ticks=np.linspace(-0.2, 0.4, 7, endpoint=True),
                     format="%.1f")
cbar1.ax.tick_params(labelsize=size_ticks)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.823, 0.132, 0.005, 0.305]) # left bottom width height
cbar2 = fig.colorbar(im[_], cax=cbar_ax, 
                     boundaries=np.linspace(.6, 1, 1000, endpoint=True), 
                     ticks=np.linspace(.6, 1, 7, endpoint=True),
                     format="%.1f")
cbar2.ax.tick_params(labelsize=size_ticks)

plt.show()

#%% VISUALISE SPLIT-HALF | LAST

size_ticks=10
size_title=10
color_title="k"
color_ticks="dimgray"
cmap="Greens"
ticks=np.array([0,1,2,3,4,5])
labels=["body", "hand", "face", "tool", "mani", "nman"]
ticks_major=[0.5,1.5,2.5,3.5,4.5,5.5]
ticks_minor=range(0,6,1)

min_ = []
max_ = []
for _ in cos_small_last:
    min_.append(np.min(_))
    max_.append(np.max(_))

# First plot
im=[]
fig, axs = plt.subplots(2,5,figsize=(13,5),sharex=True,sharey=True)
im.append(axs[0,0].imshow(ant, cmap=cmap))
axs[0,0].grid(False)

# 2
axs[0,0].set_yticks(ticks=ticks_major)
axs[0,0].set_yticklabels(labels="")
axs[0,0].set_yticks(ticks=ticks_minor, minor=True)
axs[0,0].set_yticklabels(labels=labels, minor=True, size=size_ticks, color="dimgray")

axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")

axs[0,0].set_title("ITG +\nPosterior IOG", size=size_title,fontweight="bold",color=color_title)

for _ in range(1,5):
    axs[0,_].axis("Off")

for _ in range(0,5):
    im.append(axs[1,_].imshow(cos_small_last[_],cmap=cmap,vmin=np.min(min_),vmax=np.max(max_)))
    axs[1,_].set_title(nets[_], size=size_title,fontweight="bold",color=color_title)
    axs[1,_].grid(False)

    axs[1,_].set_xticks(ticks=ticks_major)
    axs[1,_].set_xticklabels(labels="")
    axs[1,_].set_xticks(ticks=ticks_minor, minor=True)
    axs[1,_].set_xticklabels(labels=labels, minor=True, rotation=90, size=size_ticks, color="dimgray")

    axs[1,_].tick_params(axis="x", which="major", color ="w")
    axs[1,_].tick_params(axis="x", which="minor", color ="w")
    axs[1,_].tick_params(axis="y", which="major", color ="w")
    axs[1,_].tick_params(axis="y", which="minor", color ="w")
     
fig.text(.05, .10, "DNN data (last layer)", fontsize=size_title, rotation=90,fontweight="bold",color=color_title)
fig.text(.05, .475, "b.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.05, .690, "Brain data", fontsize=size_title, rotation=90, fontweight="bold",color=color_title)
fig.text(.05, .925, "a.", fontsize=size_title,fontweight="bold",color=color_title)

#plt.figlegend(
#    labels=('tool', 'mani', 'nman'),
#    loc='upper center',bbox_to_anchor=(0.5, 0.4, 0.735, 0.5))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.265, 0.55, 0.005, 0.305]) # left bottom width height
cbar1 = fig.colorbar(im[0], cax=cbar_ax, 
                     boundaries=np.linspace(-0.5, 0.8, 1000, endpoint=True), 
                     ticks=np.linspace(-0.5, 0.8, 7, endpoint=True),
                     format="%.1f")
cbar1.ax.tick_params(labelsize=size_ticks)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.823, 0.132, 0.005, 0.305]) # left bottom width height
cbar2 = fig.colorbar(im[_], cax=cbar_ax, 
                     boundaries=np.linspace(0, 0.8, 1000, endpoint=True), 
                     ticks=np.linspace(0, 0.8, 7, endpoint=True),
                     format="%.1f")
cbar2.ax.tick_params(labelsize=size_ticks)

plt.show()

#%% MDS

"""
I'm getting MDS from Matlab
1.) mbscale (rdm, dimensons=2)
2.) I get as a result: a matrix: 6, 2
3.) get scatter plot by using plot(matrix[1 column], matrix[2 column])

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat

os.chdir(r"D:\thesis-scripts\Visualizations and stats for the paper")

Y_a = loadmat("Y_a.mat")["Y_a"]
Y_c = loadmat("Y_c.mat")["Y_c"]
Y_last = loadmat("Y_last.mat")["Y_last"]
Y_first = loadmat("Y_first.mat")["Y_first"]
Y_labels=["body","hand","face","tool","mani","nman"]
colors=["orange", "black", "green","red","blue", "purple"]

#%% Visualize 
###%%%%%%%%%%%%%%%%%%%%%%%%%%%% Last

body=0
hand=1
face=2
tool=3
mani=4
nman=5
chair=6

loc="left"

labels=["body", "hand", "face"]
size_ticks=10
size_title=10
size_yticks=8
color_title="k"
color_ticks="dimgray"
width = 0.15

color_tool="#E69F00"
color_mani="#009E73"
color_nman="#D55E00"
color_body="#0072B2"
color_hand="#999999"
color_face="#CC79A7"
colors=[color_body, color_hand, color_face,
        color_tool, color_mani, color_nman]
size_scatter=80

fig, axs = plt.subplots(6,2,figsize=(5,15))

axs[0,0].bar(np.array([1,2,3])-2*width, [ant[body,tool], ant[hand,tool], ant[face,tool]], width=width,label="tool",color=color_tool,edgecolor="black")
axs[0,0].bar(np.array([1,2,3])-1*width, [ant[body,mani], ant[hand,mani], ant[face,mani]], width=width,label="mani",color=color_mani,edgecolor="black")
axs[0,0].bar(np.array([1,2,3])-0*width, [ant[body,nman], ant[hand,nman], ant[face,nman]], width=width,label="nman",color=color_nman,edgecolor="black")
axs[0,0].set_xticks(ticks=[1,2,3])
axs[0,0].grid(True)
axs[0,0].set_xticklabels(labels="")
axs[0,0].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
axs[0,0].set_xticklabels("")
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")
axs[0,0].set_title("ITG + \nPosterior IOG",loc=loc,size=size_title,fontweight="bold",color=color_title)
axs[0,0].set_yticks(np.linspace(-.5,0,5))
axs[0,0].set_yticklabels([-.5, "", -.25, "", 0], size=size_yticks)

for label in range(0,6):
    axs[0,1].scatter(Y_a[label,0], Y_a[label,1],label=Y_labels[label],color=colors[label],s=size_scatter,edgecolor="black")
axs[0,1].grid(False)
axs[0,1].tick_params(axis="x", which="major", color ="w")
axs[0,1].tick_params(axis="x", which="minor", color ="w")
axs[0,1].tick_params(axis="y", which="major", color ="w")
axs[0,1].tick_params(axis="y", which="minor", color ="w")
axs[0,1].set_xticks([-1,1])
axs[0,1].set_xticklabels(labels="")
axs[0,1].set_yticks([-1,1])
axs[0,1].set_yticklabels(labels="")
axs[0,1].set_xlim([-1.102,1.102])
axs[0,1].set_ylim([-1.102,1.102])
axs[0,1].set_aspect(1)

for _ in range(0,5):
    axs[_+1,0].grid(True)
    axs[_+1,0].bar(np.array([1,2,3])-2*width, [cos_small_last[_][body,tool], cos_small_last[_][hand,tool], cos_small_last[_][face,tool]], width=width,label="tool",color=color_tool,edgecolor="black")
    axs[_+1,0].bar(np.array([1,2,3])-1*width, [cos_small_last[_][body,mani], cos_small_last[_][hand,mani], cos_small_last[_][face,mani]], width=width,label="mani",color=color_mani,edgecolor="black")
    axs[_+1,0].bar(np.array([1,2,3])-0*width, [cos_small_last[_][body,nman], cos_small_last[_][hand,nman], cos_small_last[_][face,nman]], width=width,label="nman",color=color_nman,edgecolor="black")   
    axs[_+1,0].set_xticks(ticks=[1,2,3])
    axs[_+1,0].set_xticklabels(labels="")
    axs[_+1,0].tick_params(axis="x", which="major", color ="w")
    axs[_+1,0].tick_params(axis="x", which="minor", color ="w")
    axs[_+1,0].tick_params(axis="y", which="major", color ="w")
    axs[_+1,0].tick_params(axis="y", which="minor", color ="w")
    axs[_+1,0].set_yticks(np.linspace(0,.5,5))
    axs[_+1,0].set_yticklabels([0, "", .25, "", .5], size=size_yticks)
        
    for label in range(0,6):
        axs[_+1,1].scatter(Y_last[label,0,_], Y_last[label,1,_],label=Y_labels[label],color=colors[label],s=size_scatter,edgecolor="black")
    axs[_+1,1].grid(False)
#    axs[_+1,1].set_xticks(ticks=[1,2,3])
    axs[_+1,1].set_xticklabels(labels="")
    axs[_+1,1].set_yticklabels(labels="")
    axs[_+1,1].tick_params(axis="x", which="major", color ="w")
    axs[_+1,1].tick_params(axis="x", which="minor", color ="w")
    axs[_+1,1].tick_params(axis="y", which="major", color ="w")
    axs[_+1,1].tick_params(axis="y", which="minor", color ="w")
    axs[_+1,1].set_xlim([-.4, .4])
    axs[_+1,1].set_ylim([-.4, .4])
    axs[_+1,1].set_aspect(1)
    
    if _ == 4:
        axs[_+1,0].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
        axs[_+1,0].set_xticklabels(labels=labels, minor=True, size=size_ticks, color=color_ticks)
    else:
        axs[_+1,0].tick_params(axis="x", color ="w")
        axs[_+1,0].tick_params(axis="y", color ="w")
    axs[_+1,0].set_title(nets[_], loc=loc,size=size_title,fontweight="bold",color=color_title)
    
fig.text(.09, .500, "DNN data (last layer)", fontsize=size_title, rotation=90,fontweight="bold",color=color_title)
fig.text(.09, .790, "b.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.09, .835, "Brain data", fontsize=size_title, rotation=90, fontweight="bold",color=color_title)
fig.text(.09, .950, "a.", fontsize=size_title,fontweight="bold",color=color_title)

fig.text(.40, 0.98, "I.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.70, 0.98, "II.", fontsize=size_title,fontweight="bold",color=color_title)


handles, labels = axs[0,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',ncol=3)

plt.subplots_adjust(top=0.920,
bottom=0.095,
left=0.22,
right=0.895,
hspace=0.25,
wspace=0.1)

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% First

body=0
hand=1
face=2
tool=3
mani=4
nman=5
chair=6

loc="left"

labels=["body", "hand", "face"]
size_ticks=10
size_title=10
size_yticks=8
color_title="k"
color_ticks="dimgray"
width = 0.15

color_tool="#E69F00"
color_mani="#009E73"
color_nman="#D55E00"
color_body="#0072B2"
color_hand="#999999"
color_face="#CC79A7"
colors=[color_body, color_hand, color_face,
        color_tool, color_mani, color_nman]
size_scatter=80

fig, axs = plt.subplots(6,2,figsize=(5,15))

axs[0,0].bar(np.array([1,2,3])-2*width, [calc[body,tool], calc[hand,tool], calc[face,tool]], width=width,label="tool",color=color_tool,edgecolor="black")
axs[0,0].bar(np.array([1,2,3])-1*width, [calc[body,mani], calc[hand,mani], calc[face,mani]], width=width,label="mani",color=color_mani,edgecolor="black")
axs[0,0].bar(np.array([1,2,3])-0*width, [calc[body,nman], calc[hand,nman], calc[face,nman]], width=width,label="nman",color=color_nman,edgecolor="black")
axs[0,0].set_xticks(ticks=[1,2,3])
axs[0,0].grid(True)
axs[0,0].set_xticklabels(labels="")
axs[0,0].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
axs[0,0].set_xticklabels("")
axs[0,0].tick_params(axis="x", which="major", color ="w")
axs[0,0].tick_params(axis="x", which="minor", color ="w")
axs[0,0].tick_params(axis="y", which="major", color ="w")
axs[0,0].tick_params(axis="y", which="minor", color ="w")
axs[0,0].set_title("Calcarine \nCortex",loc=loc,size=size_title,fontweight="bold",color=color_title)
axs[0,0].set_yticks([-0.1,-0.05,0,0.05])
axs[0,0].set_yticklabels([-0.1,-0.05,0,0.05], size=size_yticks)

for label in range(0,6):
    axs[0,1].scatter(Y_c[label,0], Y_c[label,1],label=Y_labels[label],color=colors[label],s=size_scatter,edgecolor="black")
axs[0,1].grid(False)
axs[0,1].tick_params(axis="x", which="major", color ="w")
axs[0,1].tick_params(axis="x", which="minor", color ="w")
axs[0,1].tick_params(axis="y", which="major", color ="w")
axs[0,1].tick_params(axis="y", which="minor", color ="w")
axs[0,1].set_xticks([-1,1])
axs[0,1].set_xticklabels(labels="")
axs[0,1].set_yticks([-1,1])
axs[0,1].set_yticklabels(labels="")
axs[0,1].set_xlim([-.4105,.4105])
axs[0,1].set_ylim([-.4105,.4105])
axs[0,1].set_aspect(1)

for _ in range(0,5):
    axs[_+1,0].grid(True)
    axs[_+1,0].bar(np.array([1,2,3])-2*width, [cos_small_first[_][body,tool], cos_small_first[_][hand,tool], cos_small_first[_][face,tool]], width=width,label="tool",color=color_tool,edgecolor="black")
    axs[_+1,0].bar(np.array([1,2,3])-1*width, [cos_small_first[_][body,mani], cos_small_first[_][hand,mani], cos_small_first[_][face,mani]], width=width,label="mani",color=color_mani,edgecolor="black")
    axs[_+1,0].bar(np.array([1,2,3])-0*width, [cos_small_first[_][body,nman], cos_small_first[_][hand,nman], cos_small_first[_][face,nman]], width=width,label="nman",color=color_nman,edgecolor="black")   
    axs[_+1,0].set_xticks(ticks=[1,2,3])
    axs[_+1,0].set_xticklabels(labels="")
    axs[_+1,0].tick_params(axis="x", which="major", color ="w")
    axs[_+1,0].tick_params(axis="x", which="minor", color ="w")
    axs[_+1,0].tick_params(axis="y", which="major", color ="w")
    axs[_+1,0].tick_params(axis="y", which="minor", color ="w")
    axs[_+1,0].set_yticks([0,.2,.4,.6,.8,1])
    axs[_+1,0].set_yticklabels([0,.2,.4,.6,.8,1], size=size_yticks)
        
    for label in range(0,6):
        axs[_+1,1].scatter(Y_first[label,0,_], Y_first[label,1,_],label=Y_labels[label],color=colors[label],s=size_scatter,edgecolor="black")
    axs[_+1,1].grid(False)
#    axs[_+1,1].set_xticks(ticks=[1,2,3])
    axs[_+1,1].set_xticklabels(labels="")
    axs[_+1,1].set_yticklabels(labels="")
    axs[_+1,1].tick_params(axis="x", which="major", color ="w")
    axs[_+1,1].tick_params(axis="x", which="minor", color ="w")
    axs[_+1,1].tick_params(axis="y", which="major", color ="w")
    axs[_+1,1].tick_params(axis="y", which="minor", color ="w")
    axs[_+1,1].set_xlim([-.1, .1])
    axs[_+1,1].set_ylim([-.1, .1])
    axs[_+1,1].set_aspect(1)
    
    if _ == 4:
        axs[_+1,0].set_xticks(ticks=[1-width,2-width,3-width], minor=True)
        axs[_+1,0].set_xticklabels(labels=labels, minor=True, size=size_ticks, color=color_ticks)
    else:
        axs[_+1,0].tick_params(axis="x", color ="w")
        axs[_+1,0].tick_params(axis="y", color ="w")
    axs[_+1,0].set_title(nets[_], loc=loc,size=size_title,fontweight="bold",color=color_title)
    
fig.text(.09, .500, "DNN data (first layer)", fontsize=size_title, rotation=90,fontweight="bold",color=color_title)
fig.text(.09, .790, "b.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.09, .835, "Brain data", fontsize=size_title, rotation=90, fontweight="bold",color=color_title)
fig.text(.09, .950, "a.", fontsize=size_title,fontweight="bold",color=color_title)

fig.text(.40, 0.98, "I.", fontsize=size_title,fontweight="bold",color=color_title)
fig.text(.70, 0.98, "II.", fontsize=size_title,fontweight="bold",color=color_title)


handles, labels = axs[0,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',ncol=3)

plt.subplots_adjust(top=0.920,
bottom=0.095,
left=0.22,
right=0.895,
hspace=0.25,
wspace=0.1)

plt.show()

#%% MDS (only)

fig,ax=plt.subplots(figsize=(5,3))
ax.arrow(.08,.3,.8,0, width=0.01,color="dimgray")
#ax.set_xticks([1])
#ax.set_xticklabels("")
#ax.set_yticks([1])
#ax.set_yticklabels("")
#ax.tick_params(axis="x",color="w")
#ax.tick_params(axis="y",color="w")
ax.axis("Off")
