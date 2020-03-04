#%% Modules

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
import os

current_dir = r"C:\Users\victo\Desktop\nns_for_thesis"
os.chdir(current_dir)
p = pathlib.Path.cwd()

#%% Inception layers (if needed)

# Layers
a = [(("mod_A" + str(_) + " ") * 3).split(" ") for _ in range(1,4)]
a = [_[:-1] for _ in a]
a = [val for sublist in a for val in sublist]

norm1 = [("n1_" + str(_) + " ").split(" ") for _ in range(1,4)]
norm1 = [_[:-1] for _ in norm1]
norm1 = [val for sublist in norm1 for val in sublist]

b = [(("mod_B" + str(_) + " ") * 5).split(" ") for _ in range(1,5)]
b = [_[:-1] for _ in b]
b = [val for sublist in b for val in sublist]

norm2 = [("n2_" + str(_) + " ").split(" ") for _ in range(1,5)]
norm2 = [_[:-1] for _ in norm2]
norm2 = [val for sublist in norm2 for val in sublist]

c = [(("mod_C" + str(_) + " ") * 3).split(" ") for _ in range(1,3)]
c = [_[:-1] for _ in c]
c = [val for sublist in c for val in sublist]

layers = ["conv1", "conv2", "conv3", "conv4", "conv5"] + a + norm1 + b + norm2 + c
layers.append("FC")

del a, b, c, norm1, norm2

#%% Correlations between body parts and objects

mat = scipy.io.loadmat('correlations_INC_original.mat')
mat = mat["correlations_INC"]

body_objects = []
body_tool = []
body_man = []
body_nman = []

for _ in range(0,3):
    body_objects.append(mat[0][0][_][0])
    body_tool.append(mat[1][0][_][0])
    body_man.append(mat[2][0][_][0])
    body_nman.append(mat[3][0][_][0])
    
#%% Visualization
    
# Objects
for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.xlabel("Layer") 
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects (tools + manipulable + nonmanipulable) in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

# Tools
for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

# Man
for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
    plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

# NMan
for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plot all

fig = plt.figure()
fig.suptitle("INCEPTION V3 | ORIGINAL IMAGES")
# 1
plt.subplot(2,2,1)
for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n(tools + manipulable + nonmanipulable) in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()
# 2
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()
# 3
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()
# 4
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Feature maps

# Big
import glob
co_dir = glob.glob(str(p/'co_conv2d_*'))
mat = [scipy.io.loadmat(co)["co"] for co in co_dir]

fig = plt.figure()
fig.suptitle("InceptionV3\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(mat)):
    plt.subplot(6,8,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(mat[_])
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=6)
plt.show()    

# Small
import glob
co_dir = glob.glob(str(p/'co_small_conv2d_*'))
mat = [scipy.io.loadmat(co)["co_small"] for co in co_dir]

fig = plt.figure()
fig.suptitle("InceptionV3\nEvery condition is averaged\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(mat)):
    plt.subplot(6,8,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(mat[_])
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=6)
plt.show()    
