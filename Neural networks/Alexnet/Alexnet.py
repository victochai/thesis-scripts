#%% Loading libraries

import scipy.io
import matplotlib.pyplot as plt
import os

#%% Mat

mat = scipy.io.loadmat('correlations_alex_2.mat')
mat = mat["correlations_AlEX"]

body_objects = []
body_tool = []
body_man = []
body_nman = []

for _ in range(0,3):
    body_objects.append(mat[0][0][_][0])
    body_tool.append(mat[1][0][_][0])
    body_man.append(mat[2][0][_][0])
    body_nman.append(mat[3][0][_][0])
    
#%% Plot

for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects (tools + manipulable + nonmanipulable) in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

#%% All plots

fig = plt.figure()
fig.suptitle("ALEXNET")
# 1
plt.subplot(2,2,1)
for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n (tools + manipulable + nonmanipulable) in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# 2
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
#v plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# 3
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# 4
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plotting other analysis
# Small matrices

conv_matrices = []
for layer in os.listdir():
    mat = scipy.io.loadmat(layer)
    l = layer.rstrip(".mat")
    conv_matrices.append(mat[l])
    del l, mat
del layer

fig = plt.figure()
fig.suptitle("ALEXNET\nEvery condition is averaged\nBodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(conv_matrices)):
    plt.subplot(2,4,_+1)
    plt.imshow(conv_matrices[_])
#    plt.imshow(conv_matrices[_],cmap="cividis")
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

#%% Big matrices  

conv_matrices = []
for layer in os.listdir():
    mat = scipy.io.loadmat(layer)
    l = layer.rstrip(".mat")
    conv_matrices.append(mat[l])
    del l, mat
del layer

fig = plt.figure()
fig.suptitle("ALEXNET\nBodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(conv_matrices)):
    plt.subplot(2,4,_+1)
    plt.imshow(conv_matrices[_])    
#    plt.imshow(conv_matrices[_],cmap="cividis")
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  



