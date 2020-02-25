
import scipy.io
import matplotlib.pyplot as plt
import os

#%% Mat

mat = scipy.io.loadmat('correlations_VGG_2.mat')
mat = mat["correlations_VGG"]

body_objects = []
body_tool = []
body_man = []
body_nman = []

for _ in range(0,3):
    body_objects.append(mat[0][0][_][0])
    body_tool.append(mat[1][0][_][0])
    body_man.append(mat[2][0][_][0])
    body_nman.append(mat[3][0][_][0])
    
layers = ['conv1_1',
        'conv1_2',
        'conv2_1',
        'conv2_2',
        'conv3_1',
        'conv3_2',
        'conv3_3',
        'conv3_4',
        'conv4_1',
        'conv4_2',
        'conv4_3',
        'conv4_4',
        'conv5_1',
        'conv5_2',
        'conv5_3',
        'conv5_4',
        'fc6',
        'fc7',
        'fc8']
    
#%% Plot

for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects (tools + manipulable + nonmanipulable) in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45)
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% All plots

fig = plt.figure()
fig.suptitle("VGG19")
# 1
plt.subplot(2,2,1)
for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n (tools + manipupulable + nonmanipulable) in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# 2
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# 3
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# 4
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=45, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plot different analysis
# Small conv

conv_matrices = []
for layer in os.listdir():
    mat = scipy.io.loadmat(layer)
    l = layer.rstrip(".mat")
    conv_matrices.append(mat[l])
    del l, mat
del layer

fig = plt.figure()
fig.suptitle("VGG19\nEvery condition is averaged\nBodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(conv_matrices)):
    plt.subplot(3,7,_+1)
#    plt.imshow(conv_matrices[_])
    plt.imshow(conv_matrices[_],cmap="cividis")
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show() 

#%% Big conv

conv_matrices = []
for layer in os.listdir():
    mat = scipy.io.loadmat(layer)
    l = layer.rstrip(".mat")
    conv_matrices.append(mat[l])
    del l, mat
del layer

fig = plt.figure()
fig.suptitle("VGG19\nBodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(conv_matrices)):
    plt.subplot(3,7,_+1)
#    plt.imshow(conv_matrices[_])    
    plt.imshow(conv_matrices[_],cmap="cividis")
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  
