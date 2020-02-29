#%% Import modules

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#%% Brain representations + Nets

# Brain
work_dir = r"C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations"
os.chdir(work_dir)
ant = scipy.io.loadmat("anterior_big_MATRIX.mat")["anterior_big_MATRIX"]
ant_left = scipy.io.loadmat("anterior_left.mat")["anterior_left"]
ant_right = scipy.io.loadmat("anterior_right.mat")["anterior_right"]

# Inception
inception_dir = r"C:\Users\victo\Desktop\thesis-scripts\Neural networks\InceptionV3\Conv small"
os.chdir(inception_dir)
inception = scipy.io.loadmat("co_small_conv2d_48_pred.mat")["co_small"]

# Alexnet
alexnet_dir = r"C:\Users\victo\Desktop\thesis-scripts\Neural networks\Alexnet\Conv Small"
os.chdir(alexnet_dir)
alexnet = scipy.io.loadmat(r"fc8_co_small.mat")["fc8_co_small"]

# VGG19
vgg_dir = r"C:\Users\victo\Desktop\thesis-scripts\Neural networks\VGG19\Conv small"
os.chdir(vgg_dir)
vgg19 = scipy.io.loadmat(r"fc8_co_small.mat")["fc8_co_small"]

os.chdir(work_dir)

#%% Plot

fig = plt.figure()
fig.suptitle("Brain vs. Last layer of Neural Networks")

plt.subplot(2,3,1)
plt.imshow(np.mean(ant, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG")

plt.subplot(2,3,2)
plt.imshow(np.mean(ant_left, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (left)")

plt.subplot(2,3,3)
plt.imshow(np.mean(ant_right, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (right)")

plt.subplot(2,3,6)
plt.imshow(inception)
plt.colorbar()
plt.axis("off")
plt.title("InceptionV3")

plt.subplot(2,3,4)
plt.imshow(alexnet)
plt.colorbar()
plt.axis("off")
plt.title("Alexnet")

plt.subplot(2,3,5)
plt.imshow(vgg19)
plt.colorbar()
plt.axis("off")
plt.title("VGG19")

plt.show()

#%% Correlation between body parts and objects in the brain

brain_regions = [
        "calc. cortex", 
        "occip. pole",
        "post. IOG",
        "ITG + ant. IOG (right)"
        ]
body = scipy.io.loadmat("body_right.mat")["body"]
hand = scipy.io.loadmat("hand_right.mat")["hand"]
face = scipy.io.loadmat("face_right.mat")["face"]

# Objects
plt.plot(body[0,:],'-o')
plt.plot(hand[0,:],'-o')
plt.plot(face[0,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects (tools + manipulable + nonmanipulable) in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

# Tools
plt.plot(body[1,:],'-o')
plt.plot(hand[1,:],'-o')
plt.plot(face[1,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

# Man
plt.plot(body[2,:],'-o')
plt.plot(hand[2,:],'-o')
plt.plot(face[2,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

# Nman
plt.plot(body[3,:],'-o')
plt.plot(hand[3,:],'-o')
plt.plot(face[3,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plot all

fig = plt.figure()
fig.suptitle("BRAIN")
# Objects
plt.subplot(2,2,1)
plt.plot(body[0,:],'-o')
plt.plot(hand[0,:],'-o')
plt.plot(face[0,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n(tools + manipulable + nonmanipulable) in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# Tools
plt.subplot(2,2,2)
plt.plot(body[1,:],'-o')
plt.plot(hand[1,:],'-o')
plt.plot(face[1,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# Man
plt.subplot(2,2,3)
plt.plot(body[2,:],'-o')
plt.plot(hand[2,:],'-o')
plt.plot(face[2,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
# Nman
plt.subplot(2,2,4)
plt.plot(body[3,:],'-o')
plt.plot(hand[3,:],'-o')
plt.plot(face[3,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
# plt.xticks(list(range(0,48)), list(range(1,49)))
plt.xticks(list(range(0,4)), brain_regions, fontsize=8)
plt.xlabel("Brain region")
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Brain vs. Neural nets | Loading

# Alexnet
os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\Neural networks\Alexnet\Correlations. Body parts and objects")
mat = scipy.io.loadmat('correlations_alex_2.mat')["correlations_AlEX"]
body_objects_ALEX = []
body_tool_ALEX = []
body_man_ALEX = []
body_nman_ALEX = []
for _ in range(0,3):
    body_objects_ALEX.append(mat[0][0][_][0])
    body_tool_ALEX.append(mat[1][0][_][0])
    body_man_ALEX.append(mat[2][0][_][0])
    body_nman_ALEX.append(mat[3][0][_][0])
del mat

#VGG19
os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\Neural networks\VGG19\Correlations. Body parts and objects")  
mat = scipy.io.loadmat('correlations_vgg_2.mat')["correlations_VGG"] 
body_objects_VGG = []
body_tool_VGG = []
body_man_VGG = []
body_nman_VGG = []
for _ in range(0,3):
    body_objects_VGG.append(mat[0][0][_][0])
    body_tool_VGG.append(mat[1][0][_][0])
    body_man_VGG.append(mat[2][0][_][0])
    body_nman_VGG.append(mat[3][0][_][0])    
del mat

# InceptionV3
os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\Neural networks\InceptionV3\Correlations. Body parts and objects")   
mat = scipy.io.loadmat('correlations_INC.mat')["correlations_INC"]
body_objects_INC = []
body_tool_INC = []
body_man_INC = []
body_nman_INC = []

for _ in range(0,3):
    body_objects_INC.append(mat[0][0][_][0])
    body_tool_INC.append(mat[1][0][_][0])
    body_man_INC.append(mat[2][0][_][0])
    body_nman_INC.append(mat[3][0][_][0])

#%% Plotting OBJECTS

fig = plt.figure()
fig.suptitle("Brain vs. Neural Network")

# Brain vs. Objects
plt.subplot(2,2,1)
plt.plot(body[0,:],'-o')
plt.plot(hand[0,:],'-o')
plt.plot(face[0,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
plt.xticks(list(range(0,4)), brain_regions, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n(tools + manipulable + nonmanipulable) in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# Alexnet vs. Objects
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_objects_ALEX[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], fontsize=6, rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n (tools + manipulable + nonmanipulable) in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# VGG19 vs. Objects
# Layers
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
# Plot
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_objects_VGG[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=80, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n(tools + manipulable + nonmanipulable) in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# InceptioV3 vs. Objects
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
# Plot
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_objects_INC[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and objects\n(tools + manipulable + nonmanipulable) in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plotting TOOLS

fig = plt.figure()
fig.suptitle("Brain vs. Neural Network")

# Brain vs. Tools
plt.subplot(2,2,1)
plt.plot(body[1,:],'-o')
plt.plot(hand[1,:],'-o')
plt.plot(face[1,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
plt.xticks(list(range(0,4)), brain_regions, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# Alexnet vs. Tools
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_tool_ALEX[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], fontsize=6, rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# VGG19 vs. Tools
# Layers
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
# Plot
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_tool_VGG[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=80, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# InceptioV3 vs. Tools
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
# Plot
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_tool_INC[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and tools in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plotting MAN

fig = plt.figure()
fig.suptitle("Brain vs. Neural Network")

# Brain vs. MAN
plt.subplot(2,2,1)
plt.plot(body[2,:],'-o')
plt.plot(hand[2,:],'-o')
plt.plot(face[2,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
plt.xticks(list(range(0,4)), brain_regions, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# Alexnet vs. MAN
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_man_ALEX[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], fontsize=6, rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# VGG19 vs. MAN
# Layers
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
# Plot
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_man_VGG[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=80, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# InceptioV3 vs. MAN
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
# Plot
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_man_INC[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and manipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Plotting NMAN

fig = plt.figure()
fig.suptitle("Brain vs. Neural Network")

# Brain vs. NMAN
plt.subplot(2,2,1)
plt.plot(body[3,:],'-o')
plt.plot(hand[3,:],'-o')
plt.plot(face[3,:],'-o')
plt.grid()
plt.ylim((-0.5,0.1))
plt.xticks(list(range(0,4)), brain_regions, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in 4 brain regions")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# Alexnet vs. NMAN
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_nman_ALEX[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], fontsize=6, rotation=0)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in Alexnet layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# VGG19 vs. NMAN
# Layers
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
# Plot
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_nman_VGG[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
# plt.xticks(list(range(0,19)), list(range(1,20)))
plt.xticks(list(range(0,19)), layers, rotation=80, fontsize=6)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in VGG19 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')

# InceptioV3 vs. NMAN
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
# Plot
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_nman_INC[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% Brain representations

work_dir = r"C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations"
os.chdir(work_dir)

# Anterior
ant = scipy.io.loadmat("anterior_big_MATRIX.mat")["anterior_big_MATRIX"]
ant_left = scipy.io.loadmat("anterior_left.mat")["anterior_left"]
ant_right = scipy.io.loadmat("anterior_right.mat")["anterior_right"]

# Posterior
pos = scipy.io.loadmat("pos_res2_mvpa_MATRIX.mat")["pos_res2_mvpa"]
op = scipy.io.loadmat("OP_MATRIX.mat")["OP_MATRIX"]
calc = scipy.io.loadmat("CALC_MATRIX.mat")["CALC_MATRIX"]
op_calc = scipy.io.loadmat("OP_CALC_MATRIX.mat")["OP_CALC_MATRIX"]

# Plot

fig = plt.figure()
fig.suptitle("Brain split-half correlations\nBody, hand, face, tool, man, nman, chair")

plt.subplot(2,4,1)
plt.imshow(np.mean(ant, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG")

plt.subplot(2,4,2)
plt.imshow(np.mean(ant_left, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (left)")

plt.subplot(2,4,3)
plt.imshow(np.mean(ant_right, 2))
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (right)")

plt.subplot(2,4,5)
plt.imshow(np.mean(pos, 2))
plt.colorbar()
plt.axis("off")
plt.title("Posterior IOG")

plt.subplot(2,4,6)
plt.imshow(np.mean(calc, 2))
plt.colorbar()
plt.axis("off")
plt.title("Calcarine cortex")

plt.subplot(2,4,7)
plt.imshow(np.mean(op, 2))
plt.colorbar()
plt.axis("off")
plt.title("Occipital pole")

plt.subplot(2,4,8)
plt.imshow(np.mean(op_calc, 2))
plt.colorbar()
plt.axis("off")
plt.title("Calc. cort. + Occip. pole")

plt.show()
