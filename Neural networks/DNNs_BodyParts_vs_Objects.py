#%% LOAD

import os
import scipy.io
import matplotlib.pyplot as plt

#%% LOAD cos_small

# ALEX
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv small, different averaging")
alexnet = scipy.io.loadmat("cos_small")["cos_small"]

# VGG19
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv small, different averaging")
vgg19 = scipy.io.loadmat("cos_small")["cos_small"]

#INCEPTIONV3
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Experimental images\Conv small, different averaging")
inceptionv3 = scipy.io.loadmat("cos_small")["cos_small"]

#ResNet-50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Experiment images\Conv small, different averaging")
resnet50 = scipy.io.loadmat("cos_small")["cos_small"]

#ResNet-101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Experimental images\Conv small, different averaging")
resnet101 = scipy.io.loadmat("cos_small")["cos_small"]

#InceptionResNetV2
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Experimental images\Conv small, different averaging")
inception_resnetv2 = scipy.io.loadmat("cos_small")["cos_small"]

#DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Experimental images\Conv small, different averaging")
densenet201 = scipy.io.loadmat("cos_small")["cos_small"]

os.chdir(r"D:\thesis-scripts\Neural networks")

#%% SAVE ALL

allnets = {
        "alexnet" : alexnet,
        "vgg19" : vgg19,
        "inceptionv3" : inceptionv3,
        "resnet50" : resnet50,
        "resnet101" : resnet101,
        "inception_resnetv2" : inception_resnetv2,
        "densenet201" : densenet201
        }

scipy.io.savemat("allnets.mat", allnets)

#%% Normalization

import numpy as np
alexnet_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in alexnet]
vgg19_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in vgg19]
inceptionv3_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in inceptionv3]
resnet50_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in resnet50]
resnet101_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in resnet101]
inception_resnetv2_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in inception_resnetv2]
densenet201_norm = [(matrix - np.mean(matrix))/np.std(matrix) for matrix in densenet201]

#%% Save

allnets_norm = {
        "alexnet_norm" : alexnet_norm,
        "vgg19_norm" : vgg19_norm,
        "inceptionv3_norm" : inceptionv3_norm,
        "resnet50_norm" : resnet50_norm,
        "resnet101_norm" : resnet101_norm,
        "inception_resnetv2_norm" : inception_resnetv2_norm,
        "densenet201_norm" : densenet201_norm
        }    
    
scipy.io.savemat("allnets_norm.mat", allnets_norm)

#%% Body parts vs. Objects

allnets_norm = [
        alexnet_norm, # 8 layers
        vgg19_norm, # 19 layers
        inceptionv3_norm, # 49 layers
        resnet50_norm, # 50 layers
        resnet101_norm, # 101 layers
        inception_resnetv2_norm, # 176 layers
        densenet201_norm # 201 layers
        ] # A list in a list

body = 0
face = 1
hand = 2
tool = 3
mani = 4
nman = 5
chair = 6

del alexnet, alexnet_norm, vgg19, vgg19_norm, densenet201, densenet201_norm, resnet101, resnet101_norm, resnet50, resnet50_norm, inceptionv3, inceptionv3_norm, inception_resnetv2, inception_resnetv2_norm

#%% BODY parts vs. OBJECTS

body_tool = []
face_tool = []
hand_tool = []

body_mani = []
face_mani = []
hand_mani = []

body_nman = []
face_nman = []
hand_nman = []

body_objects = []
face_objects = []
hand_objects = []

for net in allnets_norm:
    
    # Tool
    body_tool.append([matrix[body, tool] for matrix in net])
    face_tool.append([matrix[face, tool] for matrix in net])
    hand_tool.append([matrix[hand, tool] for matrix in net])
    
    # Mani
    body_mani.append([matrix[body, mani] for matrix in net])
    face_mani.append([matrix[face, mani] for matrix in net])
    hand_mani.append([matrix[hand, mani] for matrix in net])

    # Nman
    body_nman.append([matrix[body, nman] for matrix in net])
    face_nman.append([matrix[face, nman] for matrix in net])
    hand_nman.append([matrix[hand, nman] for matrix in net])
    
    # Objects
    body_objects.append([          
            np.mean([matrix[body, tool], matrix[body, mani], matrix[body, nman]])
            for matrix in net])
    face_objects.append([          
            np.mean([matrix[face, tool], matrix[face, mani], matrix[face, nman]])
            for matrix in net])    
    hand_objects.append([          
            np.mean([matrix[hand, tool], matrix[hand, mani], matrix[hand, nman]])
            for matrix in net])    

#%% PLOT ALEXNET
    
fig = plt.figure()
fig.suptitle("ALEXNET | ORIGINAL IMAGES ", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[0])
plt.plot(face_objects[0])
plt.plot(hand_objects[0])
plt.grid()
plt.ylim((-1.7,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[0])
plt.plot(face_tool[0])
plt.plot(hand_tool[0])
plt.grid()
plt.ylim((-1.7,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[0])
plt.plot(face_mani[0])
plt.plot(hand_mani[0])
plt.grid()
plt.ylim((-1.7,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[0])
plt.plot(face_nman[0])
plt.plot(hand_nman[0])
plt.grid()
plt.ylim((-1.7,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% VGG-19

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

fig = plt.figure()
fig.suptitle("VGG-19", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[1])
plt.plot(face_objects[1])
plt.plot(hand_objects[1])
plt.grid()
plt.ylim((-1.9,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[1])
plt.plot(face_tool[1])
plt.plot(hand_tool[1])
plt.grid()
plt.ylim((-1.9,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[1])
plt.plot(face_mani[1])
plt.plot(hand_mani[1])
plt.grid()
plt.ylim((-1.9,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[1])
plt.plot(face_nman[1])
plt.plot(hand_nman[1])
plt.grid()
plt.ylim((-1.9,0.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% INCEPTION-V3

fig = plt.figure()
fig.suptitle("INCEPTION-V3", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[2])
plt.plot(face_objects[2])
plt.plot(hand_objects[2])
plt.grid()
plt.ylim((-2,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[2])
plt.plot(face_tool[2])
plt.plot(hand_tool[2])
plt.grid()
plt.ylim((-2,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[2])
plt.plot(face_mani[2])
plt.plot(hand_mani[2])
plt.grid()
plt.ylim((-2,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[2])
plt.plot(face_nman[2])
plt.plot(hand_nman[2])
plt.grid()
plt.ylim((-2,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT RESNET-50
    
fig = plt.figure()
fig.suptitle("RESNET-50", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[3])
plt.plot(face_objects[3])
plt.plot(hand_objects[3])
plt.grid()
plt.ylim((-1.9,1.3))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[3])
plt.plot(face_tool[3])
plt.plot(hand_tool[3])
plt.grid()
plt.ylim((-1.9,1.3))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[3])
plt.plot(face_mani[3])
plt.plot(hand_mani[3])
plt.grid()
plt.ylim((-1.9,1.3))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[3])
plt.plot(face_nman[3])
plt.plot(hand_nman[3])
plt.grid()
plt.ylim((-1.9,1.3))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT RESNET-101
    
fig = plt.figure()
fig.suptitle("RESNET-101", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[4])
plt.plot(face_objects[4])
plt.plot(hand_objects[4])
plt.grid()
plt.ylim((-2.1,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[4])
plt.plot(face_tool[4])
plt.plot(hand_tool[4])
plt.grid()
plt.ylim((-2.1,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[4])
plt.plot(face_mani[4])
plt.plot(hand_mani[4])
plt.grid()
plt.ylim((-2.1,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[4])
plt.plot(face_nman[4])
plt.plot(hand_nman[4])
plt.grid()
plt.ylim((-2.1,1.1))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT INCEPTION-RESNET-V2
    
fig = plt.figure()
fig.suptitle("INCEPTION-RESNET-V2", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[5])
plt.plot(face_objects[5])
plt.plot(hand_objects[5])
plt.grid()
plt.ylim((-2.4,1.2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[5])
plt.plot(face_tool[5])
plt.plot(hand_tool[5])
plt.grid()
plt.ylim((-2.4,1.2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[5])
plt.plot(face_mani[5])
plt.plot(hand_mani[5])
plt.grid()
plt.ylim((-2.4,1.2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[5])
plt.plot(face_nman[5])
plt.plot(hand_nman[5])
plt.grid()
plt.ylim((-2.4,1.2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT DENSENET-201
    
fig = plt.figure()
fig.suptitle("DENSENET-201", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[6])
plt.plot(face_objects[6])
plt.plot(hand_objects[6])
plt.grid()
plt.ylim((-2.3,1.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[6])
plt.plot(face_tool[6])
plt.plot(hand_tool[6])
plt.grid()
plt.ylim((-2.3,1.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[6])
plt.plot(face_mani[6])
plt.plot(hand_mani[6])
plt.grid()
plt.ylim((-2.3,1.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[6])
plt.plot(face_nman[6])
plt.plot(hand_nman[6])
plt.grid()
plt.ylim((-2.3,1.5))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% ORIGINAL IMAGES

# ALEX
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Original images")
alexnet = scipy.io.loadmat("cos_small_original")["cos_small"]

# VGG19
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Original images")
vgg19 = scipy.io.loadmat("cos_small_original")["cos_small"]

#INCEPTIONV3
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionV3\Original images")
inceptionv3 = scipy.io.loadmat("cos_small_original")["cos_small"]

#ResNet-50
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet50\Original images\Conv small, different averaging, original images")
resnet50 = scipy.io.loadmat("cos_small_original")["cos_small"]

#ResNet-101
os.chdir(r"D:\thesis-scripts\Neural networks\ResNet101\Original images\Conv small, different averaging, original images")
resnet101 = scipy.io.loadmat("cos_small_original")["cos_small_original"]

#InceptionResNetV2
os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2\Original images\Conv small, different averaging, original images")
inception_resnetv2 = scipy.io.loadmat("cos_small_original")["cos_small"]

#DenseNet-201
os.chdir(r"D:\thesis-scripts\Neural networks\Densenet201\Original images\Conv small, different averaging, original images")
densenet201 = scipy.io.loadmat("cos_small")["cos_small"]

os.chdir(r"D:\thesis-scripts\Neural networks")

#%% PLOT ALEXNET
    
fig = plt.figure()
fig.suptitle("ALEXNET | ORIGINAL IMAGES ", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[0])
plt.plot(face_objects[0])
plt.plot(hand_objects[0])
plt.grid()
plt.ylim((-1.8, 1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[0])
plt.plot(face_tool[0])
plt.plot(hand_tool[0])
plt.grid()
plt.ylim((-1.8, 1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[0])
plt.plot(face_mani[0])
plt.plot(hand_mani[0])
plt.grid()
plt.ylim((-1.8, 1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[0])
plt.plot(face_nman[0])
plt.plot(hand_nman[0])
plt.grid()
plt.ylim((-1.8, 1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,8)), [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
        ], rotation=0)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% VGG-19

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

fig = plt.figure()
fig.suptitle("VGG-19 | ORIGINAL IMAGES", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[1])
plt.plot(face_objects[1])
plt.plot(hand_objects[1])
plt.grid()
plt.ylim((-1.9,1.9))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[1])
plt.plot(face_tool[1])
plt.plot(hand_tool[1])
plt.grid()
plt.ylim((-1.9,1.9))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[1])
plt.plot(face_mani[1])
plt.plot(hand_mani[1])
plt.grid()
plt.ylim((-1.9,1.9))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[1])
plt.plot(face_nman[1])
plt.plot(hand_nman[1])
plt.grid()
plt.ylim((-1.9,1.9))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,19)), layers, rotation=45, size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% INCEPTION-V3

fig = plt.figure()
fig.suptitle("INCEPTION-V3 | ORIGINAL IMAGES", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[2])
plt.plot(face_objects[2])
plt.plot(hand_objects[2])
plt.grid()
plt.ylim((-1.9,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[2])
plt.plot(face_tool[2])
plt.plot(hand_tool[2])
plt.grid()
plt.ylim((-1.9,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[2])
plt.plot(face_mani[2])
plt.plot(hand_mani[2])
plt.grid()
plt.ylim((-1.9,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[2])
plt.plot(face_nman[2])
plt.plot(hand_nman[2])
plt.grid()
plt.ylim((-1.9,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT RESNET-50
    
fig = plt.figure()
fig.suptitle("RESNET-50 | ORIGINAL IMAGES", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[3])
plt.plot(face_objects[3])
plt.plot(hand_objects[3])
plt.grid()
plt.ylim((-1.6,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[3])
plt.plot(face_tool[3])
plt.plot(hand_tool[3])
plt.grid()
plt.ylim((-1.6,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[3])
plt.plot(face_mani[3])
plt.plot(hand_mani[3])
plt.grid()
plt.ylim((-1.6,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[3])
plt.plot(face_nman[3])
plt.plot(hand_nman[3])
plt.grid()
plt.ylim((-1.6,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(range(0,50)), list(range(1,51)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT RESNET-101
    
fig = plt.figure()
fig.suptitle("RESNET-101 | ORIGINAL IMAGES", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[4])
plt.plot(face_objects[4])
plt.plot(hand_objects[4])
plt.grid()
plt.ylim((-1.7,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[4])
plt.plot(face_tool[4])
plt.plot(hand_tool[4])
plt.grid()
plt.ylim((-1.7,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[4])
plt.plot(face_mani[4])
plt.plot(hand_mani[4])
plt.grid()
plt.ylim((-1.7,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[4])
plt.plot(face_nman[4])
plt.plot(hand_nman[4])
plt.grid()
plt.ylim((-1.7,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,101,4)), list(np.arange(1,102,4)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT INCEPTION-RESNET-V2
    
fig = plt.figure()
fig.suptitle("INCEPTION-RESNET-V2 | ORIGINAL IMAGES", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[5])
plt.plot(face_objects[5])
plt.plot(hand_objects[5])
plt.grid()
plt.ylim((-2.4,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[5])
plt.plot(face_tool[5])
plt.plot(hand_tool[5])
plt.grid()
plt.ylim((-2.4,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[5])
plt.plot(face_mani[5])
plt.plot(hand_mani[5])
plt.grid()
plt.ylim((-2.4,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[5])
plt.plot(face_nman[5])
plt.plot(hand_nman[5])
plt.grid()
plt.ylim((-2.4,2))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,176,7)), list(np.arange(1,177,7)), size=6)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% PLOT DENSENET-201
    
fig = plt.figure()
fig.suptitle("DENSENET-201 | ORIGINAL IMAGES", color='red')
# 1
plt.subplot(2,2,1)
plt.plot(body_objects[6])
plt.plot(face_objects[6])
plt.plot(hand_objects[6])
plt.grid()
plt.ylim((-1.9,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
plt.ylabel("Correlation")
plt.title("Body parts and OBJECTS\n (mean of tools + manipulable + nonmanipulable)")
plt.legend(("bodies", "hands", "faces"), loc='lower left')    
       
plt.subplot(2,2,2)
plt.plot(body_tool[6])
plt.plot(face_tool[6])
plt.plot(hand_tool[6])
plt.grid()
plt.ylim((-1.9,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and TOOLS")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,3)
plt.plot(body_mani[6])
plt.plot(face_mani[6])
plt.plot(hand_mani[6])
plt.grid()
plt.ylim((-1.9,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and MANI")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

plt.subplot(2,2,4)
plt.plot(body_nman[6])
plt.plot(face_nman[6])
plt.plot(hand_nman[6])
plt.grid()
plt.ylim((-1.9,1.8))
# plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xticks(list(np.arange(0,201,5)), list(np.arange(1,202,5)), size=4)
# plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts and NMAN")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()