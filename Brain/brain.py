#%% Import modules

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#%% Brain representations + Nets

# Brain
work_dir = r"D:\thesis-scripts\Brain\Brain representations"
os.chdir(work_dir)

ant = scipy.io.loadmat("anterior_big_MATRIX.mat")["anterior_big_MATRIX"]
ant_left = scipy.io.loadmat("anterior_left.mat")["anterior_left"]
ant_right = scipy.io.loadmat("anterior_right.mat")["anterior_right"]
new = scipy.io.loadmat("new.mat")["new"]

op = scipy.io.loadmat("OP_MATRIX.mat")["OP_MATRIX"]
calc = scipy.io.loadmat("CALC_MATRIX.mat")["CALC_MATRIX"]
op_calc = scipy.io.loadmat("OP_CALC_MATRIX.mat")["OP_CALC_MATRIX"]
pos = scipy.io.loadmat("pos_res2_mvpa_MATRIX.mat")["pos_res2_mvpa"]

#%% Correlation between body parts and objects in the brain

brain_regions = [
        "calc. cortex", 
        "occip. pole",
        "post. IOG",
        "ITG + ant. IOG (left)"
        ]

os.chdir(r"D:\thesis-scripts\Brain\Brain representations")

body = scipy.io.loadmat("body_left.mat")["body"]
hand = scipy.io.loadmat("hand_left.mat")["hand"]
face = scipy.io.loadmat("face_left.mat")["face"]

body_right = scipy.io.loadmat("body_right.mat")["body"]
hand_right = scipy.io.loadmat("hand_right.mat")["hand"]
face_right = scipy.io.loadmat("face_right.mat")["face"]

body_center = scipy.io.loadmat("body.mat")["body"]
hand_center = scipy.io.loadmat("hand.mat")["hand"]
face_center = scipy.io.loadmat("face.mat")["face"]

# Horizontal:
objects = 0
tool = 1
mani = 2
nman = 3

#%% OBJECTS

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND OBJECTS (tool, mani, nman) CORRELATION", color='red')

plt.subplot(2, 3, 1)
plt.bar(["body", "hand", "face"], 
        [body[objects, 0], hand[objects, 0], face[objects, 0]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Calcarine cortex")

plt.subplot(2, 3, 2)
plt.bar(["body", "hand", "face"], 
        [body[objects, 1], hand[objects, 1], face[objects, 1]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Occipital pole")

plt.subplot(2, 3, 3)
plt.bar(["body", "hand", "face"], 
        [body[objects, 2], hand[objects, 2], face[objects, 2]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Posterior IOG")

plt.subplot(2, 3, 4)
plt.bar(["body", "hand", "face"], 
        [body[objects, 3], hand[objects, 3], face[objects, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (left)")

plt.subplot(2, 3, 5)
plt.bar(["body", "hand", "face"], 
        [body_right[objects, 3], hand_right[objects, 3], face_right[objects, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (right)")

plt.subplot(2, 3, 6)
plt.bar(["body", "hand", "face"], 
        [body_center[objects, 3], hand_center[objects, 3], face_center[objects, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (BOTH HEMISPHERES)")

#%% TOOLS

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND TOOLS CORRELATION", color='red')

plt.subplot(2, 3, 1)
plt.bar(["body", "hand", "face"], 
        [body[tool, 0], hand[tool, 0], face[tool, 0]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Calcarine cortex")

plt.subplot(2, 3, 2)
plt.bar(["body", "hand", "face"], 
        [body[tool, 1], hand[tool, 1], face[tool, 1]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Occipital pole")

plt.subplot(2, 3, 3)
plt.bar(["body", "hand", "face"], 
        [body[tool, 2], hand[tool, 2], face[tool, 2]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Posterior IOG")

plt.subplot(2, 3, 4)
plt.bar(["body", "hand", "face"], 
        [body[tool, 3], hand[tool, 3], face[tool, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (left)")

plt.subplot(2, 3, 5)
plt.bar(["body", "hand", "face"], 
        [body_right[tool, 3], hand_right[tool, 3], face_right[tool, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (right)")

plt.subplot(2, 3, 6)
plt.bar(["body", "hand", "face"], 
        [body_center[tool, 3], hand_center[tool, 3], face_center[tool, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (BOTH HEMISPHERES)")

#%% MANI

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND MANI CORRELATION", color='red')

plt.subplot(2, 3, 1)
plt.bar(["body", "hand", "face"], 
        [body[mani, 0], hand[mani, 0], face[mani, 0]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Calcarine cortex")

plt.subplot(2, 3, 2)
plt.bar(["body", "hand", "face"], 
        [body[mani, 1], hand[mani, 1], face[mani, 1]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Occipital pole")

plt.subplot(2, 3, 3)
plt.bar(["body", "hand", "face"], 
        [body[mani, 2], hand[mani, 2], face[mani, 2]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Posterior IOG")

plt.subplot(2, 3, 4)
plt.bar(["body", "hand", "face"], 
        [body[mani, 3], hand[mani, 3], face[mani, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (left)")

plt.subplot(2, 3, 5)
plt.bar(["body", "hand", "face"], 
        [body_right[mani, 3], hand_right[mani, 3], face_right[mani, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (right)")

plt.subplot(2, 3, 6)
plt.bar(["body", "hand", "face"], 
        [body_center[mani, 3], hand_center[mani, 3], face_center[mani, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (BOTH HEMISPHERES)")

#%% NMAN

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND NMAN CORRELATION", color='red')

plt.subplot(2, 3, 1)
plt.bar(["body", "hand", "face"], 
        [body[nman, 0], hand[nman, 0], face[nman, 0]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Calcarine cortex")

plt.subplot(2, 3, 2)
plt.bar(["body", "hand", "face"], 
        [body[nman, 1], hand[nman, 1], face[nman, 1]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Occipital pole")

plt.subplot(2, 3, 3)
plt.bar(["body", "hand", "face"], 
        [body[nman, 2], hand[nman, 2], face[nman, 2]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("Posterior IOG")

plt.subplot(2, 3, 4)
plt.bar(["body", "hand", "face"], 
        [body[nman, 3], hand[nman, 3], face[nman, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (left)")

plt.subplot(2, 3, 5)
plt.bar(["body", "hand", "face"], 
        [body_right[nman, 3], hand_right[nman, 3], face_right[nman, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (right)")

plt.subplot(2, 3, 6)
plt.bar(["body", "hand", "face"], 
        [body_center[nman, 3], hand_center[nman, 3], face_center[nman, 3]], 
        color=('royalblue', 'orange', 'green'))
plt.ylabel("Correlation")
plt.title("ITG + ant. IOG (BOTH HEMISPHERES)")

#%% OTHER visualizations

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

#%% other

alexnet_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
vgg_layers = ['conv1_1',
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

# InceptioV3 Layers
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

#%% Visualizing decodings

work_dir = r"C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations"
os.chdir(work_dir)
conf_matrices = scipy.io.loadmat(r"decodings_all_confusion_matrix.mat")["confusion_matrix_mean"]

fig = plt.figure()
fig.suptitle("LDA decodings\nBody, hand, face, tool, man, nman, chair")

plt.subplot(2,4,1)
plt.imshow(conf_matrices[:,:,0])
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG")

plt.subplot(2,4,2)
plt.imshow(conf_matrices[:,:,1])
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (left)")

plt.subplot(2,4,3)
plt.imshow(conf_matrices[:,:,2])
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (right)")

plt.subplot(2,4,5)
plt.imshow(conf_matrices[:,:,3])
plt.colorbar()
plt.axis("off")
plt.title("Posterior IOG")

plt.subplot(2,4,6)
plt.imshow(conf_matrices[:,:,4])
plt.colorbar()
plt.axis("off")
plt.title("Calcarine cortex")

plt.subplot(2,4,7)
plt.imshow(conf_matrices[:,:,5])
plt.colorbar()
plt.axis("off")
plt.title("Occipital pole")

plt.subplot(2,4,8)
plt.imshow(conf_matrices[:,:,6])
plt.colorbar()
plt.axis("off")
plt.title("Calc. cort. + Occip. pole")

plt.show()

#%% Visualizing decodings | OBJECTS

work_dir = r"C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations"
os.chdir(work_dir)
conf_matrices = scipy.io.loadmat(r"decodings_objects_confusion_matrix.mat")["confusion_matrix_mean"]

fig = plt.figure()
fig.suptitle("LDA object decodings\nTool, man, nman\n")

plt.subplot(2,4,1)
plt.imshow(conf_matrices[:,:,0])
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG")

plt.subplot(2,4,2)
plt.imshow(conf_matrices[:,:,1])
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (left)")

plt.subplot(2,4,3)
plt.imshow(conf_matrices[:,:,2])
plt.colorbar()
plt.axis("off")
plt.title("ITG + Anterior IOG (right)")

plt.subplot(2,4,5)
plt.imshow(conf_matrices[:,:,3])
plt.colorbar()
plt.axis("off")
plt.title("Posterior IOG")

plt.subplot(2,4,6)
plt.imshow(conf_matrices[:,:,4])
plt.colorbar()
plt.axis("off")
plt.title("Calcarine cortex")

plt.subplot(2,4,7)
plt.imshow(conf_matrices[:,:,5])
plt.colorbar()
plt.axis("off")
plt.title("Occipital pole")

plt.subplot(2,4,8)
plt.imshow(conf_matrices[:,:,6])
plt.colorbar()
plt.axis("off")
plt.title("Calc. cort. + Occip. pole")

plt.show()
