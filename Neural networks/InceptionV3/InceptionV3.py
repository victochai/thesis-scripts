#%% Libraries

import os
import re
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import glob
import scipy.io
import matplotlib.pyplot as plt

#%% Images

# arrangement: 
# 1. body
# 2. hand
# 3. face
# 4. tool
# 5. man
# 6. nman
# 7. hair

# Arrange the images in a specific order
img_path = Path('images')
body = [name for name in glob.glob(str(img_path/'*body*'))]
hand = [name for name in glob.glob(str(img_path/'*hand*'))]
face = [name for name in glob.glob(str(img_path/'*face*'))]
tool = [name for name in glob.glob(str(img_path/'*tool*'))]
man = [name for name in glob.glob(str(img_path/'*Mani*'))]
nman = [name for name in glob.glob(str(img_path/'*NMan*'))]
hair = [name for name in glob.glob(str(img_path/'*hair*'))]
n_samples = len(os.listdir(img_path)) # or m
filenames = body + hand + face + tool + man + nman + hair
del body, hand, face, tool, man, nman, hair

# Load the images 
(height, width, nchannels) = (299, 299, 3)   # for IneptionV3
images = np.zeros((n_samples, height, width, nchannels))

for image in range(n_samples):
    images[image, :, :, :] = img_to_array(load_img(filenames[image], target_size = (height, width)))
preprocess_input(images)

del height, width, image, nchannels, filenames
!cls

#import scipy.io
#imagesV3 = {'imagesV3' : images}
#scipy.io.savemat('imagesV3.mat', imagesV3)

#%% InceptionV3

pre_trained_model = InceptionV3(input_shape = (299, 299, 3), include_top = True, weights = 'imagenet')
pre_trained_model.summary()
# plot_model(pre_trained_model, to_file = 'ICNEPTION.png', show_shapes = True, show_layer_names = True)
layer_names = [layer.name for layer in pre_trained_model.layers]
layer_outputs = [layer.output for layer in pre_trained_model.layers]

# InceptionV3 has 48 layers
# Only CONV layers are needed if we want fairer comparison with the brain representations
# Here I use regular expressions to select only CONV layers
myRegex = re.compile(r'''(\bconv2d\b|\bconv2d_\d\b|\bconv2d_\d{2}\b)+?''')
layers = myRegex.findall(str(layer_names))

#%% Output tensors

# Get layer outputs only for layers we are interested in (94 layers = 94 tensors):
# Get indecies of layers we care about
layer_names = [layer.name for layer in pre_trained_model.layers]
layer_outputs = [layer.output for layer in pre_trained_model.layers]
idcs = []
for l in layers:
    idcs.append(np.where(np.array(layer_names) == l)[0])
idcs = list(np.array(idcs).reshape(94,))

# Get tensors we care about
l_outputs = []
for idx in idcs:
    l_outputs.append(layer_outputs[idx])
del idcs, idx, l, layer_outputs, layer_names  

#%% ##### FEATURE MAPS #####
# On CPU --> memory errors. Therefore do everything partially

# InceptionV3 architecture (CONV layers percpective)
# 5 COVN layers
# 3 A blocks
# 1st normalization layer
# 4 B blocks
# 2nd normalization layer
# 2 C bocks
# Average Pool
# Softmax

# Nice architecture drawing is here: 
# http://josephpcohen.com/w/visualizing-cnn-architectures-side-by-side-with-mxnet/

#%% First 5 layers

l_outputs = l_outputs[0:5]
model = Model(pre_trained_model.input, l_outputs)
feature_maps = model.predict(images)
for _ in range(0,5): 
    feature_maps[_] = feature_maps[_].reshape((n_samples,-1))
    print(feature_maps[_].shape)
    
#%% Module A (3 blocks, 3 layers)

print(layers[5:12])
module_A_1 = l_outputs[5:12] # 3 layers here
print(layers[12:19])
module_A_2 = l_outputs[12:19] 
print(layers[19:26])
module_A_3 = l_outputs[19:26] 

model = Model(pre_trained_model.input, module_A_3) # Change this
feature_maps = model.predict(images)
feature_maps = [
     feature_maps[0], 
     np.concatenate((feature_maps[1], feature_maps[2]), axis=3),
     np.concatenate((
             feature_maps[3],
             feature_maps[4],
             feature_maps[5],
             feature_maps[6]
             ), axis=3)
     ]

for _ in range(0,3): 
    feature_maps[_] = feature_maps[_].reshape((n_samples,-1))
    print(feature_maps[_].shape)

#%% 1st normalization (1 block, 3 layers)
    
# del module_A_1, module_A_2, module_A_3
print(layers[26:30]) # 3 Layers here
norm_after_A = l_outputs[26:30]
model = Model(pre_trained_model.input, norm_after_A) # Change thi
feature_maps = model.predict(images)
# Concatenate some layers: the result shoult be 3 layers
feature_maps = [
     feature_maps[0],
     feature_maps[1],
     np.concatenate((
             feature_maps[2],
             feature_maps[3]
             ), axis=3)
     ]

for _ in range(0,3):
    feature_maps[_] = feature_maps[_].reshape((n_samples,-1))
    print(feature_maps[_].shape)

#%% Module B (4 blocks, 5 layers)
    
del norm_after_A
print(layers[30:40]) # 5 Layers here
module_B_1 = l_outputs[30:40]
module_B_2 = l_outputs[40:50]
module_B_3 = l_outputs[50:60]
module_B_4 = l_outputs[60:70]

model = Model(pre_trained_model.input, module_B_4) 
feature_maps = model.predict(images)
feature_maps = [
     feature_maps[0],
     feature_maps[1],
     np.concatenate((
             feature_maps[2],
             feature_maps[3]
             ), axis=3),
     np.concatenate((
             feature_maps[4],
             feature_maps[5]
             ), axis=3),
     np.concatenate((
             feature_maps[6],
             feature_maps[7],
             feature_maps[8],
             feature_maps[9]
             ), axis=3),
     ]

for _ in range(0,5): 
    feature_maps[_] = feature_maps[_].reshape((n_samples,-1))
    print(feature_maps[_].shape)

#%% 2nd normalization (1 block, 4 layers)
    
# del module_B_2, module_B_3, module_B_4, module_B_1
print(layers[70:76]) # 4 layers here
norm_after_B = l_outputs[70:76]
model = Model(pre_trained_model.input, norm_after_B) 
feature_maps = model.predict(images)
feature_maps = [
     feature_maps[0],
     feature_maps[1],
     np.concatenate((
             feature_maps[2],
             feature_maps[3]
             ), axis=3),
     np.concatenate((
             feature_maps[4],
             feature_maps[5]
             ), axis=3)
     ]

for _ in range(0,4):
    feature_maps[_] = feature_maps[_].reshape((n_samples,-1))
    print(feature_maps[_].shape)
    
#%% Module C (2 blocks, 3 layers)

del norm_after_B
print(layers[76:85]) # 3 Layers here
module_C_1 = l_outputs[76:85]   
print(layers[85:]) 
module_C_2 = l_outputs[85:]
 
model = Model(pre_trained_model.input, module_C_2) # Change this
feature_maps = model.predict(images)
feature_maps = [
     feature_maps[0],
     np.concatenate((
             feature_maps[1],
             feature_maps[2]
             ), axis=3),
     np.concatenate((
             feature_maps[3],
             feature_maps[4],
             feature_maps[5],
             feature_maps[6],
             feature_maps[7],
             feature_maps[8]
             ), axis=3)
     ]

for _ in range(0,3): 
    feature_maps[_] = feature_maps[_].reshape((n_samples,-1))
    print(feature_maps[_].shape) 

#%% LAST LAYER: average pool
    
feature_maps = []
# del module_C_1, module_C_2, l_outputs, layers
output = pre_trained_model.get_layer('avg_pool').output
model = Model(pre_trained_model.input, output)
feature_maps.append(model.predict(images))
_ = 0
print(feature_maps[_].shape) 

#%% Rearrange results + Correlations

# Rearrangement    
bodies = []
hands = []
faces = []
tools = []
man = []
nman = []
objects = []
chairs = []

# Correlations
corr_bodies_objects = np.zeros((48,4))
corr_hands_objects = np.zeros((48,4))
corr_faces_objects = np.zeros((48,4))
# Shape = (48 correlations for each layer, body parts vs. objects,
#                                           body parts vs. tools,
#                                           body parts vs. man,
#                                           body parts vs. nman)

k = 0 # Change this

for _ in range(0,1): 
    
    bodies.append(np.mean(feature_maps[_][0:48, :], 0))
    hands.append(np.mean(feature_maps[_][48:96, :], 0))
    faces.append(np.mean(feature_maps[_][96:144, :], 0))
    tools.append(np.mean(feature_maps[_][144:192, :], 0))
    man.append(np.mean(feature_maps[_][192:240, :], 0))
    nman.append(np.mean(feature_maps[_][240:288, :], 0))
    chairs.append(np.mean(feature_maps[_][288:336, :], 0))
    objects.append(np.squeeze(np.mean(np.array([tools[_], man[_], nman[_]]), 0)))
    
    # Objects correlations
    corr_bodies_objects[_+k, 0] = np.corrcoef(bodies[_], objects[_])[0,1]
    corr_hands_objects[_+k, 0] = np.corrcoef(hands[_], objects[_])[0,1]
    corr_faces_objects[_+k, 0] = np.corrcoef(faces[_], objects[_])[0,1]
    
    # Tools
    corr_bodies_objects[_+k, 1] = np.corrcoef(bodies[_], tools[_])[0,1]
    corr_hands_objects[_+k, 1] = np.corrcoef(hands[_], tools[_])[0,1]
    corr_faces_objects[_+k, 1] = np.corrcoef(faces[_], tools[_])[0,1]
    
    # Manipulable objects
    corr_bodies_objects[_+k, 2] = np.corrcoef(bodies[_], man[_])[0,1]
    corr_hands_objects[_+k, 2] = np.corrcoef(hands[_], man[_])[0,1]
    corr_faces_objects[_+k, 2] = np.corrcoef(faces[_], man[_])[0,1]
    
    # Non-manipulable objects
    corr_bodies_objects[_+k, 3] = np.corrcoef(bodies[_], nman[_])[0,1]
    corr_hands_objects[_+k, 3] = np.corrcoef(hands[_], nman[_])[0,1]
    corr_faces_objects[_+k, 3] = np.corrcoef(faces[_], nman[_])[0,1]

del bodies, hands, faces, tools, man, nman, chairs, feature_maps, objects
# del l_outputs

#%% Saving the results

# For Matlab
corr_bodies = {'corr_bodies_objects' : corr_bodies_objects}
corr_faces = {'corr_faces_objects' : corr_faces_objects}
corr_hands = {'corr_hands_objects' : corr_hands_objects}
scipy.io.savemat('corr_bodies.mat', corr_bodies)
scipy.io.savemat('corr_faces.mat', corr_faces)
scipy.io.savemat('corr_hands.mat', corr_hands)

# For Python
np.save('corr_bodies_objects.npy', corr_bodies_objects)
np.save('corr_faces_objects.npy', corr_faces_objects)
np.save('corr_hands_objects.npy', corr_hands_objects)

#%% Visualizing the results

#%% Loading

import numpy as np
import matplotlib.pyplot as plt
corr_bodies_objects = np.load("corr_bodies_objects.npy")
corr_hands_objects = np.load("corr_hands_objects.npy")
corr_faces_objects = np.load("corr_faces_objects.npy")

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
layers.append("pred")

del a, b, c, norm1, norm2

#%% Plotting

# Objects
plt.plot(corr_bodies_objects[:,0],'-o')
plt.plot(corr_hands_objects[:,0],'-o')
plt.plot(corr_faces_objects[:,0],'-o')
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
plt.plot(corr_bodies_objects[:,1],'-o')
plt.plot(corr_hands_objects[:,1],'-o')
plt.plot(corr_faces_objects[:,1],'-o')
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
plt.plot(corr_bodies_objects[:,2],'-o')
plt.plot(corr_hands_objects[:,2],'-o')
plt.plot(corr_faces_objects[:,2],'-o')
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
plt.plot(corr_bodies_objects[:,3],'-o')
plt.plot(corr_hands_objects[:,3],'-o')
plt.plot(corr_faces_objects[:,3],'-o')
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
fig.suptitle("INCEPTION V3")
# 1
plt.subplot(2,2,1)
plt.plot(corr_bodies_objects[:,0],'-o')
plt.plot(corr_hands_objects[:,0],'-o')
plt.plot(corr_faces_objects[:,0],'-o')
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
plt.plot(corr_bodies_objects[:,1],'-o')
plt.plot(corr_hands_objects[:,1],'-o')
plt.plot(corr_faces_objects[:,1],'-o')
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
plt.plot(corr_bodies_objects[:,2],'-o')
plt.plot(corr_hands_objects[:,2],'-o')
plt.plot(corr_faces_objects[:,2],'-o')
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
plt.plot(corr_bodies_objects[:,3],'-o')
plt.plot(corr_hands_objects[:,3],'-o')
plt.plot(corr_faces_objects[:,3],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,48)), list(range(1,49)), fontsize=6)
# plt.xticks(list(range(0,48)), layers, rotation=80, fontsize=8)
plt.ylabel("Correlation")
plt.title("Correlation between different body parts and nonmanipulable objects in InceptionV3 layers")
plt.legend(("bodies", "hands", "faces"), loc='lower left')
plt.show()

#%% ##### Confusion matrices for another analysis #####

#%% Big matrices
    
# corr_matrices = []
for _ in range(0, 1): # Change this
    corr_matrices.append(np.corrcoef(feature_maps[_]))

#%% Small matrices
    
feature_maps_small = []
for _ in range(0, 1): # Change this
    bodies = np.mean(feature_maps[_][0:48, :], 0).reshape((feature_maps[_].shape[1],1))
    hands = np.mean(feature_maps[_][48:96, :], 0).reshape((feature_maps[_].shape[1],1))
    faces = np.mean(feature_maps[_][96:144, :], 0).reshape((feature_maps[_].shape[1],1))
    tools = np.mean(feature_maps[_][144:192, :], 0).reshape((feature_maps[_].shape[1],1))
    man = np.mean(feature_maps[_][192:240, :], 0).reshape((feature_maps[_].shape[1],1))
    nman = np.mean(feature_maps[_][240:288, :], 0).reshape((feature_maps[_].shape[1],1))
    chairs = np.mean(feature_maps[_][288:336, :], 0).reshape((feature_maps[_].shape[1],1))
    feature_maps_small.append(np.concatenate(
            (bodies, hands, faces, tools, man, nman, chairs), axis=1
            ))
    del bodies, hands, faces, tools, man, nman, chairs
    
corr_matrices_small = [] 
for _ in range(0, 1): # Change this
    corr_matrices_small.append(np.corrcoef(feature_maps_small[_].T))

#%% Del stuff
    
del feature_maps, feature_maps_small
# del l_outputs

#%% Save the results

# For Matlab
corr_matrices_MATLAB = {'corr_matrices' : corr_matrices}
corr_matrices_small_MATLAB = {'corr_matrices_small' : corr_matrices_small}
scipy.io.savemat('corr_matrices_MATLAB.mat', corr_matrices_MATLAB)
scipy.io.savemat('corr_matrices_small_MATLAB.mat', corr_matrices_small_MATLAB)

# For Python
import pickle
with open('corr_M', 'wb') as f:
    pickle.dump(corr_matrices, f)
with open('corr_M_small', 'wb') as f:
    pickle.dump(corr_matrices_small, f)
    
#%% Visualize the results

import pickle
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

with open('corr_M', 'rb') as handle:
    corr_matrices = pickle.load(handle)
co = scipy.io.loadmat("co.mat")    
co = co["co"]
del corr_matrices[47]
corr_matrices.append(co)

    
# Big matrices
fig = plt.figure()
fig.suptitle("InceptionV3\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(corr_matrices)):
    plt.subplot(6,8,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(corr_matrices[_])
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=6)
plt.show()    

with open('corr_M_small', 'rb') as handle:
    corr_matrices_small = pickle.load(handle)
co_small = scipy.io.loadmat("co_small.mat")    
co_small = co_small["co_small"]
del corr_matrices_small[47]
corr_matrices_small.append(co_small)
    
# Small matrices
fig = plt.figure()
fig.suptitle("InceptionV3\nEvery condition is averaged\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(corr_matrices_small)):
    plt.subplot(6,8,_+1)
#    plt.imshow(corr_matrices_small[_],cmap="cividis")
    plt.imshow(corr_matrices_small[_])    
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=6)
plt.show()        

#%% Checking the last layer

feature_maps = []
output = pre_trained_model.get_layer('mixed10').output
model = Model(pre_trained_model.input, output)
feature_maps.append(model.predict(images))
_ = 0
print(feature_maps[_].shape) 
feature_maps[_] = feature_maps[_].reshape((336,-1))
print(feature_maps[_].shape) 
co = np.corrcoef(feature_maps[_])
plt.imshow(co)
plt.show()

feature_maps_small = []
for _ in range(0, 1): # Change this
    bodies = np.mean(feature_maps[_][0:48, :], 0).reshape((feature_maps[_].shape[1],1))
    hands = np.mean(feature_maps[_][48:96, :], 0).reshape((feature_maps[_].shape[1],1))
    faces = np.mean(feature_maps[_][96:144, :], 0).reshape((feature_maps[_].shape[1],1))
    tools = np.mean(feature_maps[_][144:192, :], 0).reshape((feature_maps[_].shape[1],1))
    man = np.mean(feature_maps[_][192:240, :], 0).reshape((feature_maps[_].shape[1],1))
    nman = np.mean(feature_maps[_][240:288, :], 0).reshape((feature_maps[_].shape[1],1))
    chairs = np.mean(feature_maps[_][288:336, :], 0).reshape((feature_maps[_].shape[1],1))
    feature_maps_small.append(np.concatenate(
            (bodies, hands, faces, tools, man, nman, chairs), axis=1
            ))
    del bodies, hands, faces, tools, man, nman, chairs
    
co = np.corrcoef(feature_maps_small[_].T)
plt.imshow(co)
plt.show()

#%% Plot everything

bodies = np.load("corr_bodies_objects.npy")
faces = np.load("corr_faces_objects.npy") 
hands = np.load("corr_hands_objects.npy")
