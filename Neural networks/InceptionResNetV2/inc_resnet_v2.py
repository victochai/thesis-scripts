#%% Modules

import os
import re
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import scipy.io
import matplotlib.pyplot as plt


#%% InceptionResnetV2

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

#%% Model

pre_trained_model = InceptionResNetV2(input_shape = (299, 299, 3), include_top = True, weights = 'imagenet')
pre_trained_model.summary()
# plot_model(pre_trained_model, to_file = 'ICNEPTION.png', show_shapes = True, show_layer_names = True)
layer_names = [layer.name for layer in pre_trained_model.layers]
layer_outputs = [layer.output for layer in pre_trained_model.layers]

#%% Loading .Mat files

os.chdir(r"D:\inception_resnet_v2\co\stem")
l1 = os.listdir()

os.chdir(r"D:\inception_resnet_v2\co\first 100")
l2 = os.listdir()

os.chdir(r"D:\inception_resnet_v2\co\last 100")
l3 = os.listdir()

os.chdir(r"D:\inception_resnet_v2\co\last 2")
l4 = os.listdir()

l = l1+l2+l3+l4
del l1, l2, l3, l4

os.chdir(r"D:\inception_resnet_v2\co")
cos = []
for _ in l:
    cos.append(scipy.io.loadmat(_)["co"])

#%% Plot big matrices

os.chdir(r"D:\thesis-scripts\Neural networks\InceptionResNetV2")

fig = plt.figure()
fig.suptitle("InceptionResNetV2\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(l)):
    plt.subplot(11,16,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos[_])
#    plt.colorbar()
    plt.axis("off")
#    plt.title(str(_+1), fontsize=9)
plt.show()  

#%% Make small matrices

import numpy as np

cos_small = []
for co in cos:
    x_ind = -48
    y_ind = -48
    small = np.zeros((7, 7))
    for x in range(0, 7):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 7):
            y_ind += 48
            small[x, y] = np.mean(co[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    cos_small.append(small)
            
#%% Plot small matrices
    
fig = plt.figure()
fig.suptitle("InceptionResNetV2\nEvery condition is averaged\n Bodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(l)):
    plt.subplot(11,16,_+1)
#    plt.imshow(corr_matrices[_],cmap="cividis")
    plt.imshow(cos_small[_])
#    plt.colorbar()
    plt.axis("off")
#    plt.title(str(_+1), fontsize=9)
plt.show()          
               