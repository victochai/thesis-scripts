#%% Modules

import os
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#%% Images

img_dir = r"D:\THESIS\input_new"
os.chdir(img_dir)
body = [name for name in glob.glob('*body*')]
hand = [name for name in glob.glob('*hand*')]
face = [name for name in glob.glob('*face*')]
tool = [name for name in glob.glob('*tool*')]
man = [name for name in glob.glob('*Mani*')]
nman = [name for name in glob.glob('*NMan*')]
chair = [name for name in glob.glob('*Chair*')]
m = len(os.listdir(img_dir))
filenames = body + hand + face + tool + man + nman + chair
del body, hand, face, tool, man, nman, chair

images = [plt.imread(image) for image in filenames]
images_bw = []
for image in images:
    if image.shape == (400, 400, 3):
        images_bw.append(image[:,:,0])
    elif image.shape == (400, 400):
        images_bw.append(image)
    else:
        print("Unexpected shape.")
    
os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\images")
images_original = {'images_original' : images_bw}
scipy.io.savemat('images_original.mat', images_original)

#%% Images + Mirror

img_dir = r"D:\THESIS\input_new"
os.chdir(img_dir)
body = [name for name in glob.glob('*body*')]
hand = [name for name in glob.glob('*hand*')]
face = [name for name in glob.glob('*face*')]
tool = [name for name in glob.glob('*tool*')]
man = [name for name in glob.glob('*Mani*')]
nman = [name for name in glob.glob('*NMan*')]
chair = [name for name in glob.glob('*Chair*')]
m = len(os.listdir(img_dir))
filenames = body + hand + face + tool + man + nman + chair
del body, hand, face, tool, man, nman, chair

images = [plt.imread(image) for image in filenames]
images_bw = []
for image in images:
    if image.shape == (400, 400, 3):
        images_bw.append(image[:,:,0])
        images_bw.append(np.fliplr(image[:,:,0]))
    elif image.shape == (400, 400):
        images_bw.append(image)
        images_bw.append(np.fliplr(image))
    else:
        print("Unexpected shape.")

plt.imshow(images_bw[0])
plt.imshow(images_bw[1])

os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\images")
images_forDNN = {'images_forDNN' : images}
scipy.io.savemat('images_forDNN.mat', images_forDNN)

#%% Save for DNNs

n_samples = 336
(height, width, nchannels) = (400, 400, 3)
images = np.zeros((n_samples, height, width, nchannels))

for idx, image in enumerate(images_bw):
   images[idx,:,:,:] = np.concatenate((image.reshape((400,400,1)), image.reshape((400,400,1)), image.reshape(400,400,1)), 2)

#%% Visualization stuff

os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations")
mat = scipy.io.loadmat("images_correlations_for_experiment.mat")["correlations"]
body = mat[0][0]
hand = mat[1][0]
face = mat[2][0]
co = mat[3][0]
co_small = mat[4][0]

#%% Plot all

n_samples = co.shape[0]

plt.subplot(2,3,1)
plt.bar([0, 1, 2], [body[1][0], hand[1][0], face[1][0]], color=('royalblue', 'orange', 'green'))
plt.title("Correlation between different body parts\n and tools in " + str(n_samples) + " images used in the experiment")
plt.ylabel("Correlation")
plt.xlabel("Body part")
plt.ylim((0,1))
plt.xticks([0,1,2], [
        "body | tool", "hand | tool", "face | tool"
        ], fontsize=6, rotation=0)

plt.subplot(2,3,2)
plt.bar([0, 1, 2], [body[2][0], hand[2][0], face[2][0]], color=('royalblue', 'orange', 'green'))
plt.title("Correlation between different body parts\n and manipulable objects in " + str(n_samples) + " images used\nin the experiment")
plt.ylabel("Correlation")
plt.xlabel("Body part")
plt.ylim((0,1))
plt.xticks([0,1,2], [
        "body | man", "hand | man", "face | man"
        ], fontsize=6, rotation=0)

plt.subplot(2,3,3)
plt.bar([0, 1, 2], [body[3][0], hand[3][0], face[3][0]], color=('royalblue', 'orange', 'green'))
plt.title("Correlation between different body parts\n and nonmanipulable objects in " + str(n_samples) + " images used\nin the experiment")
plt.ylabel("Correlation")
plt.xlabel("Body part")
plt.ylim((0,1))
plt.xticks([0,1,2], [
        "body | nman", "hand | nman", "face | nnman"
        ], fontsize=6, rotation=0)

plt.subplot(2,3,4)
plt.imshow(co)
plt.colorbar()
plt.title("Images used in the experiment correlations")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(co_small)
plt.colorbar()
plt.title("Images used in the experiment  correlations (averaged)")
plt.axis("off")

plt.show()

#%% Visualize new images ALEXNET | SMALL

os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\Brain\Images")
mat = scipy.io.loadmat("coS_small_INC.mat")["coS_small"]
conv_matrices = [mat[0][idx] for idx in range(48)]

fig = plt.figure()
fig.suptitle("INCEPTIONV3 WITH ORIGINAL IMAGES\nEvery condition is averaged\nBodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(conv_matrices)):
    plt.subplot(6,8,_+1)
    plt.imshow(conv_matrices[_])
#    plt.imshow(conv_matrices[_],cmap="cividis")
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

#%% Visualize new images ALEXNET | BIG

os.chdir(r"C:\Users\victo\Desktop\thesis-scripts\Brain\Images")
mat = scipy.io.loadmat("cos_small_VGG.mat")["cos_small"]
conv_matrices = [mat[0][idx] for idx in range(len(mat[0]))]

fig = plt.figure()
fig.suptitle("VGG19 WITH ORIGINAL IMAGES\nEvery condition is averaged\nBodies, faces, hands, tools, manipulable objects, nonmanipulable objects, chairs")
for _ in range(0, len(conv_matrices)):
    plt.subplot(3,7,_+1)
    plt.imshow(conv_matrices[_])
#    plt.imshow(conv_matrices[_],cmap="cividis")
    plt.colorbar()
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  
