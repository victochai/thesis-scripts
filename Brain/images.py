#%% Modules

import os
import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

#%% 1.) Images used in the experiment

# Just load all the filenames
img_dir = r"D:\THESIS\IMAGES EXPERIMENT, 336"
os.chdir(img_dir)
body = [name for name in glob.glob('*body*')]
hand = [name for name in glob.glob('*hand*')]
face = [name for name in glob.glob('*face*')]
tool = [name for name in glob.glob('*tool*')]
man = [name for name in glob.glob('*Mani*')]
nman = [name for name in glob.glob('*NMan*')]
chair = [name for name in glob.glob('*Chair*')]
filenames = body + hand + face + tool + man + nman + chair
del body, hand, face, tool, man, nman, chair

# Rearrange them all
# 1. body
ro = re.compile("SHINEd_body_\d.bmp|SHINEd_body_\d_flipped.bmp")
body1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_body_\d\d.bmp|SHINEd_body_\d\d_flipped.bmp")
body2 = ro.findall(str(filenames))
body = body1+body2
del body1, body2

# 2. hand
ro = re.compile("SHINEd_hand_\d.bmp|SHINEd_hand_\d_flipped.bmp")
hand1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_hand_\d\d.bmp|SHINEd_hand_\d\d_flipped.bmp")
hand2 = ro.findall(str(filenames))
hand = hand1+hand2
del hand1, hand2

# 3. face
ro = re.compile("SHINEd_face_\d.bmp|SHINEd_face_\d_flipped.bmp")
face1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_face_\d\d.bmp|SHINEd_face_\d\d_flipped.bmp")
face2 = ro.findall(str(filenames))
face = face1+face2
del face1, face2

# 4. tool
ro = re.compile("SHINEd_tool_\d.bmp|SHINEd_tool_\d_flipped.bmp")
tool1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_tool_\d\d.bmp|SHINEd_tool_\d\d_flipped.bmp")
tool2 = ro.findall(str(filenames))
tool = tool1+tool2
del tool1, tool2

# 5. mani
ro = re.compile("SHINEd_Mani_\d.bmp|SHINEd_Mani_\d_flipped.bmp")
mani1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_Mani_\d\d.bmp|SHINEd_Mani_\d\d_flipped.bmp")
mani2 = ro.findall(str(filenames))
mani = mani1+mani2
del mani1, mani2

# 6. nman
ro = re.compile("SHINEd_NMan_\d.bmp|SHINEd_NMan_\d_flipped.bmp")
nman1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_NMan_\d\d.bmp|SHINEd_NMan_\d\d_flipped.bmp")
nman2 = ro.findall(str(filenames))
nman = nman1+nman2
del nman1, nman2

# 7. chair
ro = re.compile("SHINEd_Chair_\d.bmp|SHINEd_Chair_\d_flipped.bmp")
chair1 = ro.findall(str(filenames))
ro = re.compile("SHINEd_Chair_\d\d.bmp|SHINEd_Chair_\d\d_flipped.bmp")
chair2 = ro.findall(str(filenames))
chair = chair1+chair2
del chair1, chair2, ro

filenames = body+hand+face+tool+mani+nman+chair
del body, hand, face, tool, mani, nman, chair, img_dir

images_experiment = [plt.imread(image) for image in filenames]
images_experiment = np.array(images_experiment)

del filenames

#%% 2.) Images original (making them flipped)

img_dir = r"D:\THESIS\IMAGES ORIGINAL, 168"
os.chdir(img_dir)
body = [name for name in glob.glob('*body*')]
hand = [name for name in glob.glob('*hand*')]
face = [name for name in glob.glob('*face*')]
tool = [name for name in glob.glob('*tool*')]
man = [name for name in glob.glob('*Mani*')]
nman = [name for name in glob.glob('*NMan*')]
chair = [name for name in glob.glob('*Chair*')]
filenames = body + hand + face + tool + man + nman + chair
del body, hand, face, tool, man, nman, chair, img_dir

images = [plt.imread(image) for image in filenames]
images_original = []
for _ in images:
    if _.shape == (400, 400, 3):
        images_original.append(_[:,:,0])
        images_original.append(np.fliplr(_[:,:,0]))
    elif _.shape == (400, 400):
        images_original.append(_)
        images_original.append(np.fliplr(_))
    else:
        print("Unexpected shape.")

del filenames, images
images_original = np.array(images_original)

#plt.imshow(images_original[0])
#plt.imshow(images_original[1])

#%% 3.) Making silhouettes

silhouettes = images_original
for _ in silhouettes:
    for x in range(0,400):
        for y in range(0,400):
            if _[x,y] != 255:
                _[x,y] = 0

del x, y

#%% 4.) Making co-s 
 
co_exp = np.corrcoef(images_experiment.reshape(images_experiment.shape[0], -1))
co_orig = np.corrcoef(images_original.reshape(images_original.shape[0], -1))
co_silh = np.corrcoef(silhouettes.reshape(silhouettes.shape[0], -1))

# Exp
co_small_exp = np.zeros((7, 7))
x_ind = -48
y_ind = -48
for x in range(0, 7):
    x_ind += 48
    y_ind = - 48
    for y in range(0, 7):
        y_ind += 48
        co_small_exp[x, y] = np.mean(co_exp[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
del x, x_ind, y, y_ind
    
# Orig
co_small_orig = np.zeros((7, 7))
x_ind = -48
y_ind = -48
for x in range(0, 7):
    x_ind += 48
    y_ind = - 48
    for y in range(0, 7):
        y_ind += 48
        co_small_orig[x, y] = np.mean(co_orig[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
del x, x_ind, y, y_ind

# Silh
co_small_silh = np.zeros((7, 7))
x_ind = -48
y_ind = -48
for x in range(0, 7):
    x_ind += 48
    y_ind = - 48
    for y in range(0, 7):
        y_ind += 48
        co_small_silh[x, y] = np.mean(co_silh[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
del x, x_ind, y, y_ind

#%% 5.) Plot all

fig = plt.figure()

plt.subplot(2,3,1)
plt.imshow(co_silh)
plt.colorbar()
plt.clim(-1, 1)
plt.title("Silhouettes")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(co_small_silh)
plt.colorbar()
plt.clim(0, 1)
plt.title("Silhouettes \n(averaged)")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(co_exp)
plt.colorbar()
plt.clim(-1, 1)
plt.title("Images used in the experiment")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(co_small_exp)
plt.colorbar()
plt.clim(0, 1)
plt.title("Images used in the experiment \n(averaged)")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(co_silh)
plt.colorbar()
plt.clim(-1, 1)
plt.title("Original images")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(co_small_silh)
plt.colorbar()
plt.clim(0, 1)
plt.title("Original images \n(averaged)")
plt.axis("off")

plt.show()
