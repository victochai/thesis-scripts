
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
import numpy as np

#%% 2.) Cos

os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv Big")
l = os.listdir()
del l[19]

cos = []
for _ in l:
    cos.append(loadmat(_)["co"])

mat = loadmat("rdms_vgg19.mat")["rdms_vgg19"]
rdms = []
for _ in range(0, 19):
    rdms.append(mat[_][0])
del mat, l    
    
#%% 3.) Cos small

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
del co, small, x, x_ind, y, y_ind

rdms_small = []
for rdm in rdms:
    x_ind = -48
    y_ind = -48
    small = np.zeros((7, 7))
    for x in range(0, 7):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 7):
            y_ind += 48
            small[x, y] = np.mean(rdm[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    rdms_small.append(small)    
del rdm, small, x, x_ind, y, y_ind

#%% 4.) Visualize corrs

fig = plt.figure()
fig.suptitle("VGG-19 (19 layers)\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos)):
    plt.subplot(4,5,_+1)
    plt.imshow(cos[_])
    plt.colorbar()
    plt.clim([-.25, 1])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("VGG-19 (19 layers)\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos_small)):
    plt.subplot(4,5,_+1)
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.clim([.13, .94])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

#%% 5.) Visualize RDMs

fig = plt.figure()
fig.suptitle("VGG-19 RDMs (19 layers)\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms)):
    plt.subplot(4,5,_+1)
    plt.imshow(rdms[_])
    plt.colorbar()
    #plt.clim([0, 1.25])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("VGG-19 RDMs (19 layers)\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms_small)):
    plt.subplot(4,5,_+1)
    plt.imshow(rdms_small[_])
    plt.colorbar()
    plt.clim([.28, .82])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()

#%% 6.) Save the results

os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv small, different averaging")

COS_small = {"cos_small" : cos_small}
RDMS_small = {"rdms_small" : rdms_small}

savemat("COS_small.mat", COS_small)
savemat("RDMS_small.mat", RDMS_small)

#%% BODY PARTS VS. OBJECTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cos_small_normalized = []
for _ in cos_small:
    cos_small_normalized.append((_ - np.mean(_)) / np.std(_))

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

#%% 1.) co_small normalized

fig = plt.figure()
fig.suptitle("VGG-19 | BODY PARTS AND OBJECTS CORRELATION | NORMALIZED", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in cos_small_normalized]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in cos_small_normalized]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.legend(["body", ])
plt.xticks(range(0,19), range(1,20))
plt.ylim([-1.7, 0.3])
plt.legend(["body", "hand", "face"], loc="lower left")
del body_, hand_, face_

plt.subplot(2,2,2)
plt.title("TOOLS")
body_ = [co_small[body, tool] for co_small in cos_small_normalized]
hand_ = [co_small[hand, tool] for co_small in cos_small_normalized]
face_ = [co_small[face, tool] for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.7, 0.3])
del body_, hand_, face_

plt.subplot(2,2,3)
plt.title("MANI")
body_ = [co_small[body, mani] for co_small in cos_small_normalized]
hand_ = [co_small[hand, mani] for co_small in cos_small_normalized]
face_ = [co_small[face, mani] for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.7, 0.3])
del body_, hand_, face_

plt.subplot(2,2,4)
plt.title("NMAN")
body_ = [co_small[body, nman] for co_small in cos_small_normalized]
hand_ = [co_small[hand, nman] for co_small in cos_small_normalized]
face_ = [co_small[face, nman] for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.7, 0.3])
del body_, hand_, face_

#%% 2.) rdms from normalized co_small

rdms_small_normalized = []
for _ in cos_small_normalized:
    rdms_small_normalized.append(1 - _)
    
fig = plt.figure()
fig.suptitle("VGG-19 | BODY PARTS AND OBJECTS dissimilarities (from normalized split-half corr.)", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in rdms_small_normalized]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in rdms_small_normalized]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.legend(["body", ])
plt.xticks(range(0,19), range(1,20))
plt.ylim([0, 2.8])
plt.legend(["body", "hand", "face"], loc="lower left")
del body_, hand_, face_

plt.subplot(2,2,2)
plt.title("TOOLS")
body_ = [co_small[body, tool] for co_small in rdms_small_normalized]
hand_ = [co_small[hand, tool] for co_small in rdms_small_normalized]
face_ = [co_small[face, tool] for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0, 2.8])
del body_, hand_, face_

plt.subplot(2,2,3)
plt.title("MANI")
body_ = [co_small[body, mani] for co_small in rdms_small_normalized]
hand_ = [co_small[hand, mani] for co_small in rdms_small_normalized]
face_ = [co_small[face, mani] for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0, 2.8])
del body_, hand_, face_

plt.subplot(2,2,4)
plt.title("NMAN")
body_ = [co_small[body, nman] for co_small in rdms_small_normalized]
hand_ = [co_small[hand, nman] for co_small in rdms_small_normalized]
face_ = [co_small[face, nman] for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0, 2.8])
del body_, hand_, face_    

#%% 3.) rdms from NON-normalized co_small
    
fig = plt.figure()
fig.suptitle("VGG-19 | BODY PARTS AND OBJECTS dissimilarities", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in rdms_small]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in rdms_small]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.ylim([0, .85])
plt.legend(["body", "hand", "face"], loc="lower right")
del body_, hand_, face_

plt.subplot(2,2,2)
plt.title("TOOLS")
body_ = [co_small[body, tool] for co_small in rdms_small]
hand_ = [co_small[hand, tool] for co_small in rdms_small]
face_ = [co_small[face, tool] for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([0, .85])
del body_, hand_, face_

plt.subplot(2,2,3)
plt.title("MANI")
body_ = [co_small[body, mani] for co_small in rdms_small]
hand_ = [co_small[hand, mani] for co_small in rdms_small]
face_ = [co_small[face, mani] for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([0, .85])
del body_, hand_, face_

plt.subplot(2,2,4)
plt.title("NMAN")
body_ = [co_small[body, nman] for co_small in rdms_small]
hand_ = [co_small[hand, nman] for co_small in rdms_small]
face_ = [co_small[face, nman] for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([0, .85])
del body_, hand_, face_    

#%% ORIGINAL IMAGES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
os.chdir(r"D:\thesis-scripts\Neural networks\VGG19\Original images")

cos = scipy.io.loadmat("cos_VGG.mat")["cos"][0]
cos = np.array([cos[_] for _ in range(0, 19)])
cos_small = scipy.io.loadmat("cos_small_original.mat")["cos_small"]

#%% Visualize

fig = plt.figure()
fig.suptitle("VGG-19 (19 layers) | ORIGINAL IMAGES \nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos)):
    plt.subplot(4,5,_+1)
    plt.imshow(cos[_])
    plt.colorbar()
    plt.clim([-.3, 1])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("VGG-19 (19 layers) | ORIGINAL IMAGES\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos_small)):
    plt.subplot(4,5,_+1)
    plt.imshow(cos_small[_])
    plt.colorbar()
    #plt.clim([0, .8])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()

#%% RDMS

rdms = []
for _ in range(0, 19):
    rdms.append(1 - cos[_])

rdms_small = []
for rdm in rdms:
    x_ind = -48
    y_ind = -48
    small = np.zeros((7, 7))
    for x in range(0, 7):
        x_ind += 48
        y_ind = - 48
        for y in range(0, 7):
            y_ind += 48
            small[x, y] = np.mean(rdm[0+x_ind:48+x_ind, 0+y_ind:48+y_ind])
    rdms_small.append(small)    
del rdm, small, x, x_ind, y, y_ind

fig = plt.figure()
fig.suptitle("VGG-19 RDMs (19 layers) | ORIGINAL IMAGES\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms)):
    plt.subplot(4,5,_+1)
    plt.imshow(rdms[_])
    plt.colorbar()
    plt.clim([0, 1.3])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("VGG RDMs (19 layers) | ORIGINAL IMAGES\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms_small)):
    plt.subplot(4,5,_+1)
    plt.imshow(rdms_small[_])
    plt.colorbar()
    plt.clim([0.22, .91])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()

#%% 1.) co_small normalized

fig = plt.figure()
fig.suptitle("VGG-19 | BODY PARTS AND OBJECTS CORRELATION | NORMALIZED \n ORIGINAL IMAGES", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in cos_small_normalized]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in cos_small_normalized]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.legend(["body", ])
plt.xticks(range(0,19), range(1,20))
plt.ylim([-1.53, 1.82])
plt.legend(["body", "hand", "face"], loc="top right")
del body_, hand_, face_

plt.subplot(2,2,2)
plt.title("TOOLS")
body_ = [co_small[body, tool] for co_small in cos_small_normalized]
hand_ = [co_small[hand, tool] for co_small in cos_small_normalized]
face_ = [co_small[face, tool] for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="top right")
plt.ylim([-1.53, 1.82])
del body_, hand_, face_

plt.subplot(2,2,3)
plt.title("MANI")
body_ = [co_small[body, mani] for co_small in cos_small_normalized]
hand_ = [co_small[hand, mani] for co_small in cos_small_normalized]
face_ = [co_small[face, mani] for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="top right")
plt.ylim([-1.53, 1.82])
del body_, hand_, face_

plt.subplot(2,2,4)
plt.title("NMAN")
body_ = [co_small[body, nman] for co_small in cos_small_normalized]
hand_ = [co_small[hand, nman] for co_small in cos_small_normalized]
face_ = [co_small[face, nman] for co_small in cos_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="top right")
plt.ylim([-1.53, 1.82])
del body_, hand_, face_

#%% 2.) rdms from normalized co_small

rdms_small_normalized = []
for _ in cos_small_normalized:
    rdms_small_normalized.append(1 - _)
    
fig = plt.figure()
fig.suptitle("VGG-19 | BODY PARTS AND OBJECTS dissimilarities (from normalized split-half corr.) \n ORIGINAL IMAGES", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in rdms_small_normalized]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in rdms_small_normalized]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.legend(["body", ])
plt.xticks(range(0,19), range(1,20))
plt.ylim([-0.75, 2.6])
plt.legend(["body", "hand", "face"], loc="lower right")
del body_, hand_, face_

plt.subplot(2,2,2)
plt.title("TOOLS")
body_ = [co_small[body, tool] for co_small in rdms_small_normalized]
hand_ = [co_small[hand, tool] for co_small in rdms_small_normalized]
face_ = [co_small[face, tool] for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([-0.75, 2.6])
del body_, hand_, face_

plt.subplot(2,2,3)
plt.title("MANI")
body_ = [co_small[body, mani] for co_small in rdms_small_normalized]
hand_ = [co_small[hand, mani] for co_small in rdms_small_normalized]
face_ = [co_small[face, mani] for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([-0.75, 2.6])
del body_, hand_, face_

plt.subplot(2,2,4)
plt.title("NMAN")
body_ = [co_small[body, nman] for co_small in rdms_small_normalized]
hand_ = [co_small[hand, nman] for co_small in rdms_small_normalized]
face_ = [co_small[face, nman] for co_small in rdms_small_normalized]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([-0.75, 2.6])
del body_, hand_, face_    

#%% 3.) rdms from NON-normalized co_small
    
fig = plt.figure()
fig.suptitle("VGG-19 | BODY PARTS AND OBJECTS dissimilarities \n ORIGINAL IMAGES", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in rdms_small]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in rdms_small]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.ylim([0, .9])
plt.legend(["body", "hand", "face"], loc="lower right")
del body_, hand_, face_

plt.subplot(2,2,2)
plt.title("TOOLS")
body_ = [co_small[body, tool] for co_small in rdms_small]
hand_ = [co_small[hand, tool] for co_small in rdms_small]
face_ = [co_small[face, tool] for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([0, .9])
del body_, hand_, face_

plt.subplot(2,2,3)
plt.title("MANI")
body_ = [co_small[body, mani] for co_small in rdms_small]
hand_ = [co_small[hand, mani] for co_small in rdms_small]
face_ = [co_small[face, mani] for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([0, .9])
del body_, hand_, face_

plt.subplot(2,2,4)
plt.title("NMAN")
body_ = [co_small[body, nman] for co_small in rdms_small]
hand_ = [co_small[hand, nman] for co_small in rdms_small]
face_ = [co_small[face, nman] for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.xticks(range(0,19), range(1,20))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([0, .9])
del body_, hand_, face_   
