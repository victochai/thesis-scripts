#%% 1.) Loading libraries

import scipy.io
import matplotlib.pyplot as plt
import os

#%% 2.) Cos

os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Big")
mat = scipy.io.loadmat('cos_alex.mat')["cos"]
cos = []
for _ in range(0, 8):
    cos.append(mat[_][0])
del mat    

mat = scipy.io.loadmat('rmds_alex.mat')["rdms"]
rdms = []
for _ in range(0, 8):
    rdms.append(mat[_][0])
del mat    
    
#%% 3.) Cos small

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
del co, small, x, x_ind, y, y_ind

import numpy as np
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

# layers = ["conv1", "conv2", "conv3", "conv4", "conv5", 
#           "fc6", "fc7", "fc8"]

#%% 4.) Visualize corrs

fig = plt.figure()
fig.suptitle("ALEXNET (8 layers)\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos)):
    plt.subplot(2,4,_+1)
    plt.imshow(cos[_])
    plt.colorbar()
    plt.clim([-.25, 1])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("ALEXNET (8 layers)\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos_small)):
    plt.subplot(2,4,_+1)
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.clim([.30, .75])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

#%% 5.) Visualize RDMs

fig = plt.figure()
fig.suptitle("ALEXNET RDMs (8 layers)\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms)):
    plt.subplot(2,4,_+1)
    plt.imshow(rdms[_])
    plt.colorbar()
    plt.clim([0, 1.25])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("ALEXNET RDMs (8 layers)\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms_small)):
    plt.subplot(2,4,_+1)
    plt.imshow(rdms_small[_])
    plt.colorbar()
    plt.clim([.25, .80])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()

#%% 6.) Save the results

os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv small, different averaging")

COS_small = {"cos_small" : cos_small}
RDMS_small = {"rdms_small" : rdms_small}

scipy.io.savemat("COS_small.mat", COS_small)
scipy.io.savemat("RDMS_small.mat", RDMS_small)

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
fig.suptitle("ALEXNET | BODY PARTS AND OBJECTS CORRELATION | NORMALIZED", color="red")

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
plt.xticks(range(0,8), range(1,9))
plt.ylim([-1.8, 0.5])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.8, 0.5])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.8, 0.5])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.8, 0.5])
del body_, hand_, face_

#%% 2.) rdms from normalized co_small

rdms_small_normalized = []
for _ in cos_small_normalized:
    rdms_small_normalized.append(1 - _)
    
fig = plt.figure()
fig.suptitle("ALEXNET | BODY PARTS AND OBJECTS dissimilarities (from normalized split-half corr.)", color="red")

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
plt.xticks(range(0,8), range(1,9))
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
plt.xticks(range(0,8), range(1,9))
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
plt.xticks(range(0,8), range(1,9))
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0, 2.8])
del body_, hand_, face_    

#%% 3.) rdms from NON-normalized co_small
    
fig = plt.figure()
fig.suptitle("ALEXNET | BODY PARTS AND OBJECTS dissimilarities", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in rdms_small]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in rdms_small]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.legend(["body", ])
plt.xticks(range(0,8), range(1,9))
plt.ylim([.3, .8])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([.3, .8])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([.3, .8])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([.3, .8])
del body_, hand_, face_    

#%% ORIGINAL IMAGES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
os.chdir(r"D:\thesis-scripts\Neural networks\Alexnet\Original images")

cos = scipy.io.loadmat("cos_ALEX.mat")["cos"][0]
cos = np.array([cos[_] for _ in range(0, 8)])
cos_small = scipy.io.loadmat("cos_small_original.mat")["cos_small"]

#%% Visualize

fig = plt.figure()
fig.suptitle("ALEXNET (8 layers) | ORIGINAL IMAGES \nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos)):
    plt.subplot(2,4,_+1)
    plt.imshow(cos[_])
    plt.colorbar()
    plt.clim([-.4, 1])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("ALEXNET (8 layers) | ORIGINAL IMAGES\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(cos_small)):
    plt.subplot(2,4,_+1)
    plt.imshow(cos_small[_])
    plt.colorbar()
    plt.clim([0, .7])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()

#%% RDMS

rdms = []
for _ in range(0, 8):
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
fig.suptitle("ALEXNET RDMs (8 layers) | ORIGINAL IMAGES\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms)):
    plt.subplot(2,4,_+1)
    plt.imshow(rdms[_])
    plt.colorbar()
    #plt.clim([0, 1.4])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()  

fig = plt.figure()
fig.suptitle("ALEXNET RDMs (8 layers) | ORIGINAL IMAGES\nEvery condition is averaged\nBody, hand, face, tool, mani, nman, chair")
for _ in range(0, len(rdms_small)):
    plt.subplot(2,4,_+1)
    plt.imshow(rdms_small[_])
    plt.colorbar()
    plt.clim([0.3, .97])
    plt.axis("off")
    plt.title(str(_+1), fontsize=9)
plt.show()

#%% 1.) co_small normalized

fig = plt.figure()
fig.suptitle("ALEXNET | BODY PARTS AND OBJECTS CORRELATION | NORMALIZED \n ORIGINAL IMAGES", color="red")

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
plt.xticks(range(0,8), range(1,9))
plt.ylim([-1.8, 1.8])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.8, 1.8])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.8, 1.8])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([-1.8, 1.8])
del body_, hand_, face_

#%% 2.) rdms from normalized co_small

rdms_small_normalized = []
for _ in cos_small_normalized:
    rdms_small_normalized.append(1 - _)
    
fig = plt.figure()
fig.suptitle("ALEXNET | BODY PARTS AND OBJECTS dissimilarities (from normalized split-half corr.) \n ORIGINAL IMAGES", color="red")

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
plt.xticks(range(0,8), range(1,9))
plt.ylim([-0.75, 3])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([-0.75, 3])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([-0.75, 3])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower right")
plt.ylim([-0.75, 3])
del body_, hand_, face_    

#%% 3.) rdms from NON-normalized co_small
    
fig = plt.figure()
fig.suptitle("ALEXNET | BODY PARTS AND OBJECTS dissimilarities \n ORIGINAL IMAGES", color="red")

plt.subplot(2,2,1)
plt.title("OBJECTS (tool, mani, nman)")
body_ = [(co_small[body, tool] + co_small[body, mani] + co_small[body, nman])/3 for co_small in rdms_small]
hand_ = [(co_small[hand, tool] + co_small[hand, mani] + co_small[hand, nman])/3 for co_small in rdms_small]
face_ = [(co_small[face, tool] + co_small[face, mani] + co_small[face, nman])/3 for co_small in rdms_small]
plt.plot(body_)
plt.plot(hand_)
plt.plot(face_)
plt.grid()
plt.legend(["body", ])
plt.xticks(range(0,8), range(1,9))
plt.ylim([0.6, .95])
plt.legend(["body", "hand", "face"], loc="lower left")
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0.6, .95])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0.6, .95])
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
plt.xticks(range(0,8), range(1,9))
plt.legend(["body", "hand", "face"], loc="lower left")
plt.ylim([0.6, .95])
del body_, hand_, face_   

#%% SAVE the results

