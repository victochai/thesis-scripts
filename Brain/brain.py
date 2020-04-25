#%% Import modules

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#%% Brain representations + Nets

os.chdir(r"D:\thesis-scripts\Brain\Brain representations")

ant = scipy.io.loadmat("anterior_big_MATRIX.mat")["anterior_big_MATRIX"]
ant_left = scipy.io.loadmat("anterior_left.mat")["anterior_left"]
ant_right = scipy.io.loadmat("anterior_right.mat")["anterior_right"]
# new = scipy.io.loadmat("new.mat")["new"]

op = scipy.io.loadmat("OP_MATRIX.mat")["OP_MATRIX"]
calc = scipy.io.loadmat("CALC_MATRIX.mat")["CALC_MATRIX"]
op_calc = scipy.io.loadmat("OP_CALC_MATRIX.mat")["OP_CALC_MATRIX"]
pos = scipy.io.loadmat("pos_res2_mvpa_MATRIX.mat")["pos_res2_mvpa"]

#%% RDMs + Averaging

# Averaging
ant = np.mean(ant, 2)
ant_left = np.mean(ant_left, 2)
ant_right = np.mean(ant_right, 2)

calc = np.mean(calc, 2)
op = np.mean(op, 2)
op_calc = np.mean(op_calc, 2)
pos = np.mean(pos, 2)

# [x, y] averaging
ant_av = np.zeros((7, 7))
ant_left_av = np.zeros((7, 7))
ant_right_av = np.zeros((7, 7))

calc_av = np.zeros((7, 7))
op_av = np.zeros((7, 7))
op_calc_av = np.zeros((7, 7))
pos_av = np.zeros((7, 7))

for x in range(0, 7):
    for y in range(0, 7):
        # [x, y]
        ant_av[x, y] = (ant[x, y] + ant[y, x])/2
        ant_left_av[x, y] = (ant_left[x, y] + ant_left[y, x])/2
        ant_right_av[x, y] = (ant_right[x, y] + ant_right[y, x])/2
        calc_av[x, y] = (calc[x, y] + calc[y, x])/2
        op_av[x, y] = (op[x, y] + op[y, x])/2
        op_calc_av[x, y] = (op_calc[x, y] + op_calc[y, x])/2
        pos_av[x, y] = (pos[x, y] + pos[y, x])/2
        # [y, x]
        ant_av[y, x] = (ant[x, y] + ant[y, x])/2
        ant_left_av[y, x] = (ant_left[x, y] + ant_left[y, x])/2
        ant_right_av[y, x] = (ant_right[x, y] + ant_right[y, x])/2
        calc_av[y, x] = (calc[x, y] + calc[y, x])/2
        op_av[y, x] = (op[x, y] + op[y, x])/2
        op_calc_av[y, x] = (op_calc[x, y] + op_calc[y, x])/2
        pos_av[y, x] = (pos[x, y] + pos[y, x])/2 
        
# Make RDMs       
rdm_ant = 1 - ant_av
rdm_ant_left = 1 -  ant_left_av
rdm_ant_right = 1 -  ant_right_av
rdm_calc = 1 - calc_av
rdm_op = 1 - op_av
rdm_op_calc = 1 - op_calc_av
rdm_pos = 1 - pos_av

#%% Save the results

# RDMs
os.chdir(r"D:\thesis-scripts\Brain\Brain representations\RDMs and other")
RDM_ant = {"rdm_ant" : rdm_ant}
RDM_ant_left = {"rdm_ant_left" : rdm_ant_left}
RDM_ant_right = {"rdm_ant_right" : rdm_ant_right}
RDM_calc = {"rdm_calc" : rdm_calc}
RDM_op = {"rdm_op" : rdm_op}
RDM_op_calc = {"rdm_op_calc" : rdm_op_calc}
RDM_pos = {"rdm_pos" : rdm_pos}

scipy.io.savemat("RDM_ant.mat", RDM_ant)
scipy.io.savemat("RDM_ant_left.mat", RDM_ant_left)
scipy.io.savemat("RDM_ant_right.mat", RDM_ant_right)
scipy.io.savemat("RDM_calc.mat", RDM_calc)
scipy.io.savemat("RDM_op.mat", RDM_op)
scipy.io.savemat("RDM_op_calc.mat", RDM_op_calc)
scipy.io.savemat("RDM_pos.mat", RDM_pos)

# Not RDMs
os.chdir(r"D:\thesis-scripts\Brain\Brain representations\RDMs and other")
ANT = {"ant_av" : ant_av}
ANT_LEFT = {"ant_left_av" : ant_left_av}
ANT_RIGHT = {"ant_right_av" : ant_right_av}
CALC = {"calc_av" : calc_av}
OP = {"op_av" : op_av}
OP_CALC = {"op_calc_av" : op_calc_av}
POS = {"pos_av" : pos_av}

scipy.io.savemat("ANT.mat", ANT)
scipy.io.savemat("ANT_LEFT.mat", ANT_LEFT)
scipy.io.savemat("ANT_RIGHT.mat", ANT_RIGHT)
scipy.io.savemat("CALC.mat", CALC)
scipy.io.savemat("OP.mat", OP)
scipy.io.savemat("OP_CALC.mat", OP_CALC)
scipy.io.savemat("POS.mat", POS)

#%% Visualize stuff (not RDMs)

fig = plt.figure()
fig.suptitle("Brain split-half correlations\nBody, hand, face, tool, man, nman, chair")

plt.subplot(2,4,1)
plt.imshow(ant_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("ITG + Anterior IOG")

plt.subplot(2,4,2)
plt.imshow(ant_left_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("ITG + Anterior IOG (left)")

plt.subplot(2,4,3)
plt.imshow(ant_right_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("ITG + Anterior IOG (right)")

plt.subplot(2,4,5)
plt.imshow(pos_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("Posterior IOG")

plt.subplot(2,4,6)
plt.imshow(calc_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("Calcarine cortex")

plt.subplot(2,4,7)
plt.imshow(op_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("Occipital pole")

plt.subplot(2,4,8)
plt.imshow(op_calc_av)
plt.colorbar()
plt.clim(-1, 1)
plt.axis("off")
plt.title("Calc. cort. + Occip. pole")

plt.show()

#%% Visualize RDMs

fig = plt.figure()
fig.suptitle("Brain split-half dissimilarities\nBody, hand, face, tool, man, nman, chair")

plt.subplot(2,4,1)
plt.imshow(rdm_ant)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("ITG + Anterior IOG")

plt.subplot(2,4,2)
plt.imshow(rdm_ant_left)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("ITG + Anterior IOG (left)")

plt.subplot(2,4,3)
plt.imshow(rdm_ant_right)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("ITG + Anterior IOG (right)")

plt.subplot(2,4,5)
plt.imshow(rdm_pos)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("Posterior IOG")

plt.subplot(2,4,6)
plt.imshow(rdm_calc)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("Calcarine cortex")

plt.subplot(2,4,7)
plt.imshow(rdm_op)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("Occipital pole")

plt.subplot(2,4,8)
plt.imshow(rdm_op_calc)
plt.colorbar()
#plt.clim(0, 1.5)
plt.axis("off")
plt.title("Calc. cort. + Occip. pole")

plt.show()

#%% BODY PARTS VS. OBJECTS

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND OBJECTS (tool, mani, nman) CORRELATION", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, (ant_left_av[body, tool] + ant_left_av[body, mani] + ant_left_av[body, nman])/3)
plt.bar(2, (ant_left_av[hand, tool] + ant_left_av[hand, mani] + ant_left_av[hand, nman])/3)
plt.bar(3, (ant_left_av[face, tool] + ant_left_av[face, mani] + ant_left_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, (ant_right_av[body, tool] + ant_right_av[body, mani] + ant_right_av[body, nman])/3)
plt.bar(2, (ant_right_av[hand, tool] + ant_right_av[hand, mani] + ant_right_av[hand, nman])/3)
plt.bar(3, (ant_right_av[face, tool] + ant_right_av[face, mani] + ant_right_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, (ant_av[body, tool] + ant_av[body, mani] + ant_av[body, nman])/3)
plt.bar(2, (ant_av[hand, tool] + ant_av[hand, mani] + ant_av[hand, nman])/3)
plt.bar(3, (ant_av[face, tool] + ant_av[face, mani] + ant_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, (pos_av[body, tool] + pos_av[body, mani] + pos_av[body, nman])/3)
plt.bar(2, (pos_av[hand, tool] + pos_av[hand, mani] + pos_av[hand, nman])/3)
plt.bar(3, (pos_av[face, tool] + pos_av[face, mani] + pos_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, (calc_av[body, tool] + calc_av[body, mani] + calc_av[body, nman])/3)
plt.bar(2, (calc_av[hand, tool] + calc_av[hand, mani] + calc_av[hand, nman])/3)
plt.bar(3, (calc_av[face, tool] + calc_av[face, mani] + calc_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, (op_av[body, tool] + op_av[body, mani] + op_av[body, nman])/3)
plt.bar(2, (op_av[hand, tool] + op_av[hand, mani] + op_av[hand, nman])/3)
plt.bar(3, (op_av[face, tool] + op_av[face, mani] + op_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, (op_calc_av[body, tool] + op_calc_av[body, mani] + op_calc_av[body, nman])/3)
plt.bar(2, (op_calc_av[hand, tool] + op_calc_av[hand, mani] + op_calc_av[hand, nman])/3)
plt.bar(3, (op_calc_av[face, tool] + op_calc_av[face, mani] + op_calc_av[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

#%% BODY PARTS VS. TOOLS

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND TOOLS CORRELATION", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, ant_left_av[body, tool])
plt.bar(2, ant_left_av[hand, tool])
plt.bar(3, ant_left_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, ant_right_av[body, tool])
plt.bar(2, ant_right_av[hand, tool])
plt.bar(3, ant_right_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, ant_av[body, tool])
plt.bar(2, ant_av[hand, tool])
plt.bar(3, ant_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, pos_av[body, tool])
plt.bar(2, pos_av[hand, tool])
plt.bar(3, pos_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, calc_av[body, tool])
plt.bar(2, calc_av[hand, tool])
plt.bar(3, calc_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, op_av[body, tool])
plt.bar(2, op_av[hand, tool])
plt.bar(3, op_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, op_calc_av[body, tool])
plt.bar(2, op_calc_av[hand, tool])
plt.bar(3, op_calc_av[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

#%% BODY PARTS VS. MANI

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND MANI CORRELATION", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, ant_left_av[body, mani])
plt.bar(2, ant_left_av[hand, mani])
plt.bar(3, ant_left_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, ant_right_av[body, mani])
plt.bar(2, ant_right_av[hand, mani])
plt.bar(3, ant_right_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, ant_av[body, mani])
plt.bar(2, ant_av[hand, mani])
plt.bar(3, ant_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, pos_av[body, mani])
plt.bar(2, pos_av[hand, mani])
plt.bar(3, pos_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, calc_av[body, mani])
plt.bar(2, calc_av[hand, mani])
plt.bar(3, calc_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, op_av[body, mani])
plt.bar(2, op_av[hand, mani])
plt.bar(3, op_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, op_calc_av[body, mani])
plt.bar(2, op_calc_av[hand, mani])
plt.bar(3, op_calc_av[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

#%% BODY PARTS VS. NMAN

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND NMAN CORRELATION", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, ant_left_av[body, nman])
plt.bar(2, ant_left_av[hand, nman])
plt.bar(3, ant_left_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, ant_right_av[body, nman])
plt.bar(2, ant_right_av[hand, nman])
plt.bar(3, ant_right_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, ant_av[body, nman])
plt.bar(2, ant_av[hand, nman])
plt.bar(3, ant_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, pos_av[body, nman])
plt.bar(2, pos_av[hand, nman])
plt.bar(3, pos_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, calc_av[body, nman])
plt.bar(2, calc_av[hand, nman])
plt.bar(3, calc_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, .05])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, op_av[body, nman])
plt.bar(2, op_av[hand, nman])
plt.bar(3, op_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, op_calc_av[body, nman])
plt.bar(2, op_calc_av[hand, nman])
plt.bar(3, op_calc_av[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([-.5, 0])

#%% RDMs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################

#%% BODY PARTS VS. OBJECTS

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND OBJECTS (tool, mani, nman) DISSIMILARITY (1 - corr)", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, (rdm_ant_left[body, tool] + rdm_ant_left[body, mani] + rdm_ant_left[body, nman])/3)
plt.bar(2, (rdm_ant_left[hand, tool] + rdm_ant_left[hand, mani] + rdm_ant_left[hand, nman])/3)
plt.bar(3, (rdm_ant_left[face, tool] + rdm_ant_left[face, mani] + rdm_ant_left[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylabel("(1 - corr) dissimilarity")
plt.ylim(0, 1.5)

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, (rdm_ant_right[body, tool] + rdm_ant_right[body, mani] + rdm_ant_right[body, nman])/3)
plt.bar(2, (rdm_ant_right[hand, tool] + rdm_ant_right[hand, mani] + rdm_ant_right[hand, nman])/3)
plt.bar(3, (rdm_ant_right[face, tool] + rdm_ant_right[face, mani] + rdm_ant_right[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim(0, 1.5)

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, (rdm_ant[body, tool] + rdm_ant[body, mani] + rdm_ant[body, nman])/3)
plt.bar(2, (rdm_ant[hand, tool] + rdm_ant[hand, mani] + rdm_ant[hand, nman])/3)
plt.bar(3, (rdm_ant[face, tool] + rdm_ant[face, mani] + rdm_ant[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim(0, 1.5)

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, (rdm_pos[body, tool] + rdm_pos[body, mani] + rdm_pos[body, nman])/3)
plt.bar(2, (rdm_pos[hand, tool] + rdm_pos[hand, mani] + rdm_pos[hand, nman])/3)
plt.bar(3, (rdm_pos[face, tool] + rdm_pos[face, mani] + rdm_pos[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylabel("(1 - corr) dissimilarity")
plt.ylim(0, 1.5)

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, (rdm_calc[body, tool] + rdm_calc[body, mani] + rdm_calc[body, nman])/3)
plt.bar(2, (rdm_calc[hand, tool] + rdm_calc[hand, mani] + rdm_calc[hand, nman])/3)
plt.bar(3, (rdm_calc[face, tool] + rdm_calc[face, mani] + rdm_calc[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim(0, 1.5)

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, (rdm_op[body, tool] + rdm_op[body, mani] + rdm_op[body, nman])/3)
plt.bar(2, (rdm_op[hand, tool] + rdm_op[hand, mani] + rdm_op[hand, nman])/3)
plt.bar(3, (rdm_op[face, tool] + rdm_op[face, mani] + rdm_op[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim(0, 1.5)

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, (rdm_op_calc[body, tool] + rdm_op_calc[body, mani] + rdm_op_calc[body, nman])/3)
plt.bar(2, (rdm_op_calc[hand, tool] + rdm_op_calc[hand, mani] + rdm_op_calc[hand, nman])/3)
plt.bar(3, (rdm_op_calc[face, tool] + rdm_op_calc[face, mani] + rdm_op_calc[face, nman])/3)
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim(0, 1.5)

#%% BODY PARTS VS. TOOLS

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND TOOLS DISSIMILARITY (1 - corr)", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, rdm_ant_left[body, tool])
plt.bar(2, rdm_ant_left[hand, tool])
plt.bar(3, rdm_ant_left[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])
plt.ylabel("(1 - corr) dissimilarity")

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, rdm_ant_right[body, tool])
plt.bar(2, rdm_ant_right[hand, tool])
plt.bar(3, rdm_ant_right[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, rdm_ant[body, tool])
plt.bar(2, rdm_ant[hand, tool])
plt.bar(3, rdm_ant[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, rdm_pos[body, tool])
plt.bar(2, rdm_pos[hand, tool])
plt.bar(3, rdm_pos[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylabel("(1 - corr) dissimilarity")
plt.ylim([0, 1.5])

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, rdm_calc[body, tool])
plt.bar(2, rdm_calc[hand, tool])
plt.bar(3, rdm_calc[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, rdm_op[body, tool])
plt.bar(2, rdm_op[hand, tool])
plt.bar(3, rdm_op[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, rdm_op_calc[body, tool])
plt.bar(2, rdm_op_calc[hand, tool])
plt.bar(3, rdm_op_calc[face, tool])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

#%% BODY PARTS VS. MANI

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND MANI DISSIMILARITY (1 - corr)", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, rdm_ant_left[body, mani])
plt.bar(2, rdm_ant_left[hand, mani])
plt.bar(3, rdm_ant_left[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])
plt.ylabel("(1 - corr) dissimilarity")

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, rdm_ant_right[body, mani])
plt.bar(2, rdm_ant_right[hand, mani])
plt.bar(3, rdm_ant_right[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, rdm_ant[body, mani])
plt.bar(2, rdm_ant[hand, mani])
plt.bar(3, rdm_ant[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, rdm_pos[body, mani])
plt.bar(2, rdm_pos[hand, mani])
plt.bar(3, rdm_pos[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])
plt.ylabel("(1 - corr) dissimilarity")

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, rdm_calc[body, mani])
plt.bar(2, rdm_calc[hand, mani])
plt.bar(3, rdm_calc[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, rdm_op[body, mani])
plt.bar(2, rdm_op[hand, mani])
plt.bar(3, rdm_op[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, rdm_op_calc[body, mani])
plt.bar(2, rdm_op_calc[hand, mani])
plt.bar(3, rdm_op_calc[face, mani])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

#%% BODY PARTS VS. NMAN

body = 0
hand = 1
face = 2
tool = 3
mani = 4
nman = 5
chair = 6

fig = plt.figure()
fig.suptitle("BRAIN REGIONS AND NMAN DISSIMILARITY (1 - corr)", color="red")

# ANT LEFT
plt.subplot(2,4,1)
plt.title("ITG + Anterior IOG (left)")
plt.bar(1, rdm_ant_left[body, nman])
plt.bar(2, rdm_ant_left[hand, nman])
plt.bar(3, rdm_ant_left[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])
plt.ylabel("(1 - corr) dissimilarity")

# ANT RIGHT
plt.subplot(2,4,2)
plt.title("ITG + Anterior IOG (right)")
plt.bar(1, rdm_ant_right[body, nman])
plt.bar(2, rdm_ant_right[hand, nman])
plt.bar(3, rdm_ant_right[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# ANT 
plt.subplot(2,4,3)
plt.title("ITG + Anterior IOG (both)")
plt.bar(1, rdm_ant[body, nman])
plt.bar(2, rdm_ant[hand, nman])
plt.bar(3, rdm_ant[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# POS
plt.subplot(2,4,5)
plt.title("Posterior IOG")
plt.bar(1, rdm_pos[body, nman])
plt.bar(2, rdm_pos[hand, nman])
plt.bar(3, rdm_pos[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])
plt.ylabel("(1 - corr) dissimilarity")

# CALC
plt.subplot(2,4,6)
plt.title("Calcarine cortex")
plt.bar(1, rdm_calc[body, nman])
plt.bar(2, rdm_calc[hand, nman])
plt.bar(3, rdm_calc[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# OP
plt.subplot(2,4,7)
plt.title("Occipitale pole")
plt.bar(1, rdm_op[body, nman])
plt.bar(2, rdm_op[hand, nman])
plt.bar(3, rdm_op[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

# OP & CALC
plt.subplot(2,4,8)
plt.title("Calcarine cortex + Occipital pole")
plt.bar(1, rdm_op_calc[body, nman])
plt.bar(2, rdm_op_calc[hand, nman])
plt.bar(3, rdm_op_calc[face, nman])
plt.xticks([1,2,3], ["body", "hand", "face"])
plt.ylim([0, 1.5])

#%% Visualize decodings (full matrix)

os.chdir(r"D:\thesis-scripts\Brain\Brain representations")
decodings = scipy.io.loadmat("decodings_all_confusion_matrix.mat")["confusion_matrix_mean"]

ROIs = ['ITG + ant. IOG', 
        'ITG + ant. IOG (left)',
        'ITG + ant. IOG (right)',
        'Post. IOG', 'Occipital pole', 'Calc. cortex', 'Occip. pole + calc. cortex']

fig = plt.figure()
fig.suptitle("DECODINGS (LDA classifier) \nBody, hand, face, tool, man, nman, chair")

for _ in range(0, 7):
    if _ <= 2:
        plt.subplot(2,4,_+1)
    else:
        plt.subplot(2,4,_+2)
    plt.imshow(decodings[:,:,_])
    plt.colorbar()
    plt.clim(0, 7)
    plt.axis("off")
    plt.title(ROIs[_])

#%% Visualize decodings (3x3)
    
os.chdir(r"D:\thesis-scripts\Brain\Brain representations")
decodings = scipy.io.loadmat("decodings_objects_confusion_matrix.mat")["confusion_matrix_mean"]

ROIs = ['ITG + ant. IOG', 
        'ITG + ant. IOG (left)',
        'ITG + ant. IOG (right)',
        'Post. IOG', 'Occipital pole', 'Calc. cortex', 'Occip. pole + calc. cortex']

fig = plt.figure()
fig.suptitle("DECODINGS of only OBJECTS (LDA classifier) \nTool, man, nman")

for _ in range(0, 7):
    if _ <= 2:
        plt.subplot(2,4,_+1)
    else:
        plt.subplot(2,4,_+2)
    plt.imshow(decodings[:,:,_])
    plt.colorbar()
    plt.clim(0, 4.5)
    plt.axis("off")
    plt.title(ROIs[_])
