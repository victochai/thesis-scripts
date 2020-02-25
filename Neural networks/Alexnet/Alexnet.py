#%% Loading libraries

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

#%% Mat

mat = scipy.io.loadmat('correlations_AlEX.mat')
mat = mat["correlations_AlEX"]

body_objects = []
body_tool = []
body_man = []
body_nman = []

for _ in range(0,3):
    body_objects.append(mat[0][0][_][0])
    body_tool.append(mat[1][0][_][0])
    body_man.append(mat[2][0][_][0])
    body_nman.append(mat[3][0][_][0])
    
#%% Plot

for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Alexnet: Body parts vs. objects correlations")
plt.legend(("Corr: bodies vs. objects", "Corr: hands vs. objects", "Corr: faces vs. objects"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Alexnet: Body parts vs. tools correlations")
plt.legend(("Corr: bodies vs. tools", "Corr: hands vs. tools", "Corr: faces vs. tools"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Alexnet: Body parts vs. manipulable objects correlations")
plt.legend(("Corr: bodies vs. man", "Corr: hands vs. man", "Corr: faces vs. man"), loc='lower left')
plt.show()

for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Alexnet: Body parts vs. nonmanipulable objects correlations")
plt.legend(("Corr: bodies vs. Nman", "Corr: hands vs. Nman", "Corr: faces vs. Nman"), loc='lower left')
plt.show()

#%% All plots

fig = plt.figure()
fig.suptitle("ALEXNET")
# 1
plt.subplot(2,2,1)
for _ in range(0,3):
    plt.plot(body_objects[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts vs. objects correlations")
plt.legend(("Corr: bodies vs. objects", "Corr: hands vs. objects", "Corr: faces vs. objects"), loc='lower left')
# 2
plt.subplot(2,2,2)
for _ in range(0,3):
    plt.plot(body_tool[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts vs. tools correlations")
plt.legend(("Corr: bodies vs. tools", "Corr: hands vs. tools", "Corr: faces vs. tools"), loc='lower left')
# 3
plt.subplot(2,2,3)
for _ in range(0,3):
    plt.plot(body_man[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts vs. manipulable objects correlations")
plt.legend(("Corr: bodies vs. man", "Corr: hands vs. man", "Corr: faces vs. man"), loc='lower left')
# 4
plt.subplot(2,2,4)
for _ in range(0,3):
    plt.plot(body_nman[_][:,0],'-o')
plt.grid()
plt.ylim((0,1))
plt.xticks(list(range(0,8)), list(range(1,9)))
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.title("Body parts vs. nonmanipulable objects correlations")
plt.legend(("Corr: bodies vs. Nman", "Corr: hands vs. Nman", "Corr: faces vs. Nman"), loc='lower left')
plt.show()
