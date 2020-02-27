#%% Import modules

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#%% Brain representations

mat = scipy.io.loadmat('anterior_big_matrix.mat')
mat = mat[""]
