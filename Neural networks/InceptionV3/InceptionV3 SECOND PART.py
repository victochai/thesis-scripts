#%% Modules

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#%% Inception

fm = scipy.io.loadmat("feature_map.mat")
fm = fm["feature_map"]
feature_maps = []
feature_maps.append(fm.T)
del fm

bodies_last_layer = corr_bodies_objects[0,:].reshape((1,4))
faces_last_layer = corr_faces_objects[0,:].reshape((1,4))
hands_last_layer = corr_hands_objects[0,:].reshape((1,4))

np.save('M_bodies_last_layer.npy', bodies_last_layer)
np.save('M_faces_last_layer.npy', faces_last_layer)
np.save('M_hands_last_layer.npy', hands_last_layer)

corr_bodies_objects[47,:] = bodies_last_layer[:,:]
corr_faces_objects[47,:] = faces_last_layer[:,:]
corr_hands_objects[47,:] = hands_last_layer[:,:]
