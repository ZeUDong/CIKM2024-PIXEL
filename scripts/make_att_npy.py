import numpy as np  
import pickle
import h5py
import scipy.io as sio
from torchvision import datasets, transforms
import os

split_path = './data/xlsa17/data/AWA2/att_splits.mat'
matcontent = sio.loadmat(split_path)
att = matcontent['att'].T
np.save("data/AwA2/awa2_att.npy", att)


split_path = './data/xlsa17/data/CUB/att_splits.mat'
matcontent = sio.loadmat(split_path)
att = matcontent['att'].T
np.save("data/CUB_200_2011/cub_att.npy", att)


split_path = './data/xlsa17/data/SUN/att_splits.mat'
matcontent = sio.loadmat(split_path)
att = matcontent['att'].T
np.save("data/SUN/sun_att.npy", att)
