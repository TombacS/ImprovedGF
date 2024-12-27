# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:33:43 2021

@author: Dell
"""


import numpy as np
import h5py

from .. import data_base


class USPS(data_base.Data_base):
    # name = "USPS"
    # dims = 16*16*1 = 256
    # shape = [16, 16, 1]
    # num = 7291 + 2007 = 9298
    c = 10
    
    def __init__(self, file_root):
        data_base.Data_base.__init__(self, file_root)
        

    def load_data(self):

        data_dir = self.file_root + "\\USPS\\usps.h5"

        X = []
        Y = []
        
        with h5py.File(data_dir, 'r') as f:
            train = f.get('train')
            X_tr = train.get('data')[:].astype(self.dtype)
            Y_tr = train.get('target')[:].astype(int)
            
            test = f.get('test')
            X_te = test.get('data')[:].astype(self.dtype)
            Y_te = test.get('target')[:].astype(int)
                
        X = np.concatenate((X_tr, X_te), axis=0)
        Y = np.concatenate((Y_tr, Y_te), axis=0)
        # print('X.shape=', X.shape, 'Y.shape=', Y.shape)

        return X, Y, self.c



# data_dir = "../../data/COIL/coil-100/"
# img = cv.imread('obj1__0.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# print(type(img), img.shape)
# plt.imshow(img)
