
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:16:21 2021

@author: Dell
"""
import numpy as np
# from matplotlib import pyplot as plt
# import time

class Data_base:
    # name = "base"
    # dim = 
    # num =
    # c = 
    
    dtype = np.float32
    l = file_root = c = None
    
    def __init__(self, file_root):
        self.file_root = file_root

    def load_data(self):
        print('virtual load_data')

    # def suf_process(self, X, y, if_balance=True, noise_prop=0.001):
    #     if self.l < 0:
    #         self.l = int(- X.shape[0] * self.l)
    #     if self.l >= X.shape[0]:
    #         print('error')
    #         return None
        
    #     seq = np.random.permutation(X.shape[0])
    #     X = X[seq]
    #     y = y[seq]
            
    #     if if_balance:
    #         seq = self.solve_seq(y)
    #         X = X[seq]
    #         y = y[seq]
            
    #     # print('noise_prop: {:f}'.format(noise_prop))
    #     self.y_noise(y, int(noise_prop*self.l))
        
    #     count = self.count_label(y[:self.l])
    #     # print('cnt:', count)
    #     count = np.maximum(count, 1)

    #     return X, np.eye(self.c, dtype=self.dtype)[y] *np.amax(count)/count, y, self.l
    
    def y_noise(self, y, noise_num):
        noise_idx = np.random.choice(self.l, noise_num, replace=False)
        noise_values = np.random.choice(self.c, noise_num)
        y[noise_idx] = noise_values
        return

    def draw(self, X, y, title):
        print('virtual draw')
