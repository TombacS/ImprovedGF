# -*- coding: utf-8 -*-

import numpy as np

# X: samples, [n,d]; m: num of anchors; t: num of iteration
# ret: anchors, [m,d]
def BKHK(X_tot, m_tot, t):
    U = []
    m_initial = m_tot
    
    def balance_divide(X, n1):
        if n1 <= 0 or X.shape[0] <= 0:
            print('balance_divide error')
            raise
        
        #initialize C1 and C2
        C1 = X[0]
        farthest = np.argmax(np.sum(np.square(C1 - X), axis=1))
        C2 = X[farthest]
        farthest = np.argmax(np.sum(np.square(C2 - X), axis=1))
        C1 = X[farthest]
        
        # 2-means
        for _ in range(t):
            e_dis_diff = np.sum(X * (C1 - C2), axis=1)
            g_seq = np.argsort(e_dis_diff)
            X1 = X[g_seq[:n1]]
            X2 = X[g_seq[n1:]]

            C1 = np.mean(X1, axis=0)
            C2 = np.mean(X2, axis=0)
        
        # print(g_seq[:n1], g_seq[n1:])
        return X1, X2
    
    # def centre_divide(X):
    #     if X.shape[0] <= 0:
    #         print('centre_divide error')
    #         raise
        
    #     #initialize C1 and C2
    #     C1 = X[0]
    #     farthest = np.argmax(np.sum(np.square(C1 - X), axis=1))
    #     C2 = X[farthest]
    #     farthest = np.argmax(np.sum(np.square(C2 - X), axis=1))
    #     C1 = X[farthest]
        
    #     # 2-means
    #     for _ in range(t):
    #         e_dis_diff = 2 * np.sum(X * (C1 - C2), axis=1) + C2 @ C2 - C1 @ C1
    #         X1_seq = np.where(e_dis_diff >= 0)
    #         X2_seq = np.where(e_dis_diff < 0)
    #         X1 = X[X1_seq]
    #         X2 = X[X2_seq]

    #         C1 = np.mean(X1, axis=0)
    #         C2 = np.mean(X2, axis=0)
        
    #     # print(X1_seq[0].tolist(), X2_seq[0].tolist())
    #     return X1, X2
    
    def recurse(X, m):
        nonlocal U
        nonlocal m_tot
        n = X.shape[0]
        
        if n <= 0 or m <= 0:
            return
        
        if n == 1: # m >= 1
            m_tot -= m-1 
            U.append(X[0])
            return
        else: # n > 1
            if m == 1:
                U.append(np.mean(X, axis=0))
                return
            else: # m > 1
                n1 = n // 2
                m1 = m // 2
                X1, X2 = balance_divide(X, n1)
                # print(m1, m - m1)
                recurse(X1, m1)
                recurse(X2, m - m1)
        
    recurse(X_tot, m_tot)
    if m_initial > m_tot:
        print(f'Too many anchors. {m_initial} to {m_tot}')
    print(f'{m_tot:d} anchors from {X_tot.shape[0]:d} points with t = {t:d}')
    return np.asarray(U) 

'''
data_root = 'D:\SSL'
import sys
if data_root not in sys.path:
    sys.path.append(data_root)
from data import data_input
from utils import Data
from visualize import scatter2D, show_in_plots

args_0 = {'dim':2, 'class':3, 
          'components':[{'name':'ring', 'kind':0, 'num':400, 'radius':5, 'dist':1},
                        {'name':'ring', 'kind':1, 'num':600, 'radius':10, 'dist':1},
                        {'name':'ring', 'kind':2, 'num':1000, 'radius':15, 'dist':1}]}
X, y, c = data_input.data_input(Data('Toy-3Rings-dist', -1, 0.01, args_0))
scatter2D(X, y, '.')

m = 100
t = 20
U, m = BKHK(X, m, t)

scatter2D(U, 'black', 'x')
show_in_plots('anchor')
'''