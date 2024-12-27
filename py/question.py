# -*- coding: utf-8 -*-

import time
import sys
import numpy as np
import pickle
import os

# import sys
# data_root = "D:\data\SSL"
# if data_root not in sys.path:
#     sys.path.append(data_root)

from BKHK import BKHK
# from visualize import scatterByLabels2D_graph as draw
from visualize import scatterByInd2D_graph as draw
from visualize import scatterKNN_graph as draw_knn

from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

class Que: #name, X, Y, y, c, n, l, l_of,each
    dtype = np.float32

    def __init__(self, data, data_root, logger, seed=None, if_balance=True):
        if data_root not in sys.path:
            sys.path.append(data_root)
        self.name = data.name
        self.logger = logger
        from data import data_input
        self.data_root = data_root
        if 'ckpt' not in os.listdir(self.data_root):
            os.mkdir(self.data_root + '\\ckpt')
        self.ckpt_root = data_root + '\\ckpt\\' + data.name + '\\'
        if data.name not in os.listdir(self.data_root + '\\ckpt'):
            os.mkdir(self.ckpt_root)
        
        self.X, self.y, self.c = data_input.data_input(data)
        self.real_y = self.y
        self.y = self.y % self.c
        self.Y = None
        # print(self.X.dtype, self.y.dtype)
        
        self.n, self.l = data.total_num, data.labeled_num
        if self.n and (self.n != self.X.shape[0]):
            raise Exception(f'number not match, n:{self.n}, X.size:{self.X.shape}')
        self.n = self.X.shape[0]
        if self.l < 1: # l is a rate  
            self.logger.info(f'with rate {self.l:.4}, l = {self.n} * {self.l} = {int(self.n * self.l)}')
            self.l = int(self.n * self.l)
        if self.l > self.n:
            raise Exception('l cannot be bigger than n')
        
        self.l_of_each = -1
        if if_balance:
            if self.l % self.c != 0:
                logger.debug('l is indivisible by c')
            self.l_of_each = self.l // self.c
            self.logger.debug(f'{data.name:}, {self.c:}*{self.l_of_each:} of {self.n:} within {self.X.shape[1]:}-d')        
        else:
            self.logger.debug(f'{data.name:}, {self.l} of {self.n:} within {self.X.shape[1]:}-d')        

        if seed != None:
            self.allocate(seed)
        
    def allocate(self, seed, error_prop=0, coding=1) -> None:
        self.seed = seed
        np.random.seed(seed)
        perm = np.random.permutation(self.n)
        self.X = self.X[perm]
        self.y = self.y[perm]
        self.real_y = self.real_y[perm]
        
        if self.l_of_each > 0:
            seq = self.balanceLabeled()
            self.X, self.y = self.X[seq], self.y[seq]
            self.real_y = self.real_y[seq]
            
        if error_prop > 0:
            noise_num = int(self.l*error_prop)
            noise_idx = np.random.choice(self.l, noise_num, replace=False)
            noise_values = np.random.choice(self.c, noise_num)
            self.y[noise_idx] = noise_values
        
        if len(self.y.shape) > 1:
            self.Y = self.y[:self.l]
        else:
            self.Y = np.eye(self.c, dtype=self.X.dtype)[self.y[:self.l]]
        
        if coding == 0:
            self.Y = 2 * self.Y - 1
        return
        # print(self.y.shape, self.Y.shape)
        
    def balanceLabeled(self) -> list:
        count = [0] * self.c
        l_seq = [0] * 20 * self.c
        u_seq = []
        
        for i in range(self.n):
            # if count[self.y[i]] < self.l_of_each:
            if count[self.y[i]] < 20:
                l_seq[self.c * count[self.y[i]] + self.y[i]] = i
                count[self.y[i]] += 1
            else:
                u_seq.append(i)
        # print(count)
        return l_seq + u_seq
    
    def build_anchored_graph(self, args) -> (np.ndarray, int):
        s, n, k, m, t = self.seed, self.n, args['k'], args['m'], args['t']
        if k == np.inf:
            k = m - 1
        if m > n:
            m = n
            
        try:
            # load B from storage
            with open(self.ckpt_root + f's{s}_n{n}_m{m}_t{t}_k{k}_sB', 'rb') as fd:
                B, m, time_for_UB = pickle.load(fd)
                # print(f'load B from ckpt, it takes {time_for_UB:.6f}s')
                return B, m, time_for_UB
        except:
            try:
                # load U from storage
                with open(self.ckpt_root + f's{s}_n{n}_m{m}_t{t}_U', 'rb') as fd:
                    U, m, time_for_U = pickle.load(fd)
                    # print(f'load U from ckpt, it takes {time_for_U:.6f}s')
            except:
                # build U and store
                time_st = time.time()
                U = BKHK(self.X, m, t)
                time_for_U = time.time() - time_st
                # self.logger.debug(f'store U into ckpt, it takes {time_for_U:.6f}s')
                with open(self.ckpt_root + f's{s}_n{n}_m{m}_t{t}_U', 'wb') as fd:
                    pickle.dump((U, m, time_for_U), fd)
            finally:
                # build B and store
                time_st = time.time()
                B = self.generate_B(U, m, k)
                time_for_UB = time.time() - time_st + time_for_U
                # self.logger.debug(f'store U&B into ckpt, it takes {time_for_UB:.6f}s')
                try:
                    with open(self.ckpt_root + f's{s}_n{n}_m{m}_t{t}_k{k}_sB', 'wb') as fd:
                        pickle.dump((B, m, time_for_UB), fd)
                except:
                    pass
        return B, m, time_for_UB
    
    def build_original_graph(self, args) -> np.ndarray:
        s, n, k, l = self.seed, self.n, args['k'], self.l
        if k == np.inf:
            k = n - 2
        
        try:
            # load W from storage
            with open(self.ckpt_root + f's{s}_n{n}_k{k}_l{l}_sW', 'rb') as fd:
                W, time_for_W = pickle.load(fd)
                # print(f'load W from ckpt, it takes {time_for_W:.6f}s')
                if type(W) == str:
                    raise Exception('Too large of {W.shape} matrix')
                return W, time_for_W
        except:
            # build W and store
            time_st = time.time()
            W = self.generate_W(n, k)
            time_for_W = time.time() - time_st
            try:
                with open(self.ckpt_root + f's{s}_n{n}_k{k}_l{l}_sW', 'wb') as fd:
                    if self.name[:3] != 'Toy':
                        pickle.dump((W, time_for_W), fd)
                        self.logger.debug(f'store W into ckpt, it takes {time_for_W:.6f}s')
            except:
                self.logger.debug('Failed. Too large to store.')
                with open(self.ckpt_root + f's{s}_n{n}_k{k}_l{l}_sW', 'wb') as fd:
                    pickle.dump(('Failed. Too large to store.', time_for_W), fd)
                # raise Exception('Too large of {W.shape} matrix')

        return W, time_for_W
    
    def build_anchors(self, args):
        s, n, k, m, t = self.seed, self.n, args['k'], args['m'], args['t']
        if k == np.inf:
            k = m - 1
        if m > n:
            m = n

        try:
            # load U from storage
            with open(self.ckpt_root + f's{s}_n{n}_m{m}_t{t}_U', 'rb') as fd:
                U, m, time_for_U = pickle.load(fd)
                # print(f'load U from ckpt, it takes {time_for_U:.6f}s')
        except:
            # build U and store
            time_st = time.time()
            U = BKHK(self.X, m, t)
            time_for_U = time.time() - time_st
            # self.logger.debug(f'store U into ckpt, it takes {time_for_U:.6f}s')
            with open(self.ckpt_root + f's{s}_n{n}_m{m}_t{t}_U', 'wb') as fd:
                pickle.dump((U, m, time_for_U), fd)
        return U, m, time_for_U
        
    def generate_B(self, U, m, k) -> np.ndarray:
        if m < k+1:
            print('error: m<k+1, m={m}, k={k}')
            k = m - 1
        
        neigh = NearestNeighbors(n_neighbors=k+1)
        neigh.fit(U)
        dist, knn = neigh.kneighbors(self.X)
        
        v = dist[:,k].reshape(self.n,1) - dist[:,:k]
        v_sum = np.sum(v, axis=1).reshape(self.n)
        v[v_sum==0] = 1 # for all zero
        v = v / np.sum(v, axis=1).reshape(self.n,1)
        
        v = v.reshape(self.n*k)
        col = knn[:,:k].reshape(self.n*k)
        row = np.asarray([[_]*k for _ in range(self.n)]).reshape(self.n*k)
        
        Z = sp.csr_matrix((v, (row, col)), shape=(self.n, m))
        delta = np.sum(Z, axis=0)
        if np.amin(delta) == 0:
            non_zero = np.nonzero(delta)[1]
            self.logger.debug(f'unused anchor, m: {m:}->{non_zero.shape[0]:}')
            m = non_zero.shape[0]
            Z = Z[:, non_zero]
            delta = delta[:, non_zero]
        
        v = 1/np.sqrt((np.asarray(delta)[0]))
        Delta = sp.csr_matrix((v, (np.arange(m), np.arange(m))), shape=(m,m))
        return Z @ Delta
        
        # Z = np.zeros((self.n, m), dtype=self.dtype)
        # for i in range(self.n):
        #     d = dist[i][k] - dist[i][:k]
        #     d_sum = np.sum(d)
        #     if d_sum == 0:
        #         d = [1/k] * k
        #     else:
        #         d = d / d_sum
        #     Z[i][knn[i][:k]] = d
            
        # delta = np.sum(Z, axis=0)
        # non_zero = np.nonzero(delta)
        # if m != non_zero[0].shape[0]:
        #     self.logger.debug(f'unused anchor, m: {m:}->{non_zero[0].shape[0]:}')
        #     m = non_zero[0].shape[0]
        #     Z = Z[:, non_zero[0]]
        #     delta = delta[non_zero[0]]
        # return Z / np.sqrt(delta)
    
    def generate_W(self, n, k, c='rbf') -> np.ndarray:
        
        if k == 0 or k > n-2:
            # self.X = np.asarray(self.X, dtype=np.float64)
            dist2 = np.sum(self.X**2, axis=1).reshape(-1, 1)\
                 - 2 * self.X @ self.X.T + np.sum(self.X**2, axis=1)
            gamma = 1 / self.X.shape[1] / np.var(self.X)
            W = np.exp(-gamma * dist2)
            W[np.arange(n), np.arange(n)] = 0
            W = np.asarray(W, dtype=np.float32) / n
            
        else:
            neigh = NearestNeighbors(n_neighbors=k+2) #exclude itself
            neigh.fit(self.X)
            dist, knn = neigh.kneighbors(self.X)
            dist = dist[:, 1:]
            knn = knn[:, 1:]
            
            # if self.X.shape[1] == 2:
                # draw_knn(self.X, self.y, self.l, knn, title=self.name)
            
            if c == 'rbf':
                gamma = 1 / self.X.shape[1] / np.var(dist**2)
                data = np.exp(-gamma * dist[:,:k]**2)
                col_ = knn[:,:k].reshape(self.n*k)
                row_ = np.asarray([[_]*k for _ in range(self.n)]).reshape(self.n*k)
                
                data = np.append(data, data)
                col = np.append(col_, row_)
                row = np.append(row_, col_)
                
                W = sp.coo_matrix((data, (row, col)), shape=(self.n, self.n)).tocsr()
            else:
                v = dist2[:,k].reshape(self.n,1) - dist2[:,:k]
                v_sum = np.sum(v, axis=1).reshape(self.n)
                v[v_sum==0] = 1 # for all zero
                v = v / np.sum(v, axis=1).reshape(self.n,1)
                
                v = v.reshape(self.n*k)
                col = knn[:,:k].reshape(self.n*k)
                row = np.asarray([[_]*k for _ in range(self.n)]).reshape(self.n*k)
                
                Z = sp.csr_matrix((v, (row, col)), shape=(self.n, self.n))
                delta = np.sum(Z, axis=0)
                if np.amin(delta) == 0:
                    self.logger.debug('isolid point')
                    delta[delta == 0] = 1
                    
                v = 1/np.sqrt((np.asarray(delta)[0]))
                Delta = sp.csr_matrix((v, (np.arange(n), np.arange(n))), shape=(n,n))
                B = Z @ Delta
                W = (B + B.transpose()) / 2
            
        return W
    
    def label_margin(self, mtd_name, indicator):
        l = self.l
        ind_l = indicator[:l]
        y_l = self.real_y[:l]
        ind_u = indicator[l:]
        y_u = self.real_y[l:]
        
        ret_l = []
        ret_u = []
        
        for i in range(self.c):
            ret_l.append(np.average(ind_l[y_l%self.c==i,i]) - np.average(ind_l[y_l%self.c!=i,i]))
            ret_u.append(np.average(ind_u[y_u%self.c==i,i]) - np.average(ind_u[y_u%self.c!=i,i]))
            
        ret_l = np.average(np.asarray(ret_l))
        ret_u = np.average(np.asarray(ret_u))
        return ret_l, ret_u
    
    def evaluate(self, mtd_name, indicator, if_F1=None, if_draw=True, if_detail=False):
        time_st = time.time()
                
        if len(indicator.shape) == 2: #[nxc]
            predict = np.argmax(indicator, axis=1)
            predict = np.asarray(predict).reshape(self.n)
        else: #[nx1]
            predict = indicator
            
        credible = (self.real_y == self.y)
        predict_l = predict[:self.l][credible[:self.l]]
        predict_u = predict[self.l:][credible[self.l:]]
        y_l = self.real_y[:self.l][credible[:self.l]]
        y_u = self.real_y[self.l:][credible[self.l:]]
        
        if predict_l.shape[0] > 0:
            l_acc = np.count_nonzero(predict_l==y_l) / predict_l.shape[0]
        else:
            l_acc = 1
        u_acc = np.count_nonzero(predict_u==y_u) / predict_u.shape[0]
        
        if if_draw and self.X.shape[1] == 2:
            try:
                predict[:self.l] = self.y[:self.l]
                # draw(self.X, predict, self.l, mtd_name)
                draw(self.X, indicator, self.l, mtd_name)
            except:
                pass 
            
        ret = [l_acc, u_acc]
        
                
        if if_F1:
            TPs, Ps, P_s = [], [], []
            for i in range(indicator.shape[1]):
                P = predict_u[y_u == i] #(TP + FN), predict T
                P_ = y_u[predict_u == i] #(TP + FP), y T
                TP = P[P == i]
                TP_ = P_[P_ == i]
                if len(TP) != len(TP_):
                    print('TP error')
                    raise Exception()
                TPs.append(len(TP))
                Ps.append(len(P))
                P_s.append(len(P_))
                
            TPs = np.asarray(TPs)
            Ps = np.asarray(Ps)
            P_s = np.asarray(P_s)
                
            if if_F1 == 'micro':
                precision = np.sum(TPs) / np.sum(Ps)
                recall = np.sum(TPs) / np.sum(P_s)
            else: #'macro'
                non_zero = Ps.nonzero()
                precision = np.average(TPs[non_zero] / Ps[non_zero])
                non_zero = P_s.nonzero()
                recall = np.average(TPs[non_zero] / P_s[non_zero])
                
            ret.append(2 * precision * recall / (precision + recall))
            
            ret_l, ret_u = self.label_margin(mtd_name, indicator)
            ret.append(ret_l)
            ret.append(ret_u)
        
        time_for_acc = time.time() - time_st
        return ret, time_for_acc
    




