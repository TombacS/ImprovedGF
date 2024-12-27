# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.svm import SVC
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy import sparse

from functools import partial

class Data:
    def __init__(self, name, total_num, labeled_num=None, args=None, c=None) -> None:
        self.name = name
        self.total_num = total_num
        self.labeled_num = labeled_num
        self.args = args
        self.c = c

class Timer:
    def __init__(self, logger):
        self.time_init = time.time()
        self.time_last = self.time_init
        self.logger = logger
    
    def record(self, tips='last procedure'):
        interval = time.time()-self.time_last
        self.time_last = time.time()
        self.logger.info(tips + f': {interval:.6f} s')
        return interval
    
class MethodBase:
    zero = 1e-32
    inf = 1e32
    dtype = np.float32
    
    def __init__(self):
        pass
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

    def copy(self):
        return self.copy()

    def mate(self, B) -> None:
        np.random.seed(time.time())
        
        def real_trans(x, y):
            k = np.random.rand()
            return k*x+(1-k)*y, (1-k)*x+k*y

        if np.random.rand() > 0.5:
            self.alpha, B.alpha = real_trans(self.alpha, B.alpha)
        if np.random.rand() > 0.5:
            self.beta[0], B.beta[0] = real_trans(self.beta[0], B.beta[0])
        if np.random.rand() > 0.5:
            self.beta[1], B.beta[1] = real_trans(self.beta[1], B.beta[1])
        if np.random.rand() > 0.5:
            self.eta, B.eta = real_trans(self.eta, B.eta)

    def __lt__(self, o) -> bool:
        return self.score > o.score
    
    def get_indicator(self, question, args): # preprocess
        if question.Y is None:
            print('question has not been allocated')
            raise Exception('question has not been allocated')
        Y = np.zeros((question.n, question.c), dtype=question.Y.dtype)
        Y[:question.l] = question.Y
        return (question.n, question.l, question.c, Y, None)
    
    
class kNN(MethodBase):
    def __init__(self, name=None, k=1):
        self.k = k
        if k==1:
            self.name = name or '1NN'
        else:
            self.name = name or 'kNN'
    
    def get_indicator(self, question, args) -> np.ndarray:
        n, l, c, Y, F = MethodBase.get_indicator(self, question, args)
        X = question.X
        
        time_st = time.time()
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(X[:l])
        dist2, knn = neigh.kneighbors(X)
        F = np.average(Y[:l][knn], axis=1)
        
        return F, time.time() - time_st
        
class SVM(MethodBase):
    def __init__(self, name=None, kernel='rbf'):
        self.kernel = kernel
        self.name = name or 'SVM'
        
    def get_indicator(self, question, args) -> np.ndarray:
        n, l, c, Y, F = MethodBase.get_indicator(self, question, args)
        y = question.y
        X = question.X
        
        time_st = time.time()
        F = np.zeros_like(Y)
        clf = SVC(kernel=self.kernel, decision_function_shape='ovr')
        clf.fit(X[:l], y[:l])
        f = clf.predict(X)
        print(f.shape)
        F = np.eye(c, dtype=X.dtype)[f]
        
        return F, time.time() - time_st


def rbf(X1,X2,**kwargs):
    return np.exp(-cdist(X1,X2)**2*kwargs['gamma'])

import sys
sys.path.append('D:\software\Git-project\LAMDA-SSL')

# from LAMDA_SSL.Algorithm.Classification.LapSVM import LapSVM # only for binary
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
# from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
# from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.ReMixMatch import ReMixMatch
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from LAMDA_SSL.Algorithm.Classification.GAT import GAT
from LAMDA_SSL.Algorithm.Classification.GCN import GCN
from LAMDA_SSL.Algorithm.Classification.S4L import S4L

# from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10
# dataset = CIFAR10(root='..\Download\cifar-10-python',
#                   labeled_size=4000, stratified=False, shuffle=True, download=True)
# labeled_X=dataset.labeled_X
# labeled_y=dataset.labeled_y
# unlabeled_X=dataset.unlabeled_X
# test_X=dataset.test_X
# test_y=dataset.test_y
# print(labeled_X.shape)
# print(len(labeled_y))
# print(unlabeled_X.shape)
# print(test_X.shape)
# print(len(test_y))

# from LAMDA_SSL.Dataset.Graph.Cora import Cora
# dataset=Cora(labeled_size=0.2,root='..\Download\Cora',random_state=0,default_transforms=True)
# from LAMDA_SSL.Transform.Graph.NormalizeFeatures import NormalizeFeatures
# transform = NormalizeFeatures
# data = dataset.transform.fit_transform(dataset.data)
# print(data.x.shape)
# print(data.y.shape)
# print(np.sum(data.labeled_mask.numpy()))
# print(np.sum(data.train_mask.numpy()))
# print(np.sum(data.unlabeled_mask.numpy()))
# print(np.sum(data.val_mask.numpy()))
# print(np.sum(data.test_mask.numpy()))
# print(data.edge_index.shape)

# from LAMDA_SSL.Distributed.DataParallel import DataParallel
# parallel=DataParallel(device_ids=['cuda:0','cuda:1'],output_device='cuda:0')

# model=FixMatch(threshold=0.95,lambda_u=1.0,mu=7,T=0.5,epoch=1,num_it_epoch=2**20,num_it_total=2**20)

# model=Tri_Training()
# model=UDA()

import torch
def knn_to_edge_index(knn):
    num_points, k = knn.shape
    
    src = np.repeat(np.arange(num_points), k)  # 源节点索引，每个点重复k次
    dst = knn.flatten()                        # 目标节点索引，展平后的knn数组
    
    edge_index = np.stack((src, dst))
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    return edge_index_tensor


from math import sqrt
class SSL(MethodBase):
    def __init__(self, name=None):
        self.name = name
        
    def get_indicator(self, question, args):
        n, l, c, Y, F = MethodBase.get_indicator(self, question, args)
        time_st = time.time()
        
        d = question.X[:l].shape[1]
        def decompose(n):
            sqrt_n = int(sqrt(n))
            for i in range(sqrt_n, 0, -1):
                if n % i == 0:
                    return (i, n // i)
            return (1, n)
            
        d_ = decompose(d)
        labeled_X = question.X[:l].reshape(l, d_[1], d_[0], 1)
        labeled_X = np.repeat(labeled_X, 3, axis=3)
        labeled_y = list(question.y[:l])
        unlabeled_X = question.X[l:].reshape(n-l, d_[1], d_[0], 1)
        unlabeled_X = np.repeat(unlabeled_X, 3, axis=3)
        unlabeled_y = list(question.y[l:])
        
        print(labeled_X.shape)
        print(len(labeled_y))
        print(unlabeled_X.shape)
        
        iter_length = 2**10
        dim_in = question.X.shape[1]
        if self.name == 'UDA':
            model = UDA(num_classes=c, threshold=0.95,lambda_u=1.0,T=0.5,epoch=1,num_it_epoch=iter_length,num_it_total=iter_length,verbose=True,eval_it=2**9)
            model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X, valid_X=unlabeled_X, valid_y=unlabeled_y)
        elif self.name == 'ICT':
            model = ICT(lambda_u=1.0,epoch=1,num_it_epoch=iter_length,num_it_total=iter_length,verbose=True,eval_it=2**9)
            model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X, valid_X=unlabeled_X, valid_y=unlabeled_y)
        elif self.name == 'FixMatch':
            model=FixMatch(threshold=0.95,lambda_u=1.0,T=0.5,epoch=1,num_it_epoch=iter_length,num_it_total=iter_length,verbose=True,eval_it=2**9)
            model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X, valid_X=unlabeled_X, valid_y=unlabeled_y)   
        elif self.name == 'MixMatch':
            model=MixMatch(lambda_u=1.0,T=0.5,epoch=1,num_it_epoch=iter_length,num_it_total=iter_length,verbose=True,eval_it=2**9)
            model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X, valid_X=unlabeled_X, valid_y=unlabeled_y)   
        elif self.name == 'GCN':
            model=GCN(dim_in=dim_in, 
                      num_classes=c,
                      normalize=True,
                      epoch=4096,
                      weight_decay=5e-4)
            labeled_mask = torch.zeros(n, dtype=torch.bool)
            labeled_mask[:l] = True
            unlabeled_mask = torch.zeros(n, dtype=torch.bool)
            unlabeled_mask[l:] = True
            neigh = NearestNeighbors(n_neighbors=20) #exclude itself
            neigh.fit(question.X)
            dist, knn = neigh.kneighbors(question.X)
            edge_index = knn_to_edge_index(knn)
            model.fit(X=question.X,y=question.y,edge_index=edge_index,
                  labeled_mask=labeled_mask, unlabeled_mask=unlabeled_mask,
                  train_mask=torch.ones(n, dtype=torch.bool), test_mask=unlabeled_mask, valid_mask=unlabeled_mask)
        elif self.name == 'GAT':
            model=GAT(dim_in=dim_in, 
                      num_classes=c,
                      epoch=4096,
                      weight_decay=5e-4)
            labeled_mask = torch.zeros(n, dtype=torch.bool)
            labeled_mask[:l] = True
            unlabeled_mask = torch.zeros(n, dtype=torch.bool)
            unlabeled_mask[l:] = True
            neigh = NearestNeighbors(n_neighbors=20) #exclude itself
            neigh.fit(question.X)
            dist, knn = neigh.kneighbors(question.X)
            edge_index = knn_to_edge_index(knn)
            model.fit(X=question.X,y=question.y,edge_index=edge_index,
                  labeled_mask=labeled_mask, unlabeled_mask=unlabeled_mask,
                  train_mask=torch.ones(n, dtype=torch.bool), test_mask=unlabeled_mask, valid_mask=unlabeled_mask)
        elif self.name == 'S4L':
            model=S4L(epoch=1,num_it_epoch=iter_length,num_it_total=iter_length,verbose=True,eval_it=2**9)
            model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X, valid_X=unlabeled_X, valid_y=unlabeled_y)   
        else:
            model = eval(self.name)()
            model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
        
        pred_y = model.predict(X=unlabeled_mask)
        
        # print(pred_y)
        
        f = np.array(np.hstack((labeled_y, pred_y))).astype(int)
        # print(f.shape)
        F = np.eye(c, dtype=question.X.dtype)[f]
        return F, time.time() - time_st
        

'''
class AGR(MethodBase):
    def __init__(self, name=None):
        self.name = name or 'AGR'
        self.U = None
        self.k = None
        
    def get_indicator(self, question, args) -> np.ndarray:
        n, l, c, Y, F = MethodBase.get_indicator(self, question, args)
        X = question.X
        self.k = args['k']
        self.U, m, time_for_U = question.build_anchors(args)
        
        time_st = time.time()
        neigh = NearestNeighbors(n_neighbors=args['k']+1)
        neigh.fit(self.U)
        dist2, knn = neigh.kneighbors(X)
        
        Z = np.array([self.solve(X[i].T, self.U[knn[i]]) for i in range(n)])
        
        return
    
    def solve(self, x, U): # x: [d, 1], U: [k, d]
        z = np.ones((self.k, 1), dtype=self.dtype) / self.k #[k, 1]
        z_last = np.copy(z)        
        dilta_last = 0
        dilta_now = 1
        beta = 1
        t = 0
        
        def g(z): # -> [1]
            return np.sum((x - U.T @ z)**2)/2
        
        def dg(z): # -> [k, 1]
            return U @ U.T @ z - U @ x
        
        def gbvz(v, b): # -> [1]
            return g(v) + np.sum(dg(z) * (z - v)) + b * np.sum((z-v)**2) / 2
        
        def simplexProjection(z): # [k, 1] -> [k, 1]
            z = np.zeros((self.k,1))    
            v = z.reshape((-1)).tolist()
            v.sort(reverse=True)
            
            v_sum = 0
            i = 0
            res = 1
            while True:
                v_sum += v[i]
                res = v[i] - (v_sum-1)/(i+1)
                if res <= 0:
                    break
                i += 1
            v_sum -= v[i]
            theta = (v_sum-1) / i
            return np.max(z-theta, 0)
    
        t = t + 1
        alpha = (dilta_last - 1) / dilta_now
        v = z + alpha * (z - z_last)
    
        
        gz = g(z)
        while gz > gbvz(v, b):
            b *= 2
        
        
        d_gz = U @ U.T @ z - U @ x #[d, 1]
        
class LapSVM(MethodBase):
    def __init__(self, opt):
        if opt:
            self.opt = opt
        else:
            self.opt = {'n_neighbor'   : 5,
                        't':            1,
                        'kernel_function':rbf,
                        'kernel_parameters':{'gamma':10},
                        'gamma_A':0.03125,
                        'gamma_I':10000}
        self.name = 'LapSVM'

    def get_indicator(self, question, args) -> np.ndarray:
        n, l, c, Y, F = MethodBase.get_indicator(self, question, args)
        u = n-l
        X, y = question.X, question.y[:l]
        W, time_for_W = question.build_original_graph(args)
        time_st = time.time()
        
        d = np.sum(W, axis=0)
        L = np.diag(d) - W
        
        # Computing K with k(i,j) = kernel(i, j) [nxn]
        K = self.opt['kernel_function'](X, X,**self.opt['kernel_parameters'])
        # Creating matrix J [I (l x l), 0 (l x (l+u))] [lxn]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)
        # Computing "almost" alpha [nxl]
        almost_alpha = np.linalg.inv(2 * self.opt['gamma_A'] * np.identity(n) \
                                     + ((2 * self.opt['gamma_I']) / n ** 2) * L @ K) @ J.T 
                  
        # ===== Objectives ===== beta:[]
        def objective_func(Q, beta):
            return (1 / 2) * beta.dot(Q).dot(beta) - np.ones(l).dot(beta)

        def objective_grad(Q, beta):
            return np.squeeze(np.asarray(beta.T.dot(Q) - np.ones(l)))
            
        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]
        
        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(y, beta):
            return beta.dot(y)

        def constraint_grad(y, beta):
            return y
        
        cons = {'type': 'eq', 'fun': partial(constraint_func, y), 'jac': partial(constraint_grad, y)}
        
        # Finding optimal decision boundary b using labeled data [nxl]
        new_K_l = self.opt['kernel_function'](X, X[:l], **self.opt['kernel_parameters'])
        
        F = np.zeros((n ,c))
        for i in range(c):
            y_i = 2 * (y == i).astype(int) - 1
            y_diag = np.diag(y_i)
            # Computing Q [lxl]
            Q = y_diag @ J @ K @ almost_alpha @ y_diag
            Q = (Q+Q.T)/2
            
            # ===== Solving =====
            x0 = np.zeros(l)
    
            beta_hat = minimize(partial(objective_func, Q), x0,\
                                jac=partial(objective_grad, Q), constraints=cons, bounds=bounds)['x']
    
            # Computing final alpha [nx1]
            self.alpha = almost_alpha @ y_diag @ beta_hat
    
            f_l = np.squeeze(np.asarray(self.alpha)).dot(new_K_l)
            self.sv_ind = np.nonzero((beta_hat>1e-7)*(beta_hat<(1/l-1e-7)))[0]

            ind = self.sv_ind[0]
            self.b = np.diag(Y)[ind]-f_l[ind]
        
            new_K = self.opt['kernel_function'](X, X, **self.opt['kernel_parameters'])
            f = np.squeeze(np.asarray(self.alpha)).dot(new_K) + self.b
            
            F[:, i] = f
            
        return F, time.time() - time_st + time_for_W
'''  