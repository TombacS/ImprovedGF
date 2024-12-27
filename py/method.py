# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import time
from utils import MethodBase

# import scipy

class Mtd(MethodBase): 

    def __init__(self, name, alpha=None, beta=None, eta=np.inf, acclr='origin', H=False):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.H = H
        self.acclr = acclr
        
        self.score = 0

    def __str__(self):
        if self.name == "":
            self.name = f'{self.alpha:},({self.beta[0]:},{self.beta[1]:}),{self.eta:}'
        return self.name
    
    def test_det(self, L, yes=False):
        if yes:
            a = np.exp(np.average(np.log(np.diag(L))))
            try:
                det = np.linalg.det(L/a)
                if det < self.zero:
                    print(f'Warning: may be singular, det = {det}')
            except:
                print('Warning: det failed')
    
    def get_indicator(self, question, args) -> np.ndarray:
        n, l, c, Y, F = MethodBase.get_indicator(self, question, args)
                
        d_beta = np.zeros((n))
        if self.beta:
            if type(self.beta) == str:
                self.beta = eval(self.beta)
            if type(self.beta) in (tuple, list):
                d_beta[:l] += self.beta[0]
                d_beta[l:] += self.beta[1]
            else:
                d_beta += self.beta
                
        if self.acclr[0] in ('a', 'A'):
            B, m, time_for_UB = question.build_anchored_graph(args)
            time_st = time.time()
            d = B @ np.sum(B, axis=0).T # d = np.ones(self.n)
            if self.alpha:
                d *= self.alpha
                    
            if self.H:
                F = np.copy(Y)
                
                c = (1 / d)[l:]
                Y = B[l:] @ (B[:l].T @ Y[:l])
                base_part = (Y.T * c).T
                left_part = (B[l:].T * c).T
                inv_part = (B[l:].T * c) @ B[l:]
                inv_part[np.arange(m), np.arange(m)] -= 1
                
                F[l:] = base_part - left_part @ (np.linalg.inv(inv_part) @ (left_part.T @ Y))
            
            else: # sparse
                Y = sp.csr_matrix(Y)
                
                B = sp.csr_matrix(B, shape=(n, m))
                B_T = B.transpose().tocsr()
                data = np.append(B_T.data, [1]*n)
                indices = np.append(B_T.indices, np.arange(n))
                indptr = np.append(B_T.indptr, B_T.data.shape[0] + n)
                right_part = sp.csr_matrix((data, indices, indptr), shape=(m+1, n))
                
                inv_part = np.zeros((m+1, m+1))
                inv_part[:m, :m] = (B.transpose() @ B).todense()
                inv_part[np.arange(m), np.arange(m)] -= 1+self.beta
                b = np.sum(B, axis=0)
                inv_part[:m, -1] = b
                inv_part[-1, :m] = b
                inv_part[-1, -1] = n
                inv_part_inv = np.linalg.inv(inv_part)
                F = Y - right_part.transpose() @ (inv_part_inv @ (right_part @ Y))
                
                # left_part = np.hstack((B, np.ones((n, 1))))
                # inv_part = np.zeros((m+1, m+1))
                # inv_part[:m, :m] = B.T @ B
                # inv_part[np.arange(m), np.arange(m)] -= 1+self.beta
                # b = np.sum(B, axis=0)
                # inv_part[:m, -1] = b
                # inv_part[-1, :m] = b
                # inv_part[-1, -1] = n
                # inv_part_inv = np.linalg.inv(inv_part)
                # F = (Y - left_part @ (inv_part_inv @ (left_part.T @ Y)))
                    
            time_for_UBF = time.time() - time_st + time_for_UB
        
        elif self.acclr[0] in ('g', 'G'):
            W, time_for_W = question.build_original_graph(args)
            time_st = time.time()
            W = W.todense().astype(np.float64)
            d = np.sum(W, axis=0) # d = np.ones(self.n)
            d = np.asarray(d)[0]
            if self.alpha:
                d *= self.alpha
            
            if self.H:
                L = np.diag(d) - W
                F = np.copy(Y)
                F[l:] = - np.linalg.solve(L[l:,l:], (L[l:, :l] @ F[:l]))
            else:
                L = np.diag(d+d_beta) - W
                del W
                del d
                # self.test_det(L+1)
                if self.eta > self.inf:
                    self.eta = 1e6
                    L -= d_beta / n
                try:
                    F = np.linalg.solve((L + self.eta), Y)
                    # each line except the first is minused by the first line for sparsity
                    # for _ in range(1,n):
                    #     L[_,:] -= L[0,:]
                    #     Y[_,:] -= Y[0,:]
                    # L[0,:] += self.eta
                    # F = sp.linalg.spsolve(L.tocsc(), Y.tocsc())
                    # F[1:,:] += F[0,:].todense()
                except:
                    print('solve failed')
                    
            time_for_UBF = time.time() - time_st + time_for_W
            
        else:
            W, time_for_W = question.build_original_graph(args)
            time_st = time.time()
            d = np.sum(W, axis=0)
            d = np.asarray(d)[0]
            if self.alpha:
                d *= self.alpha
            
            if self.H: #sparse
                # L = sp.csr_matrix((d, (np.arange(n), np.arange(n)))) - W
                L = np.diag(d) - W.todense()
                del W
                del d
                # self.test_det(L[l:, l:])
                F = Y
                try:
                    # F[l:] = - sp.linalg.spsolve(L[l:,l:].tocsc(), (L[l:, :l] @ Y[:l]).tocsc())
                    # F[l:] = - sp.linalg.inv(L[l:,l:].tocsc()) @ (L[l:, :l] @ Y[:l])
                    F[l:] = - np.linalg.inv(L[l:,l:]) @ (L[l:, :l] @ Y[:l])
                except:
                    print('H failed')
            else:
                # L = sp.csr_matrix((d+d_beta, (np.arange(n), np.arange(n)))) - W
                L = np.diag(d+d_beta) - W.todense()
                del W
                del d
                # self.test_det(L+1)
                if self.eta > self.inf:
                    L -=  d_beta / n
                    del d_beta
                    try:
                        # Y = sp.csc_matrix(Y)
                        # F = sp.linalg.pinv(L) @ Y # not stable
                        # F = np.linalg.pinv(L) @ Y # not stable
                        U, sigma, U_ = np.linalg.svd(L)
                        if np.amin(sigma[-2]) < 1e-6:
                            print(f'very small sigma: {sigma[-10:]}')
                        sigma = np.asarray(list(map(lambda x : 1/x if x > 1e-4 else 0, sigma)))
                        F = np.asarray(U) * sigma @ (U.T @ Y)
                        
                    except:
                        print('svd failed')
                        F = Y
                else: # sparse
                    del d_beta
                    # L = L.tocsc() #csc for LU composition
                    # Y = Y.tocsc()
                    # F = sp.linalg.inv(L + self.eta) @ Y
                    # # F = sp.linalg.spsolve(L+self.eta, Y)
                    # F = F.todense()
                    F = np.linalg.inv(L + self.eta) @ Y
            time_for_UBF = time.time() - time_st + time_for_W
        # print(F.shape)
        return F, time_for_UBF
   