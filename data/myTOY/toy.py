# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 08:51:19 2022

@author: Dell
"""
import numpy as np
import pickle
import os

from data import data_base
from matplotlib import pyplot as plt

clist = ['r','g','b','purple','orange','pink','cyan','grey','black']
mlist = ['o', 'v', '^', '<', '>', '1', '2', '3', '4']
def draw(X, y, marker='.', title=None):
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    if type(y) is int:
        label2color = clist[y]
    elif type(y) is str:
        label2color = y
    else:
        label2color = [clist[y_i] for y_i in y]
    ax.scatter(X[:, 0], X[:, 1], c=label2color, marker=marker, s=1)
    # if title:
    #     plt.title(title)
    plt.show()
    # fig.savefig('result/' + title + '.pdf', format='pdf')
    
class Toy(data_base.Data_base):
    # dim = d
    # num = n
    c = name = None
    
    def __init__(self, file_root):
        data_base.Data_base.__init__(self, file_root)
    
    def generate_cluster(self, args):
        if type(args['dist']) not in (tuple, list) :
            args['dist'] = (args['dist'],) * self.dim
        if len(args['center']) != self.dim or len(args['dist']) != self.dim:
            raise Exception('dimension fault!')
        
        return np.random.randn(args['num'], self.dim) * args['dist'] + args['center']
    
    def generate_ring(self, args):
        if self.dim != 2:
            raise Exception('ring dimension!')
        
        degree = args['degree'] if 'degree' in args else (0, 2*np.pi)
        center = args['center'] if 'center' in args else (0, 0)
        scope = args['scope'] if 'scope' in args else 1
        
        phi = np.random.rand(args['num']) * (degree[1] - degree[0]) + degree[0]
        
        if 'width' in args:
            rho = np.random.rand(args['num']) * args['width'] + args['radius'] - args['width']/2
        else:
            rho = np.random.randn(args['num']) * scope + args['radius']
        
        X = (rho * [np.cos(phi), np.sin(phi)]).T + center
        
        if 'rotate' in args:
            a = args['rotate']
            rotate = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            X = X @ rotate
            
        return X
    
    def generate_part(self, args):
        if args['name'] == 'cluster':
            return self.generate_cluster(args)
        elif args['name'] == 'ring':
            return self.generate_ring(args)
        else:
            return None

        
    def generate_data(self, data=None, args=None):
        self.dim = args['dim']
        self.c = args['class']
        
        X = np.zeros((0, self.dim))
        y = []
        
        for comp in args['components']:
            X = np.concatenate((X, self.generate_part(comp)), axis=0)
            y += [comp['kind']] * comp['num']
    
        if data:
            print(f'generate {self.name} which is {X.shape} with {self.c} classes')
            draw(X, y, title=data.name)
        else:
            draw(X, y, title='generate')
        
        return (X, y, args['class'])
    
    def load_data(self, data):
        data_dir = self.file_root + '\\myTOY\\'
        self.name = data.name
        if data.name in os.listdir(data_dir):
            try:
                with open(data_dir + data.name, 'rb') as fd:
                    (X, y, self.c, args) = pickle.load(fd)
            except:
                args = None
            if str(args) == str(data.args):
                y = np.asarray(y)
                print(X.shape, y.shape)
                draw(X, y, title=data.name)
                return X, y, self.c
            else:
                print('different args')
            
        (X, y, self.c) = self.generate_data(data, data.args)
        with open(data_dir + data.name, 'wb') as fd:
            pickle.dump((X, y, self.c, data.args), fd)
        
        y = np.asarray(y)
        # print(X.shape, y.shape)
        
        return X, y, self.c
    
    

# args_2Moons_ =  {'dim':2, 'class':2, 
#       'components':[{'name':'ring', 'kind':0, 'num':450, 'radius':6, 'degree':(0, np.pi), 'center':(6,-3)},
#                     {'name':'ring', 'kind':1, 'num':450, 'radius':6, 'degree':(-np.pi, 0)},
#                     {'name':'cluster', 'kind':2, 'num':100, 'dist':(3,3), 'center':(3, -1.5)},]}
# data_root = 'F:\SSL'
# toy = Toy(data_root)
# toy.generate_data(args=args_2Moons_)

