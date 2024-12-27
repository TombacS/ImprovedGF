# -*- coding: utf-8 -*-

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import logging
# import time

# from GA import GA 
data_root = ".."

from utils import Data, Timer, kNN, SVM, SSL
from question import Que
from method import Mtd


import visualize 

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('accs.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
if len(logger.handlers) == 0:
    logger.addHandler(fh)
    logger.addHandler(ch)
timer = Timer(logger)

args_0 = {'dim':2, 'class':2, 
      'components':[{'name':'ring', 'kind':0, 'num':500, 'radius':5},
                    {'name':'ring', 'kind':1, 'num':1000, 'radius':15},]}
args_1 = {'dim':2, 'class':2, 
      'components':[{'name':'ring', 'kind':0, 'num':500, 'radius':5, 'scope':2},
                    {'name':'ring', 'kind':1, 'num':1000, 'radius':15, 'scope':2},]}
args_2Moons =  {'dim':2, 'class':2, 
      'components':[{'name':'ring', 'kind':0, 'num':500, 'radius':6, 'degree':(0, np.pi), 'center':(6,-2)},
                    {'name':'ring', 'kind':1, 'num':500, 'radius':6, 'degree':(-np.pi, 0)},]}

def compare_with_methods(data_set=None, mtds=None, test_num=1) -> None:
    data_set = data_set or [
        Data('Toy-2Rings0', None, 2*10, args=args_0),
        # Data('Toy-2Rings1', None, 2*10, args=args_1),
        # Data('Toy-2Moons', None, 2*10, args=args_2Moons),
        # Data('Balance', 625, 3*10), #1e-4
        # Data('MobileKSD', 2856, 56*10), #1e-3
        Data('USPS', 9298, 10*10), #1e-4
        # Data('CsMap', 10845, 6*10), #1e-3
        # Data('PhishingWeb', 11055, 2*10), #1e-3
        # Data('Swarm', 24016, 4*10), #1e-5
        # Data('Steel', 35040, 3*10), #1e-4
        # Data('CreditCard', 30000, 2*10), #1e-3
        # Data('Mnist', 70000, 10*10),
        # Data('SensIT', 98528, 2*10),
        # Data('EMnist', 814255, 47*10),
        ]
    timer.record(f'set {len(data_set):} datas')
        
    mtds = mtds or [
            kNN(),
            # SVM(kernel='poly'),
            SVM(kernel='rbf'),
            # Mtd('LLGC', beta=1, eta=0, acclr='g'),
            # Mtd('HF', H=True, acclr='o'),
            # Mtd('GF0', beta='0', acclr='o'),
            # Mtd('GF1', beta='n*0.00001', acclr='o'),
            Mtd('GF(G)', beta='n*0.00001', acclr='g'),
            Mtd('GF(A)', beta='n*0.00001', acclr='a'),
            # SSL('ICT'),
            SSL('GCN'),
            # SSL('GAT'),
            ]
    timer.record(f'set {len(mtds):} methods')
        
    args = {'k': 20, 'm': 2048, 't': 10}
    
    logger.info(f'args:{args}')
    for data in data_set:
        if data.name == 'Balance':
            args['m'] = 21
        elif data.name == 'MobileKSD':
            args['m'] = 2856
        elif data.name in ('PhishingWeb', 'SensIT'):
            args['m'] = 50
        elif data.name == 'EMnist':
            args['m'] = 2048
        else:
            args['m'] = 1024
        accs = {mtd:[] for mtd in mtds} 
        question = Que(data, data_root, logger)
        timer.record(f'question {question.name}')
        for seed in range(test_num):
            question.allocate(seed, coding=0)
            logger.debug(f'allocate with seed:{seed}')
            for mtd in mtds:
                ind, time_for_UBF = mtd.get_indicator(question, args)
                acc, time_for_acc = question.evaluate(mtd.name, ind, if_F1='macro')
                time_for_all = time_for_UBF + time_for_acc
                # except:
                #     logger.debug(f'To solve {data.name} with {mtd.name} failed')
                #     acc = (np.nan, np.nan, np.nan)
                #     time_for_all = (np.nan)
                accs[mtd].append((acc[0], acc[1], acc[2], acc[3], acc[4], time_for_all))
                logger.info(f'{mtd.name:8}: {acc[0]*100:6.2f}%:{acc[1]*100:6.2f}%, F1 = {acc[2]:.4f}; time:{time_for_all:.2f}s')
                logger.debug(f'{mtd.name:8} margin: {acc[3]:.6f}:{acc[4]:.6f}')
        for mtd in mtds:
            this_accs = np.asarray(accs[mtd])
            avg = np.average(this_accs, axis=0)
            std = np.std(this_accs, axis=0)
            logger.info(f'{mtd.name:8}: {avg[0]*100:6.2f}: {avg[1]*100:6.2f}±{std[1]*100:.2f} , F1 = {avg[2]:.4f} ; time: {avg[-1]:.2f} s')
            logger.debug(f'{mtd.name:8} margin: {avg[3]:.4f}±{std[3]:.4f} : {avg[4]:.4f}±{std[4]:.4f}')

def compare_with_label(l_range=20, test_num=3) -> None:
    data_set=[
        # Data('Toy-3Rings', None, c=2, args=args_0),
        # Data('Balance', 625, c=3), #1e-4
        # Data('MobileKSD', 2856, c=56), #1e-3
        # Data('USPS', 9298, c=10), #1e-4
        # Data('CsMap', 10845, c=6), #1e-3
        # Data('PhishingWeb', 11055, c=2), #1e-3
        # Data('Swarm', 24016, c=4), #1e-5
        # Data('CreditCard', 30000, c=2), #1e-3
        Data('Mnist', 70000, c=10),
        Data('SensIT', 98528, c=2),
        
        Data('EMnist', 814255, c=47),
        ]
    timer.record(f'set {len(data_set):} datas')
    
    
    mtds = [
            kNN(),
            SVM('PolySVM', kernel='poly'),
            SVM('RbfSVM', kernel='rbf'),
            # Mtd('LLGC', acclr='g', beta=1, eta=0),
            # Mtd('HF', acclr='g', H=True),
            # Mtd('GF(G)', beta='n*0.00001', acclr='g'),
            Mtd('GF(A)', beta='n*0.00001', acclr='a'),
            ]
    timer.record(f'set {len(mtds):} methods')
       
    args = {'k': 20, 'm': 50, 't': 10}
    for data in data_set:
        if data.name == 'Balance':
            args['m'] = 21
        elif data.name == 'MobileKSD':
            args['m'] = 2856
        elif data.name in ('PhishingWeb', 'SensIT'):
            args['m'] = 50
        elif data.name == 'EMnist':
            args['m'] = 2048
        else:
            args['m'] = 1024
        logger.info(f'{data.name}: ' + str(args))
        avgs = {mtd:[] for mtd in mtds} 
        l_ = [l for l in range(1, l_range+1)]
        for l in l_:
            accs = {mtd:[] for mtd in mtds} 
            data.labeled_num = data.c * l
            question = Que(data, data_root, logger)
            for seed in range(test_num):
                question.allocate(seed)
                logger.debug(f'allocate with seed:{seed}')
                for mtd in mtds:
                    try:
                        ind, time_for_UBF = mtd.get_indicator(question, args)
                        acc, time_for_acc = question.evaluate(mtd.name, ind)
                        time_for_all = time_for_UBF + time_for_acc
                    except:
                        logger.info(f'To solve {data.name} with {mtd.name} failed')
                        acc = (0, 0)
                        time_for_all = (0)
                    accs[mtd].append((acc[0], acc[1], time_for_all))
                    logger.debug(f'{mtd.name:8}: {acc[0]*100:6.2f}:{acc[1]*100:6.2f}; time:{time_for_all:.2f}s')
            for mtd in mtds:
                this_accs = np.asarray(accs[mtd])
                avgs[mtd].append(np.average(this_accs, axis=0))
        accs_ = []
        names_ = []
        for mtd in mtds:
            this_accs = np.asarray(avgs[mtd])
            accs_.append(this_accs[:, 1].tolist())
            names_.append(mtd.name)
        logger.info(l_)
        logger.info(accs_)
        logger.info(names_)
        visualize.multiple_line_with_legend(l_, accs_, data.name+'_label', names_)
        visualize.show(xlabel="Number of labeled samples per class", ylabel="Classification Accuracy(%)")

def compare_with_anchor(data_set=None, mtds=None, test_num=10) -> None:
    data_set = data_set or [
        # Data('Toy-3Rings', None, 3*10, args=args_0),
        # Data('Balance', 625, 3*10),
        # Data('MobileKSD', 2856, 56*10),
        # Data('USPS', 9298, 10*10),
        # Data('CsMap', 10845, 6*10),
        # Data('PhishingWeb', 11055, 2*10),
        # Data('Swarm', 24016, 4*10), #1e-4
        # Data('CreditCard', 30000, 2*10),
        # Data('Mnist', 70000, 10*10),
        # Data('SensIT', 98528, 2*10),
        Data('EMnist', 814255, 47*10),
        ]
    timer.record(f'set {len(data_set):} datas')
    
    mtds = mtds or [
            Mtd('GF(A)', beta='n*0.00001', acclr='a'),
            ]
    timer.record(f'set {len(mtds):} methods')
       
    args = {'k': 20, 'm': 2048, 't': 5}
    for data in data_set:
        logger.info(f'{data.name}')
        avgs = {mtd:[] for mtd in mtds}
        # m_range = [m for m in range(20, 128, 1)]
        m_range = [m for m in range(20, 4000, 40)]
        for m in m_range:
            args['m'] = m
            logger.debug(f'anchors: {m}')
            accs = {mtd:[] for mtd in mtds} 
            question = Que(data, data_root, logger)
            for seed in range(test_num):
                question.allocate(seed)
                # if args['m'] == 300 and seed == 3:
                #     print('pause')
                # timer.record(f'question {question.name:} with seed {seed:} allocated')
                for mtd in mtds:
                    # try:
                    ind, time_for_UBF = mtd.get_indicator(question, args)
                    acc, time_for_acc = question.evaluate(mtd.name, ind)
                    time_for_all = time_for_UBF + time_for_acc
                    # except:
                    #     logger.debug(f'To solve {data.name} with {mtd.name} failed')
                    #     acc = (0, 0)
                    #     time_for_all = (0)
                    accs[mtd].append((acc[0], acc[1], time_for_all))
                    logger.debug(f'{mtd.name:8}: {acc[0]*100:6.2f}%:{acc[1]*100:6.2f}%; time:{time_for_all:.2f}s')
            for mtd in mtds:
                this_accs = np.asarray(accs[mtd])
                avgs[mtd].append(np.average(this_accs, axis=0))
        this_accs = np.asarray(avgs[mtd])
        accs_ = (this_accs[:, 1]* 100).tolist()
        times_ = this_accs[:, -1].tolist()
        logger.info(m_range)
        logger.info(accs_)
        logger.info(times_)
        visualize.twinLine2D_graph(m_range, accs_, times_, title=data.name+'_anchor')
        visualize.show()
        
def compare_with_k(data_set=None, mtds=None, test_num=1) -> None:
    data_set = data_set or [
        # Data('Toy-3Rings', None, 3*10, args=args_0),
        Data('Balance', 625, 3*10),
        # Data('PhishingWeb', 11055, 2*10),
        # Data('USPS', 9298, 10*10),
        # Data('Mnist', 70000, 10*10),
        # Data('SensIT', 98528, 2*10),
        # Data('EMnist', 814255, 62*10),
        ]
    timer.record(f'set {len(data_set):} datas')
        
    mtds = mtds or [
            # kNN(),
            # SVM(),
            # Mtd('LLGC', beta=1, eta=0, acclr='o'),
            # Mtd('HF', H=True, acclr='o'),
            # Mtd('GF', beta='n*0.001'),
            # Mtd('GF(Gause)', beta='n*0.001', acclr='g'),
            Mtd('GF(Anchor)', acclr='a'),
            Mtd('GF(Disturbance)', beta='n*0.001', acclr='a'),
            ]
    timer.record(f'set {len(mtds):} methods')

    args = {'k': 'vary', 'm': 1024, 't': 10}
    
    logger.info(f'args:{args}')
    for data in data_set:
        for _ in range(3):
            k = args['k'] = 15 + _ * 5
            logger.debug(f'k:{k}')
            accs = {mtd:[] for mtd in mtds} 
            question = Que(data, data_root, logger)
            logger.debug(f'question {question.name}')
            for seed in range(test_num):
                question.allocate(seed)
                logger.debug(f'allocate with seed:{seed}')
                for mtd in mtds:
                    try:
                        ind, time_for_UBF = mtd.get_indicator(question, args)
                        acc, time_for_acc = question.evaluate(mtd.name, ind)
                        time_for_all = time_for_UBF + time_for_acc
                    except:
                        logger.debug(f'To solve {data.name} with {mtd.name} failed')
                        acc = (np.nan, np.nan)
                        time_for_all = (np.nan)
                    accs[mtd].append((acc[0], acc[1], time_for_all))
                    logger.debug(f'{mtd.name:8}: {acc[0]*100:6.2f}%:{acc[1]*100:6.2f}%; time:{time_for_all:.2f}s')
            for mtd in mtds:
                this_accs = np.asarray(accs[mtd])
                avg = np.average(this_accs, axis=0)
                std = np.std(this_accs, axis=0)
                logger.debug(f'{mtd.name:8}: {avg[0]*100:6.2f}:{avg[1]*100:6.2f}±{std[1]*100:.4f}; time:{avg[2]:.2f}s')

def compare_with_mu(data_set=None, mtds=None, test_num=5) -> None:
    data_set = data_set or [
        # Data('Toy-3Rings', None, 3*10, args=args_0),
        Data('Balance', 625, 3*10),
        Data('MobileKSD', 2856, 56*10),
        Data('USPS', 9298, 10*10),
        Data('PhishingWeb', 11055, 2*10),
        Data('Mnist', 70000, 10*10),
        Data('SensIT', 98528, 2*10),
        Data('EMnist', 814255, 62*10),
        ]
    timer.record(f'set {len(data_set):} datas')
    
    mtd = Mtd('GF(A)', beta='n*1e', acclr='a')
       
    args = {'k': 20, 'm': 1024, 't': 10}
    for data in data_set:
        if data.name in ('Balance'):
            args['m'] = 21
        elif data.name in ('MobileKSD'):
            args['m'] = 2856
        elif data.name in ('PhishingWeb', 'SensIT'):
            args['m'] = 50
        else:
            args['m'] = 1024
        logger.info(f'{data.name}, args={args}')
        accs = []
        mu_range = [mu for mu in range(-10, 11, 1)]
        for mu in mu_range:
            mtd.beta = 'n*1e' + str(mu)
            question = Que(data, data_root, logger)
            logger.info(f'{mtd.beta}:')
            avgs = []
            for seed in range(test_num):
                question.allocate(seed)
                ind, time_for_UBF = mtd.get_indicator(question, args)
                acc, time_for_acc = question.evaluate(mtd.name, ind)
                time_for_all = time_for_UBF + time_for_acc
                avgs.append((acc[0], acc[1], time_for_all))
                logger.debug(f'{mtd.name:8}: {acc[0]*100:6.2f}%:{acc[1]*100:6.2f}%; time:{time_for_all:.2f}s')
            avgs = np.asarray(avgs)
            accs.append(np.average(avgs, axis=0))
        accs = np.asarray(accs)
        accs_ = (accs[:, 1]* 100).tolist()
        logger.info(mu_range)
        logger.info(accs_)
        visualize.single_line(mu_range, accs_)
        visualize.show(xlabel='the value of parameter(1ex)', ylabel='Classification Accuracy(%)')
        

if __name__ == '__main__':
    
    logger.info(f'data_root: {data_root}')
    
    
    compare_with_methods()
    # compare_with_label()
    # compare_with_anchor()
    # compare_with_mu()
    
 
    
    



    
