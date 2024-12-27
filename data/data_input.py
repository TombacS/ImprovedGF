# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:28:39 2021

@author: Dell
"""

# from .Matdata.matdata import Matdata

# from .Cifar.cifar import Cifar
# from .COIL.coil import COIL as Coil
# from .Connect4.connect4 import Connect4
# from .CovType.covtype import CovType
# from .Mnist.mnist import Mnist
# from .Newsgroups.newsgroup import Newsgroup
# from .SensIT.sensIT import SensIT
from .USPS.usps import USPS
# from .Balance.balance import Balance
# from .PhishingWeb.phishingWeb import PhishingWeb
# from .MobileKSD.mobileKSD import MobileKSD
# from .HTRU2.htru import HTRU
# from .CreditCard.creditCard import CreditCard
# from .CsMap.CsMap import CsMap
# from .Swarm.swarm import Swarm
# from .Steel.steel import Steel
# from .Toxicity.toxicity import Toxicity


from .myTOY.toy import Toy

import os
# data_name in ('Cifar','COIL','Connect4','Covtype','Mnist','fasion-Mnist','Newsgroup','SensIT','USPS')
# ret: (X, Y, y, lab_num)

def data_input(data, args=None): # ret label, data, c
    file_root = os.path.dirname(__file__)
    # print(file_root)
    try:
        if data.name[:3] == 'Toy':
            dataclass = eval('Toy')(file_root)
            return dataclass.load_data(data)
        if data.name == 'FMnist':
            dataclass = eval('Mnist')(file_root, kind='F')
        elif data.name == 'EMnist':
            dataclass = eval('Mnist')(file_root, kind='E')
        else:
            dataclass = eval(data.name)(file_root)
        return dataclass.load_data()
    except Exception as e:
        print('Exception: ' + e)
        print(f'data {data.name} load fail')
        raise
