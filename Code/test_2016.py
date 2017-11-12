# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:52:32 2017

@author: lenovo laptop
"""
from test5 import read_citance
from test8 import read_annt_sent
import os
import re

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Training-Set-2016"
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        dest = re.sub("\_TRAIN" , "", folder)
        file = os.path.join(path, folder)+"/annotation/"+dest+".annv2.txt"
        file = file.replace('\\','/')
        read_citance(file)
        read_annt_sent(file)
        
