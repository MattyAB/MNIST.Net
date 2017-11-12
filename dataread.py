# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:52:27 2017

@author: matth
"""

import csv
    
with open('train.csv', 'rU') as f:  #opens PW file
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
    f.close() #close the csv

print(data)