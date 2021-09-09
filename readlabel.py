import networkx as nx
import json
import r2pipe, os, sys, time, csv
import numpy as np
import signal
import pickle

def read_label():
    # read label file
    #filename = []
    label_dict = {'BenignWare':0, 'Mirai':1, 'Tsunami':2, 'Hajime':3, 'Dofloo':4, 'Bashlite':5, 'Xorddos':6, 'Android':7, 'Pnscan':8, 'Unknown':9}
    label = {}
    threshold = {}
    with open('/home/connlab/ChiaYi/CFGtest/CFG/dataset.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for row in rows:
            threshold[row[0]] = row[2]
            label[row[0]] = label_dict[row[1]]
            #filename.append(row[0])

    with open("label_dict.pkl", 'wb') as f:
        pickle.dump(label, f)
    print('---- finish read label ----\n')
    return label

