#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:07:49 2020

@author: fabian geiger
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from pathlib import Path


def plot_cm(session):
    """
    Utility function to plot and save the confusion matrices when using
    different recording sessions for testing while using the rest of the data
    for training
    """
    #resdir = Path('/home/fabian/Documents/Master_thesis/Python_Code/results/stag_realData')
    resdir = Path('../results/stag_realData')
    file = 'np44225_sess' + str(session) + '_cm_history.npy'
    data = np.load(resdir / file , allow_pickle=True)
    conf_m = data[6]
    
    classes = ['Ball', 'Battery', 'Bracket', 'Coin', 'Empty can', 'Empty hand',
               'Full can', 'Gel', 'Lotion', 'Mug', 'Pen', 'Safety glasses',
               'Scissors', 'Screw driver', 'Spray can', 'stapler', 'Tape']
    
    df_cm = pd.DataFrame(conf_m.astype(np.int), classes, classes)
    
    
    #sn.set(font_scale=1.1) # for label size
    
    plt.figure(figsize=(9,9))
    plt.title('Session ' + str(session+1))
    hm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},
               cbar=False, cmap=plt.get_cmap('Greens'), fmt='d', square=True)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    #plt.show()
    plt.savefig('sess' + str(session+1) + '_cm')
    

def plot_cm_cv():
    """
    Utility function to plot and save the confusion matrix of the cross
    validation experiment
    """
    resdir = Path('../results/stag_realData')
    file = 'np44225_random_acc_history.npy'
    data = np.load(resdir / file , allow_pickle=True)
    conf_m = data[6]
    
    classes = ['Ball', 'Battery', 'Bracket', 'Coin', 'Empty can', 'Empty hand',
               'Full can', 'Gel', 'Lotion', 'Mug', 'Pen', 'Safety glasses',
               'Scissors', 'Screw driver', 'Spray can', 'stapler', 'Tape']
    
    df_cm = pd.DataFrame(conf_m.astype(np.int), classes, classes)
    
    
    #sn.set(font_scale=1.1) # for label size
    
    plt.figure(figsize=(9,9))
    plt.title('Random Train/Test Split')
    hm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},
               cbar=False, cmap=plt.get_cmap('Greens'), fmt='d', square=True)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    plt.show()
    #plt.savefig('random_split_cm')
    
    
if __name__ == "__main__":
    sessions = [0, 1, 2, 3, 4]
    for sess in sessions:
        plot_cm(sess)
    #plot_cm_cv()
