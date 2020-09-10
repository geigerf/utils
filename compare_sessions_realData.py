#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:15:37 2020

@author: fabian geiger
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns
# from tabulate import tabulate
# from imblearn import under_sampling, over_sampling


def normalize(pressure):
    """
    Scales each array of the given array of arrays to the range [0, 1]
    Only considers values in the same tactile frame
    """
    normalized_p = np.copy(pressure)
    for i in range(pressure.shape[0]):
        min_p = np.min(pressure[i])
        normalized_p[i] = (pressure[i] - min_p) / np.max(pressure[i] - min_p)
    
    return normalized_p


def normalize_per_pixel(pressure):
    """
    Scales each element of the given array of arrays to the range [0, 1]
    Considers values in all tactile frames
    """
    normalized_p = np.copy(pressure)
    # First scale values to [0, 1]
    min_p = np.min(pressure)
    normalized_p = (pressure - min_p) / np.max(pressure - min_p)
    # Then subtract the mean for each pixel
    pixel_mean = np.mean(normalized_p, axis=0)
    # pixel_mean should be shaped like normalized_p
    normalized_p = normalized_p - pixel_mean
    
    return normalized_p


seed = 333
n_classes = 17
split = 'session'
plot = False

#filename = '/home/fabian/Documents/Master_thesis/Data_Collection/3kOhm_FB/data_MT_FabianGeiger_4sess.mat'
filename = '../../Data_Collection/3kOhm_FB/data_MT_FabianGeiger_5sess.mat'

split = 'session'

data = sio.loadmat(filename, squeeze_me=True)
# Use only frames in which objects were touched
valid_mask = data['valid_flag'] == 1
pressure = data['tactile_data'][valid_mask]
# Scale data to the range [0, 1]
pressure = np.clip((pressure.astype(np.float32)-1510)/(3000-1510), 0.0, 1.0)
object_id = data['object_id'][valid_mask]


if split == 'random':
    # Only use valid and balanced data
    train_data, test_data,\
        train_labels, test_labels = train_test_split(pressure,
                                                     object_id,
                                                     test_size=0.306,
                                                     random_state=seed,
                                                     shuffle=True,
                                                     stratify=object_id)
        
    mean_train = []
    mean_test = []
    for i in range(n_classes):
        mask = train_labels == i
        samples = train_data[mask]
        if(len(samples) != 0):
            mean_train.append(np.mean(samples, axis=0).reshape((32,32)))
        
        mask = test_labels == i
        samples = test_data[mask]
        if(len(samples) != 0):
            mean_test.append(np.mean(samples, axis=0).reshape((32,32)))
    
    if plot:
        # Plot the mean pressure frames for a random data split
        #fname = '/home/fabian/Documents/Master_thesis/Python_Code/results/stag/compare_sessions_realData/random_'
        fname = '../results/stag/compare_sessions_realData/random_'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        cbar_ax = fig.add_axes([.91, .15, .03, .7])
        for j in range(n_classes):
            vmax = np.max([mean_train[j], mean_test[j]])
            sns.heatmap(mean_train[j], vmax=vmax, square=True, ax=ax1, cbar=False)
            ax1.title.set_text('Training')
            sns.heatmap(mean_test[j], vmax=vmax, square=True, ax=ax2, cbar_ax=cbar_ax)
            ax2.title.set_text('Test')
            fig.suptitle('Random split Class {:d}'.format(j), fontsize=16)
            plt.savefig(fname=(fname + 'class' + str(j)))
        
elif split == 'session':
    num_sessions = len(np.unique(data['session_id']))
    x = []
    y = []
    valid_sessions = data['session_id'][valid_mask]
    for i in range(num_sessions):
        session_mask = valid_sessions == i
        x.append(pressure[session_mask])
        y.append(object_id[session_mask])
    
    mean_1 = []
    mean_2 = []
    mean_3 = []
    mean_4 = []
    mean_5 = []
    for i in range(n_classes):
        mask = y[0] == i
        samples = x[0][mask]
        if(len(samples) != 0):
            mean_1.append(np.mean(samples, axis=0).reshape((32,32)))
        else:
            mean_1.append(np.zeros((32,32)))
        
        mask = y[1] == i
        samples = x[1][mask]
        if(len(samples) != 0):
            mean_2.append(np.mean(samples, axis=0).reshape((32,32)))
        else:
            mean_2.append(np.zeros((32,32)))
            
        mask = y[2] == i
        samples = x[2][mask]
        if(len(samples) != 0):
            mean_3.append(np.mean(samples, axis=0).reshape((32,32)))
        else:
            mean_3.append(np.zeros((32,32)))
        
        mask = y[3] == i
        samples = x[3][mask]
        if(len(samples) != 0):
            mean_4.append(np.mean(samples, axis=0).reshape((32,32)))
        else:
            mean_4.append(np.zeros((32,32)))
    
        mask = y[4] == i
        samples = x[4][mask]
        if(len(samples) != 0):
            mean_5.append(np.mean(samples, axis=0).reshape((32,32)))
        else:
            mean_5.append(np.zeros((32,32)))
    
    
    response = [np.mean(mean_1), np.mean(mean_2), np.mean(mean_3), np.mean(mean_4), np.mean(mean_5)]
    response = response/max(response)
    sessions = np.arange(1, 6)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8, 4.5))
    plt.title('Sensor Degradation')
    plt.plot(sessions, response, marker='o', color=(204/255, 37/255, 41/255))
    plt.xlabel('Session')
    plt.xlim(0, 6)
    plt.xticks(np.arange(7))
    plt.ylim(0, 1.1)
    #plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel('Relative mean response')
    plt.grid(True)
    plt.show()
    
    if plot:
        # Plot the mean pressure frames for each recording session
        #fname = '/home/fabian/Documents/Master_thesis/Python_Code/results/stag/compare_sessions_realData/session_'
        fname = '../results/stag/compare_sessions_realData/session_'
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
        cbar_ax = fig.add_axes([.91, .15, .03, .7])
        for j in range(n_classes):
            vmax = np.max([mean_1[j], mean_2[j], mean_3[j]])
            vmin = np.min([mean_1[j], mean_2[j], mean_3[j]])
            sns.heatmap(mean_1[j], vmax=vmax, vmin=vmin,
                        square=True, ax=ax1, cbar=False)
            ax1.title.set_text('Session 1')
            sns.heatmap(mean_2[j], vmax=vmax, vmin=vmin,
                        square=True, ax=ax2, cbar=False)
            ax2.title.set_text('Session 2')
            sns.heatmap(mean_3[j], vmax=vmax, vmin=vmin,
                        square=True, ax=ax3, cbar=False)
            ax3.title.set_text('Session 3')
            sns.heatmap(mean_4[j], vmax=vmax, vmin=vmin,
                        square=True, ax=ax4, cbar_ax=cbar_ax)
            ax4.title.set_text('Session 4')
            fig.suptitle('Class {:d}'.format(j), fontsize=16)
            plt.savefig(fname=(fname + 'class' + str(j)))