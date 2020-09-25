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
plot = True

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

    
    if plot:
        # Plot the mean pressure frames for each recording session
        fname = '../results/stag/compare_sessions_realData/session_'
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 3))#, sharey=True)
        plt.subplots_adjust(left=0.1, right=0.9)
        #cbar_ax = fig.add_axes([.91, .15, .03, .7])
        for j in range(n_classes):
            vmax = np.max([mean_1[j], mean_2[j], mean_3[j], mean_4[j], mean_5[j]])
            vmin = np.min([mean_1[j], mean_2[j], mean_3[j], mean_4[j], mean_5[j]])
            sns.heatmap(np.transpose(mean_1[j]), vmax=vmax, vmin=vmin,
                        square=True, ax=ax1, cbar=False, cmap='gray')
            ax1.axes.xaxis.set_label_text('Session 1')
            ax1.axes.xaxis.set_ticks([])
            ax1.axes.yaxis.set_ticks([])
            sns.heatmap(np.transpose(mean_2[j]), vmax=vmax, vmin=vmin,
                        square=True, ax=ax2, cbar=False, cmap='gray')
            ax2.axes.xaxis.set_label_text('Session 2')
            ax2.axes.xaxis.set_ticks([])
            ax2.axes.yaxis.set_ticks([])
            sns.heatmap(np.transpose(mean_3[j]), vmax=vmax, vmin=vmin,
                        square=True, ax=ax3, cbar=False, cmap='gray')
            ax3.axes.xaxis.set_label_text('Session 3')
            ax3.axes.xaxis.set_ticks([])
            ax3.axes.yaxis.set_ticks([])
            sns.heatmap(np.transpose(mean_4[j]), vmax=vmax, vmin=vmin,
                        square=True, ax=ax4, cbar=False, cmap='gray')
            ax4.axes.xaxis.set_label_text('Session 4')
            ax4.axes.xaxis.set_ticks([])
            ax4.axes.yaxis.set_ticks([])
            sns.heatmap(np.transpose(mean_5[j]), vmax=vmax, vmin=vmin,
                        square=True, ax=ax5, cmap='gray', cbar=False)#cbar_ax=cbar_ax)
            ax5.axes.xaxis.set_label_text('Session 5')
            ax5.axes.xaxis.set_ticks([])
            ax5.axes.yaxis.set_ticks([])
            fig.suptitle('Class {:d}'.format(j))#, fontsize=16)
            plt.savefig(fname=(fname + 'class' + str(j)))
