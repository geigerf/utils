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
from sklearn.model_selection import train_test_split
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
n_classes = 27
split = 'recording'
plot = True

#metafile = '/home/fabian/Documents/Master_thesis/Research/STAG_MIT/classification_lite/metadata.mat'
metafile = '../../Research/STAG_MIT/classification_lite/metadata.mat'

data = sio.loadmat(metafile)
valid_mask = data['hasValidLabel'].flatten() == 1
balanced_mask = data['isBalanced'].flatten() == 1
splits = data['splitId'].flatten()
# indices now gives a subset of the data set that contains only valid
# pressure frames and the same number of frames for each class
mask = np.logical_and(valid_mask, balanced_mask)

pressure = np.transpose(data['pressure'], axes=(0, 2, 1))
object_id = data['objectId'].flatten()

if split == 'original':
    # Prepare the data the same way as in the MIT paper
    pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)

    split_mask = splits == 0
    train_indices = np.logical_and(mask, split_mask)
    pressure_train = pressure[train_indices]
    
    train_data = pressure_train
    train_labels = object_id[train_indices]

    split_mask = splits == 1
    test_indices = np.logical_and(mask, split_mask)
    pressure_test = pressure[test_indices]
    
    test_data = pressure_test
    test_labels = object_id[test_indices]
    
    mean_train = []
    mean_test = []
    # Calculate the mean pressure frame for each class
    for i in range(n_classes):
        mask = train_labels == i
        samples = train_data[mask]
        mean_train.append(np.mean(samples, axis=0))
        
        mask = test_labels == i
        samples = test_data[mask]
        mean_test.append(np.mean(samples, axis=0))
    
    if plot:
        # Plot the mean pressure frames for the original data split
        #fname = '/home/fabian/Documents/Master_thesis/Python_Code/results/stag/compare_sessions/original_'
        fname = '../results/stag/compare_sessions/original_'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        cbar_ax = fig.add_axes([.91, .15, .03, .7])
        for j in range(n_classes):
            vmax = np.max([mean_train[j], mean_test[j]])
            sns.heatmap(mean_train[j], vmax=vmax, square=True, ax=ax1,
                        cbar=False)
            ax1.title.set_text('Training')
            sns.heatmap(mean_test[j], vmax=vmax, square=True, ax=ax2,
                        cbar_ax=cbar_ax)
            ax2.title.set_text('Test')
            fig.suptitle('Original split Class {:d}'.format(j), fontsize=16)
            plt.savefig(fname=(fname + 'class' + str(j)))

elif split == 'random':
    # Prepare the data the same way as in the paper
    pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)

    # Only use valid and balanced data
    train_data, test_data,\
        train_labels, test_labels = train_test_split(pressure[mask],
                                                     object_id[mask],
                                                     test_size=0.306,
                                                     random_state=seed+1,
                                                     shuffle=True,
                                                     stratify=object_id[mask])
        
    mean_train = []
    mean_test = []
    for i in range(n_classes):
        mask = train_labels == i
        samples = train_data[mask]
        mean_train.append(np.mean(samples, axis=0))
        
        mask = test_labels == i
        samples = test_data[mask]
        mean_test.append(np.mean(samples, axis=0))
    
    if plot:
        # Plot the mean pressure frames for a random data split
        #fname = '/home/fabian/Documents/Master_thesis/Python_Code/results/stag/compare_sessions/random_'
        fname = '../results/stag/compare_sessions/random_'
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
        
elif split == 'recording':
    ## pressure = boost_normalize(pressure.astype(np.float32))
    ## pressure = normalize(pressure.astype(np.float32))
    # Each class has three recording IDs that correspond to the different
    # experiment days. There are 81 recording IDs (3*27)
    # 0  - 26 belong to the first recording
    # 27 - 53 belong to the second recording
    # 54 - 81 belong to the third recording
    recording_id = data['recordingId'].flatten()
    recordings = []
    for i in range(3):
        # Find valid samples from the different recording days
        recording_mask = np.logical_and(recording_id >= i*27,
                                        recording_id < (i+1)*27)
        recording_mask = np.logical_and(recording_mask, valid_mask)
        
        # The data is not yet balanced!
        recordings.append([pressure[recording_mask],
                           object_id[recording_mask]])
        
    x1, y1 = recordings[0][0], recordings[0][1]
    x2, y2 = recordings[1][0], recordings[1][1]
    x3, y3 = recordings[2][0], recordings[2][1]
    
    x1 = normalize_per_pixel(x1)
    x2 = normalize_per_pixel(x2)
    x3 = normalize_per_pixel(x3)
    
    mean_1 = []
    mean_2 = []
    mean_3 = []
    for i in range(n_classes):
        mask = y1 == i
        samples = x1[mask]
        mean_1.append(np.mean(samples, axis=0))
        
        mask = y2 == i
        samples = x2[mask]
        mean_2.append(np.mean(samples, axis=0))
        
        mask = y3 == i
        samples = x3[mask]
        mean_3.append(np.mean(samples, axis=0))
        
    if plot:
        # Plot the mean pressure frames for each recording session
        #fname = '/home/fabian/Documents/Master_thesis/Python_Code/results/stag/compare_sessions/perpixel_'
        fname = '../results/stag/compare_sessions/perpixel_'
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
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
                        square=True, ax=ax3, cbar_ax=cbar_ax)
            ax3.title.set_text('Session 3')
            fig.suptitle('Boosted Normalization Class {:d}'.format(j),
                         fontsize=16)
            plt.savefig(fname=(fname + 'class' + str(j)))
