#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:33:56 2020

@author: fabian geiger
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn
from sklearn import decomposition
from sklearn import cluster
from sklearn.model_selection import train_test_split
# from tabulate import tabulate
# from imblearn import under_sampling, over_sampling


seed = 333
n_classes = 27
split = 'recording'

#metafile = '/home/fabian/Documents/Master_thesis/Research/STAG_MIT/classification_lite/metadata.mat'
metafile = '.../../Research/STAG_MIT/classification_lite/metadata.mat'

data = sio.loadmat(metafile)
valid_mask = data['hasValidLabel'].flatten() == 1
balanced_mask = data['isBalanced'].flatten() == 1
splits = data['splitId'].flatten()
# indices now gives a subset of the data set that contains only valid
# pressure frames and the same number of frames for each class
mask = np.logical_and(valid_mask, balanced_mask)

pressure = np.transpose(data['pressure'], axes=(0, 2, 1))
# Prepare the data the same way as in the paper
pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)
pressure = pressure.reshape((-1, 32*32))
object_id = data['objectId'].flatten()

if split == 'original':
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

elif split == 'random':
    # Only use the valid and balanced data
    train_data, test_data,\
        train_labels, test_labels = train_test_split(pressure[mask],
                                                     object_id[mask],
                                                     test_size=0.306,
                                                     random_state=seed+1,
                                                     shuffle=True,
                                                     stratify=object_id[mask])
        
elif split == 'recording':
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
    
    # Balance data using the python package 'imbalanced-learn'
    # Random undersampling.
    # undersampler = under_sampling.RandomUnderSampler(random_state=seed+2,
    #                                                  sampling_strategy='not minority')
    
    # KMeansSMOTE oversampling. This generates NEW samples!
    # Can be seen as data augmentation. kmeans_estimator tells the sampler
    # how many clusters to generate
    # oversampler = over_sampling.KMeansSMOTE(random_state=seed+2,
    #                                         kmeans_estimator=20)
    
    # # First undersample the majority class
    # x1_resampled, y1_resampled = undersampler.fit_resample(x1, y1)
    # x2_resampled, y2_resampled = undersampler.fit_resample(x2, y2)
    # x3_resampled, y3_resampled = undersampler.fit_resample(x3, y3)
    
    # # Then oversample the rest of the classes such that the set is balanced
    # x1_resampled, y1_resampled = oversampler.fit_resample(x1_resampled,
    #                                                       y1_resampled)    
    # x2_resampled, y2_resampled = oversampler.fit_resample(x2_resampled,
    #                                                       y2_resampled)
    # x3_resampled, y3_resampled = oversampler.fit_resample(x3_resampled,
    #                                                       y3_resampled)
    
    # # Try to oversample without undersampling class 0 first
    # x1_resampled, y1_resampled = oversampler.fit_resample(x1, y1)    
    # x2_resampled, y2_resampled = oversampler.fit_resample(x2, y2)
    # x3_resampled, y3_resampled = oversampler.fit_resample(x3, y3)


if split == 'random' or split == 'original':
    # Find different grip types with PCA
    pca = decomposition.PCA(n_components=8, random_state=seed+3)
    kmeans = cluster.KMeans(n_clusters=8, init='k-means++', n_init=50,
                            random_state=seed+4)
    
    cluster_train = []
    cluster_test = []
    for i in range(n_classes):
        table = []
        # print('CLASS ' + str(i), file=f)
        mask_train = (train_labels == i)
        samples_train = train_data[mask_train]
        
        Xemb_train = pca.fit_transform(samples_train)
        x_min = np.min(Xemb_train)
        x_max = np.max(Xemb_train)
        Xemb_train = (Xemb_train - x_min) / (x_max - x_min)
        
        Xkmeans_train = kmeans.fit(Xemb_train)
    
        # Training
        for j in range(8):
            mask = Xkmeans_train.labels_ == j
            n = np.count_nonzero(mask)
            table.append([str(j), n])
            # Find the samples corresponding to this cluster and average them
            cluster_train.append(np.mean(samples_train[mask],
                                         axis=0).reshape((32,32)))
    
        mask_test = (test_labels == i)
        samples_test = test_data[mask_test]
        
        Xemb_test = pca.fit_transform(samples_test)
        x_min = np.min(Xemb_test)
        x_max = np.max(Xemb_test)
        Xemb_test = (Xemb_test - x_min) / (x_max - x_min)
        
        Xkmeans_test = kmeans.fit(Xemb_test)
        
        # Test
        for j in range(8):
            mask = Xkmeans_test.labels_ == j
            n = np.count_nonzero(mask)
            table[j].append(n)
            table[j].append(n+table[j][1])
            cluster_test.append(np.mean(samples_test[mask],
                                        axis=0).reshape((32,32)))
            
        # print(tabulate(table, headers=['Cluster', 'Training', 'Test', 'Total'],
        #                tablefmt='github'), file=f)
        # print('\n', file=f)
            
    #fname = '/home/fabian/Documents/Master_thesis/Python_Code/results/stag/cluster/'
    fname = '../results/stag/cluster/'
    fname = fname + split + '_test_split/'
    fig, axs = plt.subplots(2,8, sharex=True, sharey=True, figsize=(22,8), dpi=80)
    fig.tight_layout()
    axs[0][0].set_ylabel('Training')
    axs[1][0].set_ylabel('Test')
    for i in range(n_classes):
        for j in range(8):
            axs[0][j].imshow(cluster_train[i*8+j])
            axs[0][j].tick_params(axis='both', which='both',
                                  labelbottom=False, labelleft=False,
                                  bottom=False, top=False, left=False, right=False)
            axs[1][j].imshow(cluster_test[i*8+j])
            axs[1][j].tick_params(axis='both', which='both',
                                  labelbottom=False, labelleft=False,
                                  bottom=False, top=False, left=False, right=False)
        fig.suptitle('Class {:d}'.format(i))
        plt.savefig(fname=(fname + 'class' + str(i)))


elif split == 'recording':
    # Do clustering without PCA
    pca = decomposition.PCA(n_components=16, random_state=seed+3)
    nclusters = 8
    kmeans = cluster.KMeans(n_clusters=nclusters, init='k-means++', n_init=50,
                            random_state=seed+4)

    cluster_1 = {'mean': [], 'std': []}
    cluster_2 = {'mean': [], 'std': []}
    cluster_3 = {'mean': [], 'std': []}
    for i in range(n_classes):
        table = []
        # print('CLASS ' + str(i), file=f)
        # Recording 1
        mask_1 = (y1 == i)
        samples_1 = x1[mask_1]
        
        Xemb_1 = pca.fit_transform(samples_1)
        x_min = np.min(Xemb_1)
        x_max = np.max(Xemb_1)
        Xemb_1 = (Xemb_1 - x_min) / (x_max - x_min)
        
        Xkmeans_1 = kmeans.fit(Xemb_1)

        for j in range(nclusters):
            mask = Xkmeans_1.labels_ == j
            n = np.count_nonzero(mask)
            table.append([str(j), n])
            # Find the samples corresponding to this cluster and average them
            cluster_1['mean'].append(np.mean(samples_1[mask],
                                             axis=0).reshape((32,32)))
            cluster_1['std'].append(np.std(samples_1[mask],
                                           axis=0).reshape((32,32)))
    
        # Recording 2
        mask_2 = (y2 == i)
        samples_2 = x2[mask_2]
        
        Xemb_2 = pca.fit_transform(samples_2)
        x_min = np.min(Xemb_2)
        x_max = np.max(Xemb_2)
        Xemb_2 = (Xemb_2 - x_min) / (x_max - x_min)
        
        Xkmeans_2 = kmeans.fit(Xemb_2)
    
        for j in range(nclusters):
            mask = Xkmeans_2.labels_ == j
            n = np.count_nonzero(mask)
            table.append([str(j), n])
            # Find the samples corresponding to this cluster and average them
            cluster_2['mean'].append(np.mean(samples_2[mask],
                                             axis=0).reshape((32,32)))
            cluster_2['std'].append(np.std(samples_2[mask],
                                           axis=0).reshape((32,32)))

        # Recording 3
        mask_3 = (y3 == i)
        samples_3 = x3[mask_3]
        
        Xemb_3 = pca.fit_transform(samples_3)
        x_min = np.min(Xemb_3)
        x_max = np.max(Xemb_3)
        Xemb_3 = (Xemb_3 - x_min) / (x_max - x_min)
        
        Xkmeans_3 = kmeans.fit(Xemb_3)

        for j in range(nclusters):
            mask = Xkmeans_3.labels_ == j
            n = np.count_nonzero(mask)
            table.append([str(j), n])
            # Find the samples corresponding to this cluster and average them
            cluster_3['mean'].append(np.mean(samples_3[mask],
                                             axis=0).reshape((32,32)))
            cluster_3['std'].append(np.std(samples_3[mask],
                                           axis=0).reshape((32,32)))
    
            
        # print(tabulate(table, headers=['Cluster', 'Training', 'Test', 'Total'],
        #                tablefmt='github'), file=f)
        # print('\n', file=f)
            
    # fname = '/media/sf_Master_thesis/Python_Code/results/stag/cluster/'
    # fname = fname + split + '_test_split/'
    # fig, axs = plt.subplots(2,8, sharex=True, sharey=True, figsize=(22,8), dpi=80)
    # fig.tight_layout()
    # axs[0][0].set_ylabel('Training')
    # axs[1][0].set_ylabel('Test')
    # for i in range(n_classes):
    #     for j in range(8):
    #         axs[0][j].imshow(cluster_train[i*8+j])
    #         axs[0][j].tick_params(axis='both', which='both',
    #                               labelbottom=False, labelleft=False,
    #                               bottom=False, top=False, left=False, right=False)
    #         axs[1][j].imshow(cluster_test[i*8+j])
    #         axs[1][j].tick_params(axis='both', which='both',
    #                               labelbottom=False, labelleft=False,
    #                               bottom=False, top=False, left=False, right=False)
    #     fig.suptitle('Class {:d}'.format(i))
    #     plt.savefig(fname=(fname + 'class' + str(i)))
    