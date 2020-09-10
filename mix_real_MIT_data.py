#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:13:12 2020

@author: fabian geiger
"""


import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn import under_sampling
from pathlib import Path


def mix_real_MIT(kfold=3, seed=333, undersample=True, split='random'):
    """
    Utility function to mix the MIT dataset with the self-acquired dataset
    """
    #realData_path = Path('/home/fabian/Documents/Master_thesis/Data_Collection/3kOhm_FB/data_MT_FabianGeiger_5sess.mat')
    realData_path = Path('../../Data_Collection/3kOhm_FB/data_MT_FabianGeiger_5sess.mat')
    #MITdata_path = Path('/home/fabian/Documents/Master_thesis/Research/STAG_MIT/classification_lite/metadata.mat')
    MITdata_path = Path('../../Research/STAG_MIT/classification_lite/metadata.mat')
    
    # These two lists will contain valid data split into recording sessions
    x = []
    y = []
    
    realData = sio.loadmat(realData_path, squeeze_me=True)
    real_valid_mask = realData['valid_flag'] == 1
    # Scale all data to the range [0, 1]
    realPressure = realData['tactile_data'][real_valid_mask].astype(np.float32)
    realPressure = np.clip((realPressure - 1510)/(3000 - 1510), 0.0, 1.0)
    realObject_id = realData['object_id'][real_valid_mask]
    realSession_id = realData['session_id'][real_valid_mask]
    # Split into sessions
    num_sessions = len(np.unique(realSession_id))
    for i in range(num_sessions):
        session_mask = realSession_id == i
        x.append(realPressure[session_mask])
        y.append(realObject_id[session_mask])

    MITdata = sio.loadmat(MITdata_path, squeeze_me=True)
    MIT_valid_mask = MITdata['hasValidLabel'] == 1
    MITpressure = MITdata['pressure'].reshape((-1, 32*32)).astype(np.float32)
    MITpressure = MITpressure[MIT_valid_mask]
    MITpressure = np.clip((MITpressure - 500)/(650 - 500), 0.0, 1.0)
    MITobject_id = MITdata['objectId'][MIT_valid_mask]    
    # Only use the same objects as in the real data set
    MITobjects = list(MITdata['objects'])
    used_objects = list(map(str.strip, realData['objects']))
    # Each class has three recording IDs that correspond to the different
    # experiment days. There are 81 recording IDs (3*27)
    # 0  - 26 belong to the first recording
    # 27 - 53 belong to the second recording
    # 54 - 81 belong to the third recording
    MITrecording_id = MITdata['recordingId'][MIT_valid_mask]
    for i in range(3):
        # Find valid samples from the different recording days
        recording_mask = np.logical_and(MITrecording_id >= i*27,
                                        MITrecording_id < (i+1)*27)
        
        used_pressure = []
        used_object_id = []
        for i, obj in enumerate(used_objects):
            idx = MITobjects.index(obj)
            used_mask = np.logical_and(MITobject_id == idx, recording_mask)
            used_pressure.append(MITpressure[used_mask])
            used_object_id.append(np.full(len(MITobject_id[used_mask]), i))
            
        x.append(np.concatenate(used_pressure))
        y.append(np.concatenate(used_object_id))
    
    
    if kfold is not None:
        # Decrease the test size if cross validation is used
        test_size = 0.15
    else:
        kfold = 3
        test_size = 0.33

    if(split == 'random'):
        pressure = np.concatenate(x)
        object_id = np.concatenate(y)
        if(undersample):
            us = under_sampling.RandomUnderSampler(random_state=seed,
                                               sampling_strategy='not minority')
            us_pressure, us_object_id = us.fit_resample(pressure, object_id)
            
            pressure, object_id = us_pressure, us_object_id
    
        # Split the already balanced dataset in a stratified way -> training
        # and test set will still be balanced
        train_data, test_data,\
            train_labels, test_labels = train_test_split(pressure, object_id,
                                                         test_size=test_size,
                                                         random_state=seed,
                                                         shuffle=True,
                                                         stratify=object_id)
        #print(train_data.shape, train_labels.shape)
        # This generates a k fold split in a stratified way.
        # Easy way to do k fold cross validation
        skf = StratifiedKFold(n_splits=kfold, shuffle=True,
                              random_state=seed)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        # skf_gen = skf.split(train_data, train_labels)
        
        return train_data, train_labels, test_data, test_labels, skf
    
    elif(split == 'session'):
        return x, y