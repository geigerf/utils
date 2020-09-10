#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:19:35 2020

@author: fabian geiger
"""


import numpy as np
from pathlib import Path
import scipy.io as sio


def load_real_data(filename, object_id, data_path, recording_session):
    """
    Utility function to load the data of one object interaction and add metadata
    like valid flag, object ID
    """
    IMU_data = []
    tactile_data = []
    data_desc = []
    line_count = 1
    
    session = 'Recording_session_0' + str(recording_session)
    full_path = data_path / session / (filename + '.log')
    with open(full_path) as f:
        # Expect data separated by \t and if it belongs to the same block. Blocks are separated by \n
        for line in iter(f.readline, ''):
            line = line.split('\t')
            line.pop(-1) # get rid of '\n' at the end
            if(len(line) == 7): # is IMU data
                if(line[0].isdigit()):
                    line.pop(0) # get rid of timestamp
                    IMU_data.append(line)
                else:
                    data_desc.append(line)
            else: # is tactile data
                if(len(line) > 1024):
                    line.pop(0) # get rid of timestamp
                if(len(line) != 1024):
                    print('Error in', session, 'file', filename,
                          'on line', line_count)
                else:
                    tactile_data.append(line)
                
            line_count += 1
            
    object_ids = np.full((len(tactile_data),), object_id)
                
    IMU_data = np.array(IMU_data).astype(np.int16)
    tactile_data = np.array(tactile_data).astype(np.uint16)
    if(len(IMU_data) != len(tactile_data)):
        print('Length mismatch in session', session, 'file', filename)
    
    return tactile_data, IMU_data, object_ids


if __name__ == "__main__":
    #data_folder = Path('/home/fabian/Documents/Master_thesis/Data_Collection/3kOhm_FB')
    data_folder = Path('../../Data_Collection/3kOhm_FB')
    tactile_threshold = np.loadtxt(data_folder/'tactile_threshold',
                                   dtype=np.uint16, delimiter=', ')
    # There is a multimeter recording in session 0, 1 but not 2
    objects = ['ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 
                'full_can', 'gel', 'lotion', 'mug', 'pen',
                'safety_glasses', 'scissors', 'screw_driver', 'spray_can',
                'stapler', 'tape']
    
    data = {}
    
    tactile_data = []
    IMU_data = []
    object_id = []
    valid_flag = []
    session_id = []
      
    session = [0, 1, 2, 3, 4]
    for sess in session:
        for i, obj in enumerate(objects):
            tactile, IMU, ids = load_real_data(obj, i, data_folder, sess)
            if(len(tactile) != 0): # if it is zero, the log file was empty
                tactile_data.append(tactile)
                IMU_data.append(IMU)
                object_id.append(ids)
                # Check which frames were actually in touch with an object
                for tac in tactile:
                    if(obj == 'empty_hand'):
                        valid_flag.append(1)
                    else:
                        if(any(tac > tactile_threshold)):
                            valid_flag.append(1)
                        else:
                            valid_flag.append(0)
                    session_id.append(sess)
        
    data['tactile_data'] = np.concatenate(tactile_data)
    data['IMU_data'] = np.concatenate(IMU_data)
    data['object_id'] = np.concatenate(object_id).astype(np.uint8)
    data['valid_flag'] = np.array(valid_flag).astype(np.uint8)
    data['session_id'] = np.array(session_id).astype(np.uint8)
    data['objects'] = np.array(objects)
    data['threshold'] = tactile_threshold
        
    #session = 'Recording_session_0' + str(session)
    full_path = data_folder / 'data_MT_FabianGeiger_5sess.mat'
    sio.savemat(full_path, data, appendmat=False, do_compression=True)