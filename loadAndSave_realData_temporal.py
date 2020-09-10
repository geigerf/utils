#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:19:35 2020

@author: fabian geiger
"""


import numpy as np
from pathlib import Path
import scipy.io as sio


def load_real_data(filename, direction_id, data_path, recording_session):
    """
    Utility function to load the data of one object interaction and add metadata
    like valid flag, object ID
    """
    tactile_data = []
    line_count = 1
    
    session = 'Recording_session_0' + str(recording_session) + '_temporal'
    full_path = data_path / session / (filename + '.log')
    with open(full_path) as f:
        # I expect data separated by \t and if it belongs to the same block. Blocks are separated by \n
        for line in iter(f.readline, ''):
            line = line.split('\t')
            line.pop(-1) # get rid of '\n' at the end
            if(len(line) != 1024):
                print('Error in', filename, 'on line', line_count)
            else:
                tactile_data.append(line)
                
            line_count += 1
            
    directions = np.full((len(tactile_data),), direction_id).astype(np.uint8)
    tactile_data = np.array(tactile_data).astype(np.uint16)
    
    return tactile_data, directions


if __name__ == "__main__":
    #data_folder = Path('/home/fabian/Documents/Master_thesis/Data_Collection/3kOhm_FB/Temporal_sequences')
    data_folder = Path('../../Data_Collection/3kOhm_FB/Temporal_sequences')
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    data = {}
    
    tactile_data = []
    direction_id = []
    session_id = []
      
    session = [0, 1, 2, 3]
    for sess in session:
        for i, direction in enumerate(dirs):
            tactile, IMU, ids = load_real_data(direction, i, data_folder, sess)
            if(len(tactile) != 0): # if it is zero, the log file was empty
                tactile_data.append(tactile)
                direction_id.append(ids)
                # Add session flag
                session_id.append(np.full((len(tactile),), sess))
        
    data['tactile_data'] = np.concatenate(tactile_data)
    data['direction_id'] = np.concatenate(direction_id).astype(np.uint8)
    data['session_id'] = np.concatenate(session_id).astype(np.uint8)
    data['directions'] = np.array(dirs)
        
    #session = 'Recording_session_0' + str(session)
    full_path = data_folder / 'temporaldata_MT_FabianGeiger.mat'
    sio.savemat(full_path, data, appendmat=False, do_compression=True)