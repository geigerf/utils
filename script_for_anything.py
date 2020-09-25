#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:56:13 2020

@author: fabian geiger
"""


if False:
    ## Plot accuracy vs. number of classes
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    n_dropped = [0, 6, 9, 11, 14]
    top1 = [[39.7, 42.2, 29.5],
            [40.225, 49.948, 27.570], [47.180, 47.898, 34.830],
            [45.634, 52.877, 33.673], [54.542, 53.513, 41.615]]
    top3 = [[63.1, 66.0, 53.2],
            [67.042, 76.064, 51.968], [73.069, 73.869, 61.440],
            [72.090, 77.982, 60.250], [80.518, 80.816, 69.908]]
    
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(7, 6))
    plt.title('Dropping low accuracy classes')
    plt.errorbar(n_dropped, np.mean(top1, 1), yerr=np.std(top1, 1), capsize=3,
                 marker='o', color='k', linestyle=':', label='Top 1')
    plt.errorbar(n_dropped, np.mean(top3, 1), yerr=np.std(top3, 1), capsize=3,
                 marker='o', color='r', linestyle=':', label='Top 3')
    plt.xlabel('Number of dropped classes')
    plt.xlim(-1, 15)
    plt.xticks(np.arange(-1, 16))
    plt.ylim(0, 100)
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel('Classification accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    

if False:
    ## Save validation data for STM Cube Mx
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    import numpy as np
    import scipy.io as sio
    import sys
    #code_dir = '/home/fabian/Documents/Master_thesis/Python_Code/pytorch_stag/STAG_slim_minimal'
    code_dir = '../pytorch_stag/STAG_slim_minimal'
    sys.path.insert(0, code_dir)

    from shared.dataset_tools import load_data
    
    #metafile = '/home/fabian/Documents/Master_thesis/Research/STAG_MIT/classification_lite/metadata.mat'
    metafile = '../../Research/STAG_MIT/classification_lite/metadata.mat'
    data_set = load_data(metafile, split='recording', undersample=True)
    
    x1, x2, x3 = data_set[0], data_set[2], data_set[4]
    y1, y2, y3 = data_set[1], data_set[3], data_set[5]
    
    np.savetxt('data1_full.csv', x1, delimiter=',', fmt='%f')
    np.savetxt('data2_full.csv', x2, delimiter=',', fmt='%f')
    np.savetxt('data3_full.csv', x3, delimiter=',', fmt='%f')
    
    binarizer = LabelBinarizer()
    y1 = binarizer.fit_transform(y1).astype(np.int8)
    y2 = binarizer.fit_transform(y2).astype(np.int8)
    y3 = binarizer.fit_transform(y3).astype(np.int8)
    
    np.savetxt('label1_full.csv', y1, delimiter=',', fmt='%d')
    np.savetxt('label2_full.csv', y2, delimiter=',', fmt='%d')
    np.savetxt('label3_full.csv', y3, delimiter=',', fmt='%d')
    
    x2_large, x2_small, y2_large, y2_small = train_test_split(x2, y2,
                                                              test_size = 0.01,
                                                              stratify=y2)
    np.savetxt('data2_small.csv', x2_small, delimiter=',', fmt='%f')
    np.savetxt('label2_small.csv', y2_small, delimiter=',', fmt='%d')


if False:
    ## Save a small amount of validation data to test real data on the model
    ## trained on MIT data
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    import numpy as np
    import sys
    #code_dir = '/home/fabian/Documents/Master_thesis/Python_Code/results'
    code_dir = '../results'
    sys.path.insert(0, code_dir)

    from real_data import load_real_data
    
    file = 'empty_hand.log'
    emptyhand_data, emptyhand_labels = load_real_data(file, object_id = 0)
    file = 'mug.log'
    mug_data, mug_labels = load_real_data(file, object_id = 16)
    
    real_data = np.concatenate((emptyhand_data, mug_data))
    real_labels = np.concatenate((emptyhand_labels, mug_labels))
    
    binarizer = LabelBinarizer()
    binarizer.fit(range(27))
    real_labels = binarizer.transform(real_labels).astype(np.int8)
    
    np.savetxt('real_data.csv', real_data, delimiter=',', fmt='%f')
    np.savetxt('real_labels.csv', real_labels, delimiter=',', fmt='%d')
    

if False:
    ## Plot intra-session accuracy
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    plt.rcParams.update({'font.size': 18})
    
    labels = ['Session 1', 'Session 2', 'Session 3']
    top1_mean = [93.4, 89.2, 88.3]
    top1_std = [0.597, 0.733, 0.952]
    top3_mean = [98.2, 96.2, 96.8]
    top3_std = [0.262, 0.450, 0.398]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, top1_mean, width=width, yerr=top1_std,
                    color='k', capsize=8, label='Top 1')
    rects2 = ax.bar(x + width/2, top3_mean, width=width, yerr=top3_std,
                    color='r', capsize=8, label='Top 3')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0,100)
    ax.set_title('Intra-session accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()
    
    
if False:
    ## Plot 7-fold cross validation with random split
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    plt.rcParams.update({'font.size': 14})
    
    top1_mean = 98.859
    top1_std = 0.112
    top3_mean = 99.826
    top3_std = 0.026
    
    x = 1  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width/2, top1_mean, width=width, yerr=top1_std,
                    color=(211/255,94/255,96/255), capsize=8, label='Top 1')
    rects2 = ax.bar(x + width/2, top3_mean, width=width, yerr=top3_std,
                    color=(128/255,133/255,133/255), capsize=8, label='Top 3')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0,100)
    ax.set_xlim(0, 2)
    ax.set_title('Cross validation accuracy')
    ax.set_xticks([0, 1, 2])
    #ax.set_xticklabels(labels)
    ax.grid()
    ax.legend()
    plt.savefig('7foldCV_realData.png')
    
    
if False:
    ## Save a threshold based on the empty hand sensor data
    import numpy as np
    from pathlib import Path
    
    
    def load_real_data(filename, path_to_dir, object_id):
        """
        Utility function to load the data of one object interaction and add metadata
        like valid flag, object ID
        """
        IMU_data = []
        tactile_data = []
        data_desc = []
        line_count = 1
        
        with open(path_to_dir / (filename + '.log')) as f:
            # I expect data separated by \t and if it belongs to the same block. Blocks are separated by \n
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
                    if(len(line) != 1024):
                        print('Error in file', filename,'on line', line_count)
                    tactile_data.append(line)
                line_count += 1
                    
        object_ids = np.full((len(tactile_data),), object_id)
                    
        IMU_data = np.array(IMU_data).astype(np.uint16)
        tactile_data = np.array(tactile_data).astype(np.uint16)
        # Normalize pressure data to range 0,1
        #tactile_data = np.clip((tactile_data-1820)/(3100-1820), 0.0, 1.0)
        # Per frame normalization
        #tactile_data = normalize(tactile_data)
        
        return tactile_data, IMU_data, object_ids
    
    
    #data_folder = Path('/home/fabian/Documents/Master_thesis/Data_Collection/3kOhm_FB')
    data_folder = Path('../../Data_Collection/3kOhm_FB')
    #data_folder = Path('/home/fabian/Documents/Master_thesis/Data_Collection/3kOhm_FB/Recording_session_00')
    
    tactile_threshold, _, _ = load_real_data('empty_hand_large', data_folder, 5)
    tactile_threshold = np.amax(tactile_threshold, axis=0, keepdims=True)
    
    mask = np.array([np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))])
    mask = mask.reshape((1,1024)).astype(np.bool)
    
    # To exclude the non-existing sensels from the threshold
    tactile_threshold[np.logical_not(mask)] += 100
    
    save_path = data_folder / 'tactile_threshold'
    np.savetxt(save_path, tactile_threshold, fmt='%u', delimiter=', ')
    
if True:
    import matplotlib.pyplot as plt
    import numpy as np
    ## Plot inter-session variability
    plt.rcParams.update({'font.size': 14})
    
    labels = ['Sess.1', 'Sess. 2', 'Sess. 3', 'Sess. 4', 'Sess. 5']
    top1 = [[25.082, 27.818, 25.286, 27.664, 27.211, 24.872, 26.379, 24.104],
            [44.680, 42.896, 41.046, 43.767, 42.904, 41.614, 42.485, 41.859],
            [49.029, 50.972, 49.016, 49.975, 47.851, 49.498, 49.802, 50.912],
            [40.685, 41.357, 42.136, 43.196, 40.623, 42.218, 43.133, 41.409],
            [26.727, 28.625, 28.996, 29.081, 27.624, 27.943, 30.771, 28.897]]
    top3 = [[50.818, 50.621, 47.780, 51.340, 52.390, 51.272, 50.646, 48.089],
            [67.763, 69.885, 68.558, 69.801, 68.693, 67.462, 68.861, 68.391],
            [74.757, 79.196, 78.766, 78.330, 76.675, 78.592, 78.617, 77.802],
            [68.568, 67.722, 69.319, 69.681, 67.159, 68.782, 69.254, 68.639],
            [49.774, 51.498, 51.360, 52.758, 49.970, 51.188, 52.566, 53.066]]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(8,4.5))
    rects1 = ax.bar(x - width/2, np.mean(top1, axis=1), width=width, yerr=np.std(top1, axis=1),
                    color=(211/255,94/255,96/255), capsize=8, label='Top 1')
    rects2 = ax.bar(x + width/2, np.mean(top3, axis=1), width=width, yerr=np.std(top3, axis=1),
                    color=(128/255,133/255,133/255), capsize=8, label='Top 3')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0,100)
    ax.set_title('Inter-session accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid()
    plt.savefig('intersession_realData.png')
    
    
if False:
    ## Plot one example frame
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    mask = np.array([np.ones(32), np.ones(32), np.ones(32),np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))])

    fig, ax = plt.subplots(figsize=(5, 5), frameon=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    sns.heatmap(np.transpose(mask), square=True, ax=ax, cbar=False, cmap='Greys', linewidths=0.5, linecolor='grey')
    plt.show()