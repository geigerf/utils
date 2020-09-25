#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:54:21 2020

@author: fabian geiger
"""


import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot Results.')
parser.add_argument('--experiment', type=str, default='stag',
                    help="Name of the current experiment.")
parser.add_argument('--plot', type=str,
                    help="Name of the plot.")
parser.add_argument('--network', type=str, default='stag',
                    help="Name of the current network.")
args = parser.parse_args()

results = '/home/fabian/Documents/results/' + args.network + '/'


def plot_results(experiment):
    """
    Utility function to plot evolution of training/test loss and precision
    """

    data = np.load(results + experiment + '_history.npy', allow_pickle=True)
    trainloss = np.array(data[0])
    trainprec = np.array(data[1])
    testloss = np.array(data[2])
    testprec = np.array(data[3])
    
    res = data[4]
    
    print('--------------\nResults:')
    for k,v in res.items():
        print('\t%s: %.3f %%' % (k,v))

    epochs = np.arange(trainloss.size)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(epochs, trainloss, label='train')
    plt.plot(epochs, testloss, label='test')
    plt.xlabel('Epoch')
    plt.xlim(0, epochs[-1] + 1)
    plt.ylim(-0.3, 10)
    plt.ylabel('Cross Entropy Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.subplot(122)
    plt.plot(epochs, trainprec, label='train')
    plt.plot(epochs, testprec, label='test')
    plt.xlabel('Epoch')
    plt.xlim(0, epochs[-1] + 1)
    plt.ylabel('Top 1 Accuracy')
    plt.ylim(-3, 103)
    plt.legend(loc='best')
    plt.grid(True)
    plt.suptitle('Results of experiment '+experiment)
    plt.show()
    return
    

def plot_summary():
    """
    Utility function to recreate the plot from the STAG paper
    """
    
    frames = np.arange(1,9)
    
    top1_orig = np.array([[36.4, 38.7, 38.0, 38.8, 39.2, 38.1, 36.4, 38.5, 36.8, 37.4],
                          [50.0, 51.2, 51.3, 52.0, 49.9, 51.0, 53.2, 55.4, 51.9, 51.4],
                          [59.5, 63.4, 62.3, 61.0, 63.3, 55.0, 64.7, 60.7, 60.3, 60.8],
                          [67.0, 66.5, 64.9, 64.4, 70.7, 65.9, 70.9, 69.7, 60.5, 69.4],
                          [76.3, 63.9, 69.6, 69.3, 72.9, 71.6, 73.1, 71.9, 66.7, 74.0],
                          [67.8, 73.6, 73.5, 73.3, 73.4, 72.9, 68.3, 76.7, 62.8, 73.6],
                          [68.3, 74.9, 72.3, 76.1, 67.8, 68.8, 77.2, 79.3, 74.2, 71.2],
                          [71.2, 71.0, 72.7, 80.3, 68.8, 76.1, 74.8, 70.6, 73.1, 79.8]])
    top3_orig = np.array([[59.6, 61.9, 59.6, 61.9, 60.4, 61.0, 58.7, 59.4, 60.2, 58.9],
                          [73.2, 74.2, 73.5, 74.1, 71.6, 74.7, 75.0, 76.8, 73.7, 74.0],
                          [80.6, 83.5, 82.9, 80.6, 83.3, 76.6, 82.9, 81.1, 80.7, 80.8],
                          [85.3, 86.3, 84.4, 82.2, 86.8, 87.3, 87.6, 87.7, 80.6, 87.3],
                          [90.3, 84.2, 88.6, 84.4, 88.3, 87.3, 88.7, 88.1, 85.2, 88.6],
                          [87.7, 89.6, 87.4, 89.1, 88.4, 87.6, 86.4, 90.9, 82.1, 90.2],
                          [87.5, 90.4, 87.5, 91.6, 86.6, 84.3, 92.1, 91.5, 90.0, 87.4],
                          [85.2, 89.6, 88.4, 93.9, 86.7, 88.3, 90.1, 86.8, 89.4, 92.3]])
    
    top1_cl_orig = np.array([[36.4, 38.7, 38.0, 38.8, 39.2, 38.1, 36.4, 38.5, 36.8, 37.4],
                             [53.8, 55.2, 55.4, 55.7, 52.9, 55.3, 57.6, 58.8, 57.0, 57.3],
                             [65.6, 68.3, 65.9, 63.8, 66.6, 60.0, 68.6, 65.5, 65.7, 65.5],
                             [69.6, 67.7, 62.5, 66.9, 71.3, 61.9, 73.6, 70.2, 65.1, 68.2],
                             [78.7, 68.5, 72.8, 70.7, 73.4, 72.8, 69.8, 73.3, 68.2, 81.0],
                             [62.7, 69.8, 75.6, 73.8, 72.9, 70.1, 68.3, 76.7, 60.8, 67.2],
                             [68.2, 74.3, 72.7, 78.0, 68.1, 69.3, 76.1, 77.9, 71.9, 68.0],
                             [71.9, 69.7, 70.6, 81.6, 62.6, 72.5, 74.6, 68.9, 70.7, 80.2]])
    top3_cl_orig = np.array([[59.6, 61.9, 59.6, 61.9, 60.4, 61.0, 58.7, 59.4, 60.2, 58.9],
                             [76.8, 76.6, 77.6, 76.7, 74.5, 78.6, 79.0, 79.1, 77.4, 79.2],
                             [83.7, 88.1, 84.7, 84.1, 86.7, 77.2, 85.3, 86.0, 84.4, 84.0],
                             [85.4, 88.1, 84.0, 81.7, 86.1, 87.3, 89.3, 89.2, 83.5, 87.5],
                             [89.5, 84.1, 88.6, 84.6, 86.0, 88.4, 90.4, 89.2, 83.3, 92.7],
                             [89.1, 91.6, 89.3, 90.1, 89.3, 86.3, 86.5, 93.2, 79.4, 89.8],
                             [83.8, 87.4, 86.5, 91.1, 82.7, 83.3, 92.2, 94.2, 85.2, 83.4],
                             [81.0, 87.6, 86.1, 96.5, 84.1, 89.4, 85.6, 83.8, 88.3, 92.8]])
    
    top1_slim16 = np.array([[36.2, 35.0, 34.0, 38.3, 36.7, 36.1, 34.0, 35.6, 36.5, 35.1],
                            [49.8, 51.6, 48.1, 50.1, 43.9, 51.5, 50.1, 45.2, 49.0, 49.0],
                            [58.4, 61.5, 56.2, 55.5, 59.8, 56.7, 54.3, 62.4, 57.6, 55.6],
                            [59.1, 65.9, 58.3, 63.6, 61.8, 58.6, 62.2, 63.0, 62.4, 62.3],
                            [65.9, 69.5, 57.9, 68.7, 67.8, 68.9, 61.7, 60.2, 65.5, 66.3],
                            [64.9, 74.5, 72.9, 73.0, 64.3, 62.5, 68.7, 67.3, 66.5, 73.5],
                            [67.9, 70.2, 70.9, 69.8, 68.4, 67.9, 77.9, 67.6, 74.0, 67.4],
                            [79.5, 76.7, 72.8, 66.2, 71.3, 72.7, 74.0, 77.1, 75.5, 70.5]])
    top3_slim16 = np.array([[58.9, 57.9, 57.6, 62.0, 60.9, 58.9, 58.4, 59.5, 60.7, 59.1],
                            [73.3, 74.4, 72.3, 74.3, 69.3, 74.0, 73.7, 69.1, 73.7, 73.5],
                            [81.7, 83.2, 79.2, 80.0, 81.9, 79.3, 76.5, 81.8, 80.3, 79.4],
                            [81.6, 87.0, 80.8, 85.6, 82.9, 81.7, 83.3, 83.2, 84.4, 82.4],
                            [85.9, 87.0, 80.1, 86.6, 86.1, 88.6, 84.8, 82.8, 85.2, 86.8],
                            [86.9, 91.7, 89.4, 89.8, 88.1, 82.5, 87.8, 86.4, 86.9, 89.2],
                            [87.6, 89.6, 88.3, 89.0, 89.0, 86.5, 92.6, 89.0, 90.5, 86.5],
                            [94.7, 91.0, 90.7, 87.1, 90.0, 89.3, 89.5, 91.2, 92.0, 90.4]])
    
    top1_cl_slim16 = np.array([[36.2, 35.0, 34.0, 38.3, 36.7, 36.1, 34.0, 35.6, 36.5, 35.1],
                               [54.2, 55.7, 51.1, 56.8, 47.8, 57.5, 55.5, 49.8, 53.4, 53.4],
                               [63.4, 68.2, 63.4, 62.2, 63.4, 59.6, 60.2, 68.5, 61.3, 61.8],
                               [61.4, 68.0, 59.2, 69.2, 65.2, 58.9, 65.0, 66.8, 59.6, 64.4],
                               [68.6, 72.7, 64.4, 72.8, 72.3, 74.5, 66.3, 65.1, 67.5, 71.3],
                               [65.5, 82.4, 73.5, 74.2, 65.1, 59.9, 72.0, 67.7, 68.6, 77.4],
                               [73.1, 74.4, 73.8, 76.0, 77.0, 70.2, 77.4, 74.2, 75.9, 71.2],
                               [78.0, 77.8, 72.5, 60.3, 78.9, 71.8, 73.1, 83.0, 79.0, 72.2]])
    top3_cl_slim16 = np.array([[58.9, 57.9, 57.6, 62.0, 60.9, 58.9, 58.4, 59.5, 60.7, 59.1],
                               [77.1, 77.3, 75.6, 79.2, 72.7, 79.5, 78.3, 74.2, 77.6, 78.7],
                               [86.5, 85.3, 84.4, 86.0, 85.4, 82.3, 81.8, 85.8, 85.0, 83.6],
                               [82.3, 89.8, 82.7, 91.5, 83.1, 81.7, 83.7, 84.2, 84.9, 81.0],
                               [86.9, 91.0, 86.6, 89.7, 89.5, 88.8, 86.5, 86.8, 87.2, 87.3],
                               [86.6, 96.8, 91.6, 91.9, 88.7, 81.7, 90.4, 85.9, 89.4, 89.3],
                               [88.3, 93.0, 90.0, 91.5, 94.3, 86.4, 92.4, 89.8, 86.5, 86.1],
                               [95.3, 89.4, 89.7, 82.3, 93.4, 88.4, 87.3, 94.5, 92.8, 91.2]])
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8, 4.5))
    plt.title('Original')
    plt.errorbar(frames, np.mean(top1_orig, 1), yerr=np.std(top1_orig, 1), capsize=3,
                 marker='o', color=(204/255, 37/255, 41/255), linestyle=':', label='Top 1 random')
    #plt.errorbar(frames, np.mean(top3_orig, 1), yerr=np.std(top3_orig, 1), capsize=3,
    #             marker='o', color=(83/255, 81/255, 84/255), linestyle=':', label='Top 3 random')
    #plt.errorbar(frames, np.mean(top1_cl_orig, 1), yerr=np.std(top1_orig, 1), capsize=3,
    #             marker='o', color=(204/255, 37/255, 41/255), label='Top 1 clustering')
    #plt.errorbar(frames, np.mean(top3_cl_orig, 1), yerr=np.std(top3_orig, 1), capsize=3,
    #             marker='o', color=(83/255, 81/255, 84/255), label='Top 3 clustering')
    plt.xlabel('Number of input frames, N')
    plt.xlim(0, 9)
    plt.xticks(np.arange(10))
    plt.ylim(0, 100)
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel('Classification accuracy (%)')
    #plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.figure(figsize=(8, 4.5))
    plt.title('Adapted vs. Original')#'Adapted')
    plt.errorbar(frames, np.mean(top1_slim16, 1), yerr=np.std(top1_slim16, 1), capsize=3,
                 marker='o', color=(204/255, 37/255, 41/255), linestyle=':', label='Adapted')
    #plt.errorbar(frames, np.mean(top3_slim16, 1), yerr=np.std(top3_slim16, 1), capsize=3,
    #             marker='o', color=(83/255, 81/255, 84/255), linestyle=':', label='Top 3 random')
    #plt.errorbar(frames, np.mean(top1_cl_slim16, 1), yerr=np.std(top1_slim16, 1), capsize=3,
    #             marker='o', color=(204/255, 37/255, 41/255), label='Top 1 clustering')
    #plt.errorbar(frames, np.mean(top3_cl_slim16, 1), yerr=np.std(top3_slim16, 1), capsize=3,
    #             marker='o', color=(83/255, 81/255, 84/255), label='Top 3 clustering')
    plt.errorbar(frames, np.mean(top1_orig, 1), yerr=np.std(top1_orig, 1), capsize=3,
                 marker='o', color=(153/255, 151/255, 154/255), linestyle=':', label='Original')
    plt.xlabel('Number of input frames, N')
    plt.xlim(0, 9)
    plt.xticks(np.arange(10))
    plt.ylim(0, 100)
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel('Classification accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.savefig('compare_acc.png')
    return


def plot_cm(experiment):
    """
    Utility function to plot the confusion matrices of training and test runs
    """
    
    if experiment == 'slim16':
        resdir = results + 'slim16/'
        for i in range(1,9):
            data = np.load(resdir + 'nf'+str(i) + '_slim16_history.npy',
                           allow_pickle=True)
            cm_train, cm_test, cm_test_cl = data[5], data[6], data[7]
            
            classes = ['Empty hand', 'Allen key set', 'Ball', 'Battery',
                       'Board eraser', 'Bracket', 'Brain', 'Cat', 'Chain',
                       'Clip', 'Coin', 'Empty can', 'Full can', 'Gel',
                       'Kiwano', 'Lotion', 'Mug', 'Multimeter', 'Pen',
                       'Safety glasses', 'Scissors', 'Screw driver',
                       'Spoon', 'Spray can', 'stapler', 'Tape', 'Tea box']
            
            df_cm_train = pd.DataFrame(cm_train.astype(np.int),
                                       classes, classes)
            df_cm_test = pd.DataFrame(cm_test.astype(np.int),
                                      classes, classes)
            df_cm_test_cl = pd.DataFrame(cm_test_cl.astype(np.int),
                                         classes, classes)
            
            
            #sn.set(font_scale=1.1) # for label size
            
            plt.figure(figsize=(9,9))
            plt.title(str(i) + ' frames confusion matrix training')
            hm = sn.heatmap(df_cm_train, annot=True, annot_kws={"size": 8},
                       cbar=False, cmap=plt.get_cmap('Greens'), fmt='d',
                       square=True)
            hm.set_xticklabels(hm.get_xticklabels(), rotation=45,
                               horizontalalignment='right')
            plt.show()
            
            plt.figure(figsize=(9,9))
            plt.title(str(i) + ' frames confusion matrix test random')
            hm = sn.heatmap(df_cm_test, annot=True, annot_kws={"size": 8},
                       cbar=False, cmap=plt.get_cmap('Greens'), fmt='d',
                       square=True)
            hm.set_xticklabels(hm.get_xticklabels(), rotation=45,
                               horizontalalignment='right')
            plt.show()
            
            plt.figure(figsize=(9,9))
            plt.title(str(i) + ' frames confusion matrix test clustering')
            hm = sn.heatmap(df_cm_test_cl, annot=True, annot_kws={"size": 8},
                       cbar=False, cmap=plt.get_cmap('Greens'), fmt='d',
                       square=True)
            hm.set_xticklabels(hm.get_xticklabels(), rotation=45,
                               horizontalalignment='right')
            plt.show()
            
    else:
        data = np.load(results + experiment + '_history.npy', allow_pickle=True)
        cm_train, cm_test, cm_test_cl = data[5], data[6], data[7]
        
        classes = np.array(['Empty hand', 'Allen key set', 'Ball', 'Battery',
                            'Board eraser', 'Bracket', 'Brain', 'Cat', 'Chain',
                            'Clip', 'Coin', 'Empty can', 'Full can', 'Gel',
                            'Kiwano', 'Lotion', 'Mug', 'Multimeter', 'Pen',
                            'Safety glasses', 'Scissors', 'Screw driver',
                            'Spoon', 'Spray can', 'stapler', 'Tape', 'Tea box'])
        
        if cm_train.shape[0] != 27:
            #to_drop = [1, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26]
            to_drop = [0, 1, 5, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 26]
            print('Dropping classes: ', classes[to_drop])
            classes = np.delete(classes, to_drop)

        df_cm_train = pd.DataFrame(cm_train.astype(np.int),
                                   classes, classes)
        df_cm_test = pd.DataFrame(cm_test.astype(np.int),
                                  classes, classes)
        df_cm_test_cl = pd.DataFrame(cm_test_cl.astype(np.int),
                                     classes, classes)
        
        
        #sn.set(font_scale=1.1) # for label size
        
        plt.figure(figsize=(9,9))
        plt.title(experiment + ' confusion matrix test')
        hm = sn.heatmap(df_cm_test, annot=True, annot_kws={"size": 8},
                   cbar=False, cmap=plt.get_cmap('Greens'), fmt='d',
                   square=True)
        hm.set_xticklabels(hm.get_xticklabels(), rotation=45,
                           horizontalalignment='right')
        plt.show()
    return


def plot_random_CV():
    """
    Utility function to plot the accuracy for a random cross validation
    """
    frames = np.arange(1,9)
    
    top1_mean = np.array([86.1, 94.9, 97.6, 98.2, 98.5, 98.1, 97.8, 98.8])
    top1_std = np.array([0.836, 0.968, 0.505, 0.604, 0.482, 1.182, 0.892, 0.957])
    top3_mean = np.array([94.7, 99.0, 99.7, 99.8, 99.9, 99.8, 99.8, 99.9])
    top3_std = np.array([0.398, 0.245, 0.101, 0.098, 0.083, 0.203, 0.136, 0.098])
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(7, 6))
    plt.title('Random split 6 fold cross validation')
    plt.errorbar(frames, top1_mean, yerr=top1_std, capsize=3,
                 marker='o', color='k', linestyle=':', label='Top 1 random')
    plt.errorbar(frames, top3_mean, yerr=top3_std, capsize=3,
                 marker='o', color='r', linestyle=':', label='Top 3 random')
    # plt.errorbar(frames, np.mean(top1_cl_orig, 1), yerr=np.std(top1_orig, 1),
    #              capsize=3,
    #              marker='o', color='k', label='Top 1 clustering')
    # plt.errorbar(frames, np.mean(top3_cl_orig, 1), yerr=np.std(top3_orig, 1),
    #              capsize=3,
    #              marker='o', color='r', label='Top 3 clustering')
    plt.xlabel('Number of input frames, N')
    plt.xlim(0, 9)
    plt.xticks(np.arange(10))
    plt.ylim(0, 100)
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylabel('Classification accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if args.plot == 'comparison':
        plot_summary()
    elif args.plot == 'confusion_matrix':
        plot_cm(args.experiment)
    elif args.plot == 'random_split_CV':
        plot_random_CV()
    else:
        plot_results(args.experiment)
