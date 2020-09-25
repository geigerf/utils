#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:08:30 2020

@author: fabian
"""


import argparse
import numpy as np
import time
import torch
import sys
code_dir = '../pytorch_stag/STAG_slim_minimal'
#code_dir = '../pytorch_stag/STAG_slim_minimal_realData'
sys.path.insert(0, code_dir) # needed to make classes and functions available, maybe an absolute path needs to be provided
#from torchsummary import summary
from classification.CustomDataLoader import CustomDataLoader
from classification.ClassificationModel import ClassificationModel
from shared.dataset_tools import load_data

parser = argparse.ArgumentParser(description='Convert network to ONNX.')
parser.add_argument('--nframes', type=int,
                    help="Number of frames.")
parser.add_argument('--experiment', type=str, default='orig_RCV',
                    help="Name of the current experiment.")
parser.add_argument('--test', type=bool, nargs='?', const=True, default=False,
                    help="If set, the loaded model will be evaluated once.")
args = parser.parse_args()

nframes = args.nframes
nclasses = 17 # Careful about this!
batch_size = 32
if 'orig' in args.experiment:
    nfilters = 64
    dropout = 0.2
elif 'slim16' in args.experiment:
    nfilters = 16
    dropout = 0.4
elif '32' in args.experiment:
    nfilters = 32
    dropout = 0.4
else:
    nfilters = 16
    dropout = 0


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step(data_loader, epoch, model):
    """
    Implements one step through all batches.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        Either training or test data loader.
    epoch : int
        Current training epoch.
    isTrain : bool, optional
        Whether this step is used for training or test. The default is True.
    sinkName : string, optional
        Name of the results. The default is None.

    Returns
    -------
    top1.avg : AverageMeter
        Average value of top1 precision after current step.
    top3.avg : AverageMeter
        Average value of top3 precision after current step.
    losses.avg : AverageMeter
        Average value of loss after current step.
    conf_matrix : numpy array
        Confusion matrix after current step.

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    end = time.time()
    conf_matrix = torch.zeros(nclasses, nclasses).cpu()
    for i, (inputs) in enumerate(data_loader):
        data_time.update(time.time() - end)
      
        inputsDict = {
            'image': inputs[1],
            'pressure': inputs[2],
            'objectId': inputs[3],
            }

        res, loss = model.step(inputsDict, isTrain=False,
                               params = {'debug': True})

        losses.update(loss['Loss'], inputs[0].size(0))
        top1.update(loss['Top1'], inputs[0].size(0))
        top3.update(loss['Top3'], inputs[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Clear line before printing
        sys.stdout.write('\033[K')
        sys.stdout.flush()
        print('{phase}: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'
              .format(epoch, i, len(data_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top3=top3,
                      phase='Test'),
              end='\r', flush=True)

        # Calculate the confusion matrix
        for t, p in zip(inputs[3].view(-1), res['pred'].view(-1)):
            conf_matrix[t.long(), p.long()] += 1
  
    print('')

    return top1.avg, top3.avg, losses.avg, conf_matrix


if __name__ == "__main__":
    #model_dir = '/media/sf_Master_thesis/Python_Code/models/'
    # network = ('nf' + str(nframes) + '_' + args.experiment 
    #            + '/checkpoint.pth.tar')
    #model_dir = '/home/fabian/Documents/Master_thesis/Python_Code/models/'
    model_dir = '../models/'
    network = (args.experiment + '/checkpoint.pth.tar')
    state = torch.load(model_dir + network)
    epoch = state['epoch']

    Model = ClassificationModel(numClasses=nclasses, inplanes=nfilters,
                                dropout=dropout, cuda=False)

    # Load pretrained weights into model
    Model.importState(state)

    if args.test:
        # Test consistency of the loaded model
        #metaFile = '/home/fabian/Documents/Master_thesis/Research/STAG_MIT/classification_lite/metadata.mat'
        metaFile = '../../Research/STAG_MIT/classification_lite/metadata.mat'
        #metaFile = '/media/sf_Master_thesis/Research/STAG_MIT/classification_lite/metadata.mat'
        data_set = load_data(filename=metaFile, split='recording', seed=333,
                             undersample=True)
    
        x = np.array([data_set[0], data_set[2], data_set[4]])
        y = np.array([data_set[1], data_set[3], data_set[5]])
    
        test_data = x[1]
        test_labels = y[1]

        set_size = len(test_labels)
        test_loader = torch.utils.data.DataLoader(
                        CustomDataLoader(test_data.reshape((set_size, 32, 32)),
                                         test_labels, augment=False,
                                         use_clusters=False, nclasses=nclasses,
                                         balance=True, split='test'),
                        batch_size=batch_size, shuffle=True, num_workers=1)
        
        step(data_loader=test_loader, epoch=epoch, model=Model)

    Model.model.eval()

    # Print model summary
    #summary(Model.model, (nframes, 32, 32))

    dummy_input = torch.randn(batch_size, nframes, 32, 32, requires_grad=True)
    #model_name = args.experiment + '_nf' + str(nframes) + '.onnx'
    model_name = args.experiment + '.onnx'
    save_dir = model_dir + '/onnx/' + model_name
    # Convert pytorch model to onnx model by running a dummy input through
    # the network and saving the trace of the dummy input
    torch.onnx.export(Model.model.module, dummy_input, save_dir,
                      verbose=True, opset_version=9,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})

