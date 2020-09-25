#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 08:23:38 2020

@author: fabian
"""


from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import matplotlib.pyplot as plt


def load_data(path):
    
    mask = np.zeros((32,32))
    mask[0:16, 0:16] = 1
    mask = mask.reshape((1024,)).astype(np.bool)
    
    tactile_data = []
    line_count = 0
    with open(path) as f:
        # Expect data separated by \t and if it belongs to the same block.
        # Blocks are separated by \n
        for line in iter(f.readline, ''):
            line = line.split('\t')
            line.pop(0) # get rid of frame number at the beginning
            line.pop(-1) # get rid of '\n' at the end
            if(len(line) != 1024):
                print('Error on line', line_count)
            else:
                frame = np.array(line)
                tactile_data.append(frame[mask])
            line_count += 1

    tactile_data = np.array(tactile_data).astype(np.uint16)
    data = tactile_data.reshape((-1, 16, 16))
    return data

folder = Path('../../Data_Collection/3kOhm_FB/square_sensor/Slip/')
#file = 'screwdriver.log'
file = 'coke.log'

data = load_data(folder / file)

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
imv = pg.ImageView()
imv.ui.histogram.hide()
imv.ui.menuBtn.hide()
imv.ui.roiBtn.hide()
win.setCentralWidget(imv)
win.resize(1000,1000)
win.show()
win.setWindowTitle('Slip experiment')
imv.setImage(data, xvals=np.linspace(0., data.shape[0]/100, data.shape[0]))
              #autoHistogramRange=False, levels=lev)
              
if __name__ == "__main__":
    QtGui.QApplication.instance().exec_()
    
    