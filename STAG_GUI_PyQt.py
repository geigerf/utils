#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 6 08:46:57 2020

@author: fabian geiger
"""


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import serial
from serial.tools import list_ports

tactile_img_only = True

class DynamicPlotter():
    """
    Cass to plot sensor data in real time, coming in through a serial port
    """
    def __init__(self, sampleinterval=0.01, size=(800,800)):
        self._interval = int(sampleinterval*1000) # update interval
        # Interpret image data as row-major instead of col-major
        #pg.setConfigOptions(imageAxisOrder='row-major')
        self.app = QtGui.QApplication([])
        # Create window with ImageView widget
        self.win = QtGui.QMainWindow()
        self.win.resize(*size)
        self.imv = pg.ImageView()
        if(tactile_img_only):
            self.imv.ui.histogram.hide()
            self.imv.ui.menuBtn.hide()
            self.imv.ui.roiBtn.hide()
        self.win.setCentralWidget(self.imv)
        self.win.show()
        self.win.setWindowTitle('Tactile data')
        # QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(self._interval)
        # Serial stuff
        self.serial_port = serial.Serial(list_ports.comports()[0].device,
                                         timeout=0.02, baudrate=921600)

    def update_data(self):
        data = self.serial_port.readline()
        data = data.decode()
        if (len(data) > 0):
            # Data is a sensor frame
            # Draw on imageview
            data = data.split('\t')
            data.pop(-1)
            try:
                data = np.array(data).astype(np.float)
                data = np.reshape(data, (32,32))                
                self.imv.setImage(data)#, levels=(1500,1700))
            except Exception as e: 
                print(e)

    def run(self):
        self.app.exec_()
        
    
if __name__ == '__main__':

    gui = DynamicPlotter(sampleinterval=0.01)
    gui.run()
    gui.serial_port.close()
    