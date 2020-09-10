#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:06:21 2020

@author: fabian geiger
"""


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import serial
from serial.tools import list_ports
import struct
from pathlib import Path
from PIL import Image
import collections

tactile_img_only = True
#img_path = Path('C:/Users/fabia/Documents/Schweiz/ETH/Master/4_Semester_Spring_2020/Master_thesis/Report/Pictures/objects')
img_path = Path('../../Report/Pictures/objects')
objects = ['ball', 'battery', 'bracket', 'coin', 'coke', 'empty_hand', 
           'coke', 'gel', 'lotion', 'mug', 'pen', 'safety_glasses',
           'scissors', 'screwdriver', 'spraycan', 'stapler', 'tape']
extension = '.jpg'
images = []
for obj in objects:
    img = Image.open(img_path / (obj + extension))
    images.append(np.array(img))
objects = ['Ball', 'Battery', 'Bracket', 'Coin', 'Coke', 'Empty hand', 
           'Coke', 'Gel', 'Lotion', 'Mug', 'Pen', 'Safety glasses',
           'Scissors', 'Screwdriver', 'Spraycan', 'Stapler', 'Tape']    


class ReadUntil:
    """
    Class that reads serial data until a specific sequence more efficiently than
    the built-in read_until function of pyserial
    """
    def __init__(self, s, expected=b'\xff\xff'):
        self.buf = bytearray()
        self.s = s
        self.expected = expected
    
    def readuntil(self):
        # Returns the data WITHOUT the expected sequence
        i = self.buf.find(self.expected)
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2051, self.s.in_waiting))
            data = self.s.read(i)
            # To make the call to this function not unblocking if there are no
            # bytes in the buffer
            if(i == 1 and len(self.buf) == 0):
                return bytearray(b'')
            i = data.find(self.expected)
            if i >= 0:
                r = self.buf + data[:i]
                self.buf[0:] = data[i+len(self.expected):]
                return r
            else:
                self.buf.extend(data)


class DynamicPlotter():
    """
    Cass to plot sensor data in real time, coming in through a serial port
    """
    def __init__(self, sampleinterval=0.0125, size=(1600,800)):
        self._interval = int(sampleinterval*1000) # update interval
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.app = QtGui.QApplication([])
        # Create a top-level widget to hold everything
        self.win = QtGui.QWidget()
        self.win.resize(*size)
        self.layout = QtGui.QGridLayout()
        
        self.pred_buf = collections.deque(maxlen=4)
        self.pred_img = images
        # Create the rest of the necessary widgets
        self.tactile = pg.ImageView()
        if(tactile_img_only):
            self.tactile.ui.histogram.hide()
            self.tactile.ui.menuBtn.hide()
            self.tactile.ui.roiBtn.hide()
            
        #self.prediction = pg.ImageView()
        #self.prediction.ui.histogram.hide()
        #self.prediction.ui.menuBtn.hide()
        #self.prediction.ui.roiBtn.hide()
        self.prediction = QtGui.QLabel('')
        self.prediction.setFont(QtGui.QFont('Times', 30))
        
        self.runBtn = QtGui.QPushButton(' Run ')
        self.runBtn.clicked.connect(self.send_run)
        self.runBtn.setDisabled(False)
        
        self.pauseBtn = QtGui.QPushButton('Pause')
        self.pauseBtn.clicked.connect(self.send_pause)
        self.paused = 1
        self.pauseBtn.setDisabled(True)
        
        self.tactileLbl = QtGui.QLabel('Tactile data')
        self.tactileLbl.setFont(QtGui.QFont('Times', 15))
        
        self.predictionLbl = QtGui.QLabel('Prediction')
        self.predictionLbl.setFont(QtGui.QFont('Times', 15))
        
        self.win.setLayout(self.layout)
        self.layout.addWidget(self.tactile, 1, 0)
        self.layout.addWidget(self.prediction, 1, 1)
        self.layout.addWidget(self.runBtn, 2, 0)
        self.layout.addWidget(self.pauseBtn, 2, 1)
        self.layout.addWidget(self.tactileLbl, 0, 0)
        self.layout.addWidget(self.predictionLbl, 0, 1)
        self.win.show()
        self.win.setWindowTitle('STAG Demo')
        # QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(self._interval)
        # Serial stuff
        self.serial_port = serial.Serial(list_ports.comports()[0].device,
                                         timeout=0.15, baudrate=921600)
        # Read until the sequence b'\xff\xff' is received, which was used as a
        # guard sequence because the combination will never show up in the data
        self.serial_read = ReadUntil(self.serial_port, expected=b'\xff\xff')

    def update_data(self):
        #data = self.serial_read.readuntil()
        data = self.serial_port.read_until(b'\xff\xff')
        #data = self.serial_port.read(2049)
        #if(self.serial_port.in_waiting != 0):
            #print(self.serial_port.in_waiting, 'bytes in buffer')
        if(len(data) > 0):
            if(len(data) != 2051):
                print('Data has length', len(data))
                return
                
            fra = data[0:2048]
            cla = data[2048]
            self.pred_buf.append(cla)
            try:
                frame = []
                for j in range(1024):
                    frame.append(struct.unpack('<H', fra[j*2:(j+1)*2])[0])
                    
                frame = np.array(frame).astype(np.uint16)
                frame = np.reshape(frame, (32,32))                
                self.tactile.setImage(np.transpose(frame))#, levels=(1500,2000))
                if(self.pred_buf.count(cla) == 4): # only change prediction after four consecutive predictions were the same
                    #self.prediction.setImage(self.pred_img[cla])
                    self.prediction.setText(objects[cla])
            except Exception as e:
                print(e)
                print(cla)
                print(self.serial_port.in_waiting)
            
    def send_run(self):
        if(self.paused == 1):
            self.runBtn.setDisabled(True)
            self.pauseBtn.setDisabled(False)
            self.serial_port.write("r".encode())
            self.serial_port.flush()
            #self.timer.start(self._interval)
            self.update_data()
            self.paused = 0
            
    def send_pause(self):
        if(self.paused == 0):            
            self.pauseBtn.setDisabled(True)
            self.runBtn.setDisabled(False)
            self.serial_port.write("p".encode())
            self.serial_port.flush()
            #self.timer.stop()
            self.paused = 1

    def run(self):
        self.app.exec_()
        
    
if __name__ == '__main__':

    demo = DynamicPlotter(sampleinterval=0.05)
    demo.run()
    demo.serial_port.close()
    