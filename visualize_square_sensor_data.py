import scipy.io as sio
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


def load_data(date):
    #path = Path('C:/Users/fabia/Documents/Schweiz/ETH/Master/4_Semester_Spring_2020/Master_thesis/Data_Collection/3kOhm_FB/square_sensor/')
    path = Path('../../Data_Collection/3kOhm_FB/square_sensor/')
    file = 'lotion_footprint.log'
    full_path = path / date / file
    
    tactile_data = []
    line_count = 0
    with open(full_path) as f:
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


def normalize(pressure):
    """
    Scales each array of the given array of arrays to the range [0, 1]
    Only considers values in the same tactile frame
    """
    normalized_p = np.copy(pressure)
    for i, press in enumerate(pressure):
        min_p = np.min(press)
        normalized_p[i] = (press - min_p) / np.max(press - min_p)
    
    return normalized_p
 

def boost(pressure):
    """
    The higher a value is from the mean of the frame, the more it gets boosted.
    The idea is that tactile features are robuster
    """ 
    for press in pressure:
        mean_p = np.mean(press[mask])
        boost_mask = press > mean_p
        press[boost_mask] = list(map(lambda x: 4*(x-mean_p), press[boost_mask]))
        
    return pressure


mask = np.zeros((32,32))
mask[0:16, 0:16] = 1
mask = mask.reshape((1024,)).astype(np.bool)

data1 = load_data('06.08.2020')
data2 = load_data('07.08.2020')
data3 = load_data('11.08.2020')

app = QtGui.QApplication([])

w = QtGui.QWidget()
# Create a bunch of imageview widgets
imv1 = pg.ImageView()
imv2 = pg.ImageView()
imv3 = pg.ImageView()
# Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)
# Add imageview widgets to layout
layout.addWidget(imv1, 0, 0)
layout.addWidget(imv2, 0, 1)
layout.addWidget(imv3, 0, 2)
w.show()
w.setWindowTitle('Tactile data square sensor')
lev = (1500, 3600)
imv1.setImage(data1, xvals=np.linspace(0., data1.shape[0]/100, data1.shape[0]),
              autoHistogramRange=False, levels=lev)
imv2.setImage(data2, xvals=np.linspace(0., data2.shape[0]/100, data2.shape[0]),
              autoHistogramRange=False, levels=lev)
imv3.setImage(data3, xvals=np.linspace(0., data3.shape[0]/100, data3.shape[0]),
              autoHistogramRange=False, levels=lev)

if __name__ == "__main__":
    QtGui.QApplication.instance().exec_()