import scipy.io as sio
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


def load_data(direction):
    #path = Path('C:/Users/fabia/Documents/Schweiz/ETH/Master/4_Semester_Spring_2020/Master_thesis/Data_Collection/3kOhm_FB/square_sensor/Movement_direction')
    path = Path('../../Data_Collection/3kOhm_FB/square_sensor/Movement_direction')
    file = direction + '.log'
    full_path = path / file
    
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

mask = np.zeros((32,32))
mask[0:16, 0:16] = 1
mask = mask.reshape((1024,)).astype(np.bool)


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
        mean_p = np.mean(press)
        boost_mask = press > mean_p
        press[boost_mask] = np.array(list(map(lambda x: 4*(x-mean_p),
                                              press[boost_mask])))
        
    return pressure


directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
data_orig = load_data(directions[0])
# Calculate 'center(s) of mass'
data_mod = np.zeros(data_orig.shape)
for idx, frame in enumerate(data_orig):
    if(np.amax(frame) > 1900):
        max_idx = np.argmax(frame)
        max_idx = np.unravel_index(max_idx, frame.shape)
        data_mod[idx][max_idx] = 1

app = QtGui.QApplication([])

w = QtGui.QWidget()
# Create a bunch of imageview widgets
imv_orig = pg.ImageView()
imv_mod = pg.ImageView()
# Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)
# Add imageview widgets to layout
layout.addWidget(imv_orig, 0, 0)
layout.addWidget(imv_mod, 0, 1)
w.show()
w.setWindowTitle('Tactile data square sensor')
lev = (1500, 3600)
imv_orig.setImage(data_orig, xvals=np.linspace(0., data_orig.shape[0]/100,
                  data_orig.shape[0]))#autoHistogramRange=False, levels=lev)
imv_mod.setImage(data_mod, xvals=np.linspace(0., data_mod.shape[0]/100,
                 data_mod.shape[0]))#autoHistogramRange=False, levels=lev)

if __name__ == "__main__":
    QtGui.QApplication.instance().exec_()