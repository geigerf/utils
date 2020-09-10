import scipy.io as sio
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


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
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))]).astype(np.bool)
mask = mask.reshape((1024,))


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


#filename = Path('C:/Users/fabia/Documents/Schweiz/ETH/Master/4_Semester_Spring_2020/Master_thesis/Data_Collection/3kOhm_FB/data_MT_FabianGeiger_5sess.mat')
filename = Path('../../Data_Collection/3kOhm_FB/data_MT_FabianGeiger_5sess.mat')

data = sio.loadmat(filename, squeeze_me=True)
pressure = data['tactile_data']
# Scale data to the range [0, 1]
pressure = np.clip((pressure.astype(np.float32)-1500)/(2700-1500), 0.0, 1.0)
#pressure = normalize(pressure.astype(np.float32))
#pressure = np.exp2(pressure)
#pressure = np.clip((pressure-1), 0.0, 1.0)
pressure = boost(pressure)
pressure = np.clip(pressure, 0.0, 1.0)
object_id = data['object_id']

pressure[:, ~mask] = 0.0

num_sessions = len(np.unique(data['session_id']))
x = []
y = []
sessions = data['session_id']
for i in range(num_sessions):
    session_mask = sessions == i
    x.append(pressure[session_mask])
    y.append(object_id[session_mask])
    
app = QtGui.QApplication([])
# Create a top-level widget to hold everything
w = QtGui.QWidget()
# Create a bunch of imageview widgets
imv1 = pg.ImageView()
imv2 = pg.ImageView()
imv3 = pg.ImageView()
imv4 = pg.ImageView()
imv5 = pg.ImageView()
# Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)
# Add imageview widgets to layout
layout.addWidget(imv1, 0, 0)
layout.addWidget(imv2, 0, 1)
layout.addWidget(imv3, 1, 0)
layout.addWidget(imv4, 1, 1)
layout.addWidget(imv5, 0, 2)
# Show top-level widget
w.show()

# Get data from a specific class
obj_id = 9
data = []
for i in range(num_sessions):
    obj_mask = y[i] == obj_id
    data.append(x[i][obj_mask].reshape((-1, 32, 32)))
    
# Display the data and assign each frame a time value
imv1.setImage(data[0], xvals=np.linspace(0., data[0].shape[0]/100, data[0].shape[0]), levels=(0, 1))
imv2.setImage(data[1], xvals=np.linspace(0., data[1].shape[0]/100, data[1].shape[0]), levels=(0, 1))
imv3.setImage(data[2], xvals=np.linspace(0., data[2].shape[0]/100, data[2].shape[0]), levels=(0, 1))
imv4.setImage(data[3], xvals=np.linspace(0., data[3].shape[0]/100, data[3].shape[0]), levels=(0, 1))
imv5.setImage(data[4], xvals=np.linspace(0., data[4].shape[0]/100, data[4].shape[0]), levels=(0, 1))

if __name__ == "__main__":
    QtGui.QApplication.instance().exec_()