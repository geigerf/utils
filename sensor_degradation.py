import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


seed = 333
n_classes = 17
plot = True

filename = '../../Data_Collection/3kOhm_FB/data_MT_FabianGeiger_5sess.mat'

data = sio.loadmat(filename, squeeze_me=True)
# Use only frames in which objects were touched
valid_mask = data['valid_flag'] == 1
pressure = data['tactile_data'][valid_mask]
object_id = data['object_id'][valid_mask]
# Scale data to the range [0, 1]
pressure = np.clip((pressure.astype(np.float32)-1510)/(3000-1510), 0.0, 1.0)

num_sessions = len(np.unique(data['session_id']))
x = []
y = []
valid_sessions = data['session_id'][valid_mask]
for i in range(num_sessions):
    session_mask = valid_sessions == i
    x.append(pressure[session_mask])
    y.append(object_id[session_mask])

mean_1 = []
mean_2 = []
mean_3 = []
mean_4 = []
mean_5 = []
for i in range(n_classes):
    mask = y[0] == i
    samples = x[0][mask]
    if(len(samples) != 0):
        mean_1.append(np.mean(samples, axis=0).reshape((32,32)))
    else:
        mean_1.append(np.zeros((32,32)))
    
    mask = y[1] == i
    samples = x[1][mask]
    if(len(samples) != 0):
        mean_2.append(np.mean(samples, axis=0).reshape((32,32)))
    else:
        mean_2.append(np.zeros((32,32)))
        
    mask = y[2] == i
    samples = x[2][mask]
    if(len(samples) != 0):
        mean_3.append(np.mean(samples, axis=0).reshape((32,32)))
    else:
        mean_3.append(np.zeros((32,32)))
    
    mask = y[3] == i
    samples = x[3][mask]
    if(len(samples) != 0):
        mean_4.append(np.mean(samples, axis=0).reshape((32,32)))
    else:
        mean_4.append(np.zeros((32,32)))

    mask = y[4] == i
    samples = x[4][mask]
    if(len(samples) != 0):
        mean_5.append(np.mean(samples, axis=0).reshape((32,32)))
    else:
        mean_5.append(np.zeros((32,32)))

glove_response = [np.mean(mean_2), np.mean(mean_3), np.mean(mean_4), np.mean(mean_5)]
glove_response = glove_response/max(glove_response)
sessions = np.arange(2, 6)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 4.5))
plt.title('Sensor Degradation Glove')
plt.plot(sessions, glove_response, marker='o', color=(204/255, 37/255, 41/255))
plt.xlabel('Session')
plt.xlim(1, 6)
plt.xticks(np.arange(1,7))
plt.ylim(0, 1.1)
#plt.yticks([0, 25, 50, 75, 100])
plt.ylabel('Relative mean response')
plt.grid(True)
#plt.show()
#plt.savefig('glove_degradation.png')

filename = '../../Data_Collection/3kOhm_FB/square_sensor/'
file = '/lotion_footprint.log'
dates = ['06.08.2020', '07.08.2020', '11.08.2020', '13.08.2020']

square_response = [[],[],[],[]]

for i, d in enumerate(dates):
    with open(filename + d + file) as f:
        # Expect data separated by \t and if it belongs to the same block. Blocks are separated by \n
        for line in iter(f.readline, ''):
            line = line.split('\t')
            line.pop(-1) # get rid of '\n' at the end
            if(len(line) > 1024):
                line.pop(0) # get rid of timestamp
            if(len(line) != 1024):
                print('Error in file', d)
            else:
                line = np.array(line).astype(np.float32)
                line = np.clip((line-1510)/(3000-1510), 0.0, 1.0)
                square_response[i].append(np.mean(line))

square_response = np.mean(square_response, axis=1)
square_response = square_response/max(square_response)
sessions = np.arange(1, 5)

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 4.5))
plt.title('Sensor Degradation Square')
plt.plot(sessions, square_response, marker='o', color=(204/255, 37/255, 41/255))
plt.xlabel('Session')
plt.xlim(0, 5)
plt.xticks(np.arange(6))
plt.ylim(0, 1.1)
#plt.yticks([0, 25, 50, 75, 100])
plt.ylabel('Relative mean response')
plt.grid(True)
#plt.show()
#plt.savefig('square_degradation.png')


## Compare both in the same plot
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 4.5))
plt.title('Sensor Degradation')
plt.plot(sessions, square_response, marker='o', color=(204/255, 37/255, 41/255), label='Square')
plt.plot(sessions, glove_response, marker='^', color=(153/255, 151/255, 154/255), label='Glove')
plt.xlabel('Session')
plt.xlim(0, 5)
plt.xticks(np.arange(6))
plt.ylim(0, 1.1)
#plt.yticks([0, 25, 50, 75, 100])
plt.ylabel('Relative mean response')
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig('degradation_comparison.png')