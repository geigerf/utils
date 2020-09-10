import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#path = Path('C:/Users/fabia/Desktop/stag.csv')
path = Path('C:/Users/fabia/Desktop/stag_lowPower.csv')
data = np.genfromtxt(path, delimiter=',', skip_header=7)#, skip_footer=100)

num_ch = data.shape[1] - 1
ch_names = ['IMU', 'Microcontroller', 'Readout Circuit', 'Discovery Board']
samples = data[:,0]
time = samples*0.00098394
v = []
i = []
for j in range(num_ch//2):
    v.append(data[:, j*2+1])
    i.append(data[:, j*2+2])
    
# Power
for j in range(num_ch//2):
    plt.figure(figsize=(12,4))
    #plt.title('Power channel ' + str(j+1))
    plt.title('Power ' + ch_names[j])
    power = v[j]*i[j]
    ylabel = 'Power [W]'
    if(np.mean(power) < 1):
        power = 1000*power
        ylabel = 'Power [mW]'
    plt.plot(time, power)
    #average = np.full(time.shape, np.mean(power))
    #plt.plot(time, average)
    #plt.legend(['Trace', 'Average'])
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.ylim([0, max(power)*1.05])
    plt.grid()
    plt.show()

# Current
for j in range(num_ch//2):
    plt.figure(figsize=(12,4))
    #plt.title('Current channel ' + str(j+1))
    plt.title('Current ' + ch_names[j])
    current = i[j]
    ylabel = 'Current [A]'
    if(np.mean(current) < 1):
        current = 1000*current
        ylabel = 'Current [mA]'
    plt.plot(time, current)
    #average = np.full(time.shape, np.mean(current))
    #plt.plot(time, average)
    #plt.legend(['Trace', 'Average'])
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.ylim([0, max(current)*1.05])
    plt.grid()
    plt.show()