import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

path = Path('../../Power_Measurements/stag_lowPower.csv')
data = np.genfromtxt(path, delimiter=',', skip_header=7)#, skip_footer=100)

num_ch = data.shape[1] - 1
ch_names = ['IMU', 'Microcontroller', 'Readout Circuit', 'Discovery Board']
samples = data[:,0]
time = samples*0.00098394
# Cut first part of the plot because it's ugly -> take 15s - 110s
mask = np.logical_and(15 < time, time < 110)
time = time[mask]
time = time - min(time)
tmax = max(time)
v = []
i = []
for j in range(num_ch//2):
    v.append(data[:, j*2+1][mask])
    i.append(data[:, j*2+2][mask])

# Plot colors
blue = (57/255, 106/255, 177/255)
orange = (218/255, 124/255, 48/255)
green = (62/255, 150/255, 81/255)
red = (204/255, 37/255, 41/255)
black = (83/255, 81/255, 84/255)
purple = (107/255, 76/255, 154/255)
dark_red = (146/255, 36/255, 40/255)
olive = (148/255, 139/255, 61/255)

figsize = [8, 4.5]
# Current
'''
for j in range(num_ch//2):
    plt.figure(figsize=(8,4))
    #plt.title('Current channel ' + str(j+1))
    plt.title('Current ' + ch_names[j])
    current = i[j]
    ylabel = 'Current [A]'
    if(np.mean(current) < 1):
        current = 1000*current
        ylabel = 'Current [mA]'
    plt.plot(time, current, color=black)
    #average = np.full(time.shape, np.mean(current))
    #plt.plot(time, average)
    #plt.legend(['Trace', 'Average'])
    plt.xlabel('Time [s]')
    plt.xlim([0, 120])
    plt.ylabel(ylabel)
    plt.ylim([0, max(current)*1.05])
    plt.grid()
    plt.show()
'''
# Zoom into plots
# zoom areas run phase
ylims = [(3.5, 8.9),(87, 145), (4.5, 14), (155, 173)]
x1, x2 = 51.9, 53.1
# zoom areas standby phase
ylims_sb = [(0, 0.7)]
x1_sb, x2_sb = 31.09, 31.21

# Power consumption in run and standby
for j in range(3):
    run_mask = np.logical_and(47 < time, time < 55)
    run_pow = np.mean(v[j][run_mask]*i[j][run_mask])
    if(run_pow < 0.001):
        run_pow = run_pow*1000000
        run_unit = 'µW'
    elif(run_pow > 1):
        run_unit = 'W'
    else:
        run_unit = 'mW'
        run_pow = run_pow*1000
    
    sb_mask = np.logical_and(17 < time, time < 43)
    sb_pow = np.mean(v[j][sb_mask]*i[j][sb_mask])
    if(sb_pow < 0.001):
        sb_pow = sb_pow*1000000
        sb_unit = 'µW'
    elif(sb_pow > 1):
        sb_unit = 'W'
    else:
        sb_unit = 'mW'
        sb_pow = sb_pow*1000
    print('\nAverage power consumption of '+ch_names[j]+'\n\tRun:', run_pow, run_unit+'\n\tStandby:', sb_pow, sb_unit)

## IMU
idx = 0
# Main plot standby phase
fig, ax = plt.subplots(figsize=figsize)
ax.set_title('Current ' + ch_names[idx])
ax.plot(time, i[idx]*1000, color=blue)
ax.grid()
ax.set_xlabel('Time [s]')
ax.set_xlim([0, tmax])
ax.set_ylabel('Current [mA]')
ax.set_ylim([0, max(i[idx]*1000)*1.05])
ax.annotate('Standby', xy=(0.33, 0.03),  xycoords='axes fraction',
            xytext=(0.3, 0.219), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='right', verticalalignment='bottom')
ax.annotate('Run', xy=(0.55, 0.7),  xycoords='axes fraction',
            xytext=(0.35, 0.75), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='left', verticalalignment='center')
# Inset plot            
axins = inset_axes(ax, width="33%", height='50%', loc=5) # location: center-right
axins.plot(time, i[idx]*1000)
axins.grid()
axins.set_xlim(x1_sb, x2_sb) # apply the x-limits
axins.set_ylim(ylims_sb[idx]) # apply the y-limits
axins.set_xticks([x1_sb+0.01, x2_sb-0.01])
#axins.xaxis.set_visible(False)
# Put inset into main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.draw()
plt.show()

# Main plot run phase
fig, ax = plt.subplots(figsize=figsize)
ax.set_title('Current ' + ch_names[idx])
ax.plot(time, i[idx]*1000, color=blue)
ax.grid()
ax.set_xlabel('Time [s]')
ax.set_xlim([0, tmax])
ax.set_ylabel('Current [mA]')
ax.set_ylim([0, max(i[idx]*1000)*1.05])
ax.annotate('Standby', xy=(0.33, 0.03),  xycoords='axes fraction',
            xytext=(0.3, 0.219), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='right', verticalalignment='bottom')
ax.annotate('Run', xy=(0.55, 0.7),  xycoords='axes fraction',
            xytext=(0.35, 0.75), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='left', verticalalignment='center')
# Inset plot            
axins = inset_axes(ax, width="33%", height='50%', loc=5) # location: center-right
axins.plot(time, i[idx]*1000)
axins.grid()
axins.set_xlim(x1_sb+20, x2_sb+20) # apply the x-limits
axins.set_ylim(ylims[idx]) # apply the y-limits
axins.set_xticks([x1_sb+20.01, x2_sb+19.99])
#axins.xaxis.set_visible(False)
# Put inset into main plot
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.draw()
plt.show()

## MCU
idx = 1
# Main plot
fig, ax = plt.subplots(figsize=figsize)
ax.set_title('Current ' + ch_names[idx])
ax.plot(time, i[idx]*1000, color=blue)
ax.grid()
ax.set_xlabel('Time [s]')
ax.set_xlim([0, tmax])
ax.set_ylabel('Current [mA]')
ax.set_ylim([0, max(i[idx]*1000)*1.05])
ax.annotate('Standby', xy=(0.3, 0.001),  xycoords='axes fraction',
            xytext=(0.3, 0.19), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='center', verticalalignment='top')
ax.annotate('Run', xy=(0.55, 0.75),  xycoords='axes fraction',
            xytext=(0.35, 0.72), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='left', verticalalignment='center')
# Inset plot            
axins = inset_axes(ax, width="33%", height='50%', loc=5) # location: center-right
axins.plot(time, i[idx]*1000)
axins.grid()
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(ylims[idx]) # apply the y-limits
axins.set_xticks([x1+0.1, x2-0.1])
#axins.xaxis.set_visible(False)
# Put inset into main plot
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.draw()
plt.show()

## Readout circuit
idx = 2
# Main plot
fig, ax = plt.subplots(figsize=figsize)
ax.set_title('Current ' + ch_names[idx])
ax.plot(time, i[idx]*1000, color=blue)
ax.grid()
ax.set_xlabel('Time [s]')
ax.set_xlim([0, tmax])
ax.set_ylabel('Current [mA]')
ax.set_ylim([0, max(i[idx]*1000)*1.05])
'''ax.annotate('Standby', xy=(0.3, 0.001),  xycoords='axes fraction',
            xytext=(0.3, 0.19), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='center', verticalalignment='top')'''
ax.annotate('Run', xy=(0.55, 0.6),  xycoords='axes fraction',
            xytext=(0.35, 0.5), textcoords='axes fraction',
            arrowprops=dict(color='black', shrink=0.05, width=2, headwidth=5),
            horizontalalignment='center', verticalalignment='center')
# Inset plot            
axins = inset_axes(ax, width="33%", height='50%', loc=4) # location: lower-right
axins.plot(time, i[idx]*1000)
axins.grid()
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(ylims[idx]) # apply the y-limits
axins.set_xticks([x1+0.1, x2-0.1])
#axins.xaxis.set_visible(False)
# Put inset into main plot
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.draw()
plt.show()

