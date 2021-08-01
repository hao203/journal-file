import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from scipy import signal

plt.figure(figsize=(1, 1))
fig = plt.gcf()
csv = pd.read_csv(r'C:\Users\Eliot Drizzle\Documents\data\PPG.csv', low_memory=False)
data = csv.iloc()[:]
_PPG = list(data['PPG'])
ABP = data['ABP']

def smooth(a, WSZ):
    out0 = np.convolve(a,np.ones(WSZ,dtype=int), 'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

def cwt(data, path, name):
    plt.rcParams['savefig.dpi'] = 224
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    
    t = np.linspace(0, 1, len(data), endpoint=False)
    cwtmatr1, freqs1 = pywt.cwt(data, np.arange(1, len(data)), 'cgau1')
    plt.contourf(t, freqs1, abs(cwtmatr1))
    
    fig.savefig(path + '%s.jpg' % name)
    plt.clf()

def meanBP(indexes, base):
    BPs = []
    for index in indexes:
        BPs.append(ABP[base+index])
    return np.mean(BPs)

# pre-process
s = smooth(_PPG, len(_PPG) - 1)
PPG = []
for (index, _) in enumerate(_PPG):
    PPG.append(_PPG[index] - s[index])

total = 311000
interval = 300
SBPs = []
for i in range(0,total,interval):
    SBPs.append(meanBP(signal.find_peaks(ABP[i:i+interval])[0], 0))

index = 0
pre = 'PPG_'
for i in range(0,total,interval):
    if SBPs[index] < 120.:
        cwt(PPG[i:i+interval], r'C:\Users\Eliot Drizzle\Documents\data\Normal\\', pre + str(i))
    else:
        cwt(PPG[i:i+interval], r'C:\Users\Eliot Drizzle\Documents\data\Abnormal\\', pre + str(i))
    index += 1