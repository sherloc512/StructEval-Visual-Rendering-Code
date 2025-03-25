import numpy as np
import math

import matplotlib.pyplot as plt

# sample: sampled data
# result: fft output
def plot(sample, result, sample_rate):
    plt.figure(1)
    a = plt.subplot(211)
    # Deciding the range of the y axis
    r = max(abs(max(sample)), abs(min(sample))) * 1.5
    a.set_ylim([-r, r])
    a.set_xlabel('time [s]')
    a.set_ylabel('sample value [-]')
    ax = np.arange(len(sample)) / (sample_rate)
    a.set_xlim([0, max(ax)])
    plt.plot(ax, sample)
    b = plt.subplot(212)
    b.set_xscale('log')
    b.set_xlabel('frequency [Hz]')
    b.set_ylabel('|amplitude|')
    # Find the fundamental frequency in Hz
    df = 1/(sample_rate * len(result))
    # Find the frequencies in Hz relative to each index of result
    bx = np.fft.fftfreq(len(result))*len(result)*df
    plt.plot(bx, result)
    plt.savefig('sample-graph.png')

# Generate a sampled sine wave
sample = []
for i in range(50):
    sample.append(math.sin(2*math.pi*0))
    sample.append(math.sin(2*math.pi*0.2))
    sample.append(math.sin(2*math.pi*0.4))
    sample.append(math.sin(2*math.pi*0.6))
    sample.append(math.sin(2*math.pi*0.8))

# Generate two overlapped sine waves at different frequencies
#sample = []
#for i in range(500):
#    sample.append(math.sin(i*0.1) + math.sin(5*i*0.1))
    
sample_rate = 200 #hertz
result = abs(np.fft.fft(sample))

plot(sample, result, sample_rate)