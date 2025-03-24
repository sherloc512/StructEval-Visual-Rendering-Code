#!/usr/bin/env python3

"""sound_plots.py: Simple example using NumPy & matplotlib to generate some
sound waveforms, plot them with their frequency spectra, and write out WAV
files of them."""

import numpy
import matplotlib.pyplot as plt
import wave
import struct

# Oscillators:
# For all of these, 'f' is the frequency per 'n' samples, and 'n' is the total
# number of samples. That is, for f=2, the oscillator will undergo 2 cycles in
# 'n' samples, regardless of 'n'.
# Since 'f' is (cycles / 'n' samples) and sampling rate 'R' is in
# samples / second, f*R/n is then cycles/second - actual frequency in Hz.
sine       = lambda f,n:   numpy.sin(numpy.arange(n) * numpy.pi * 2 * f / float(n))
sawtooth   = lambda f,n:   numpy.mod(numpy.arange(n) * 2.0 * f/n + f/2.0, 2) - 1
square     = lambda f,n:   (sawtooth(f,n) >= 0) * 2.0 - 1
def tri(f,n):
    osc = 2 * sawtooth(f,n) * square(f,n) - 1
    # This needs a phase shift on top:
    sh = n / f / 4.0
    return numpy.concatenate((osc[sh:], osc[:sh]))
whiteNoise = lambda n:     numpy.random.random(n) * 2.0 - 1
sineClip   = lambda f,n,r: numpy.clip(sine(f,n), -r, r)
# sineClip: 'r' sets the point at which to clip the sinewave, in [0,1]; e.g.
# r = 0.8 clips the sinewave at 80% intensity, both top and bottom.

# sineTrunc: Truncated sinewave; t0 and t1 set start and end time for the
# sinewave, as ratios relative to N (0 < t0 < t1 < 1). For instance, t0=0.1
# and t1 = 0.9 means to cut off the sinewave at 10% of the start and end.
def sineTrunc(f, n, t0, t1):
    r = numpy.arange(n) / float(n)
    envelope = (r > t0) * (r <= t1) * 1
    return sine(f,n) * envelope

def plotWithSpectrum(timeSeries, rate=48000):
    """Plot the given waveform (timeSeries), both as time-domain and as its
    frequency-domain spectrum. Returns a matplotlib.figure.Figure object."""
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    n = len(timeSeries)
    # (1) Plot time-domain data:
    timesMsec = numpy.arange(n) * 1000.0 / rate
    axs[0].plot(timesMsec, timeSeries)
    # Limit the X axis to our input samples:
    axs[0].set_xlim([0, max(timesMsec)])
    # Add a little extra on Y axis so peaks are more visible:
    axs[0].set_ylim([-1.2, 1.2])
    axs[0].set_xlabel("Time (ms)")
    axs[0].grid(True)
    # (2) Compute and plot frequency spectrum:
    spectrum = numpy.abs(numpy.fft.rfft(timeSeries)) / n
    specFreq = numpy.fft.rfftfreq(n, 1.0 / rate)
    # Note that for the bar plot, we set the width to the size of each
    # frequency bin:
    axs[1].bar(specFreq, spectrum, width = rate / n, linewidth = 0)
    #axs[1].set_title("Frequency spectrum")
    # Limit the X axis to the given frequencies:
    axs[1].set_xlim([0, max(specFreq)])
    axs[1].set_ylim([0, 1])
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Intensity")
    return fig

def writeWav(data, filename, rate=48000):
    """Write the time-series (as an array in the range [-1,1]) as a WAV file to
    the given filename. Sample rate is 48000 unless overridden."""
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(1)
    wavefile.setframerate(rate)
    width = 2
    wavefile.setsampwidth(width)
    bytes_ = [struct.pack('h', int((2**(8*width - 1) - 1) * s)) for s in data]
    rawBytes = b"".join(bytes_)
    wavefile.writeframes(rawBytes)
    wavefile.close()

# rate = sample rate (samples/second)
rate = 48000
# wavTime = length of written WAV in seconds
wavTime = 2
# plotN = how many samples to plot
plotN = 128
# f = desired frequency in Hz
f = 750

wavN = rate * wavTime
wavF  = f * wavN  / rate
plotF = f * plotN / rate

# plots: Tuple of (waveform to plot, waveform to write to WAV, name)
# I separate the waveforms for plotting and waveforms for writing to WAV
# because the former are far too short to be useful for listening.
plots = (
    (sine(plotF, plotN), sine(wavF, wavN), "Sine, pure"),
    (-sine(plotF, plotN), -sine(wavF, wavN), "Sine, inverted"),
    (sineClip(plotF, plotN, 0.7), sineClip(wavF, wavN, 0.7), "Sine, clipped (70%)"),
    (sineTrunc(plotF, plotN, 0.1, 0.9), sineTrunc(wavF, wavN, 0.1, 0.9), "Sine, truncated"),
    (square(plotF, plotN), square(wavF, wavN), "Square wave"),
    (tri(plotF, plotN), tri(wavF, wavN), "Triangle wave"),
    (sawtooth(plotF, plotN), sawtooth(wavF, wavN), "Sawtooth wave"),
    (whiteNoise(plotN), whiteNoise(wavN), "White Noise"),
)

for wavePlot, waveListen, name in plots:
    print("Writing: \"%s\"" % (name,))
    writeWav(waveListen, name + ".wav", rate)
    p = plotWithSpectrum(wavePlot, rate)
    p.suptitle(name)
    p.savefig(name + ".pdf")
    p.savefig(name + ".png")
    plt.close()
