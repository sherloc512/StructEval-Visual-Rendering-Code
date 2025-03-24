#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2015 Massachusetts Institute of Technology

"""Example plots using Matplotlib.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(time, xdata, xlbl, ylbl, ttl):
    """Plot a simple time series.
    """
    # Create figure.
    f = plt.figure()

    # Create axis to plot on.
    ax = f.add_subplot(111)

    # Plot!
    p = ax.plot(time, xdata, "r--", label=ylbl)

    # Set some plot properties.
    ax.grid(True)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_title(ttl)
    ax.axis("tight")

    plt.legend(loc='best')

    return f, ax, p

def label_bars(ax, rects):
    """Plot numbers on top of bars in bar plot.
    Taken from http://matplotlib.org/examples/api/barchart_demo.html"""
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                '%d' % int(height),
                ha='center', va='bottom')

def plot_bar_chart(data, lbls, xlbl, ylbl, ttl):
    """Plot a bar chart.
    """
    # Create figure.
    f = plt.figure()

    # Create axis to plot on.
    ax = f.add_subplot(111)

    # Create bars.
    bar_width = 0.35
    bar_x = np.array(range(len(data)))

    # Plot!
    b = ax.bar(bar_x, data, bar_width, color="b")

    # Set some plot properies.
    ax.grid(True)

    ax.set_xlabel(xlbl)
    ax.set_xticks(bar_x + bar_width/2)
    ax.set_xticklabels(lbls)

    ax.set_ylabel(ylbl)
    ax.set_title(ttl)

    ax.legend(b, lbls, loc="best")

    label_bars(ax, b)

    return f, ax, b

def main():
    """Make some example plots.
    """
    # Uncomment to see plots as they're produced.
    # plt.ion()
    output_dir = "./"

    # Simple timeseries.
    time = np.linspace(0, 2*np.pi)
    data = np.sin(time)
    f_time, ax_time, p_time = plot_time_series(time, data, "Time [sec]",
                                               "sin(x)", "TimeSeries")
    f_time.savefig(os.path.join(output_dir, "time_series.eps"))

    # Simple bar plot.
    bar_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    bar_lbls = ["A", "B", "C", "D", "E"]
    f_bar, ax_bar, p_bar = plot_bar_chart(bar_data, bar_lbls, "Category",
                                          "Count", "BarChart")
    f_bar.savefig(os.path.join(output_dir, "bar_chart.eps"))


if __name__ == '__main__':
    main()
