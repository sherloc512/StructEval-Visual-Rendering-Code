#!/usr/bin/env python
# Extracted from plot_1_exp.html

import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for saved plots
os.makedirs('plots', exist_ok=True)


# Code Block 1
# Code source: Óscar Nájera
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np


N = 100


def main():
    """Plot exponential functions."""
    x = np.linspace(-1, 2, N)
    y = np.exp(x)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("$x$")
    plt.ylabel(r"$\exp(x)$")
    plt.title("Exponential function")
    plt.savefig("plots/exponential_function.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(x, -np.exp(-x))
    plt.xlabel("$x$")
    plt.ylabel(r"$-\exp(-x)$")
    plt.title("Negative exponential\nfunction")
    # To avoid matplotlib text output
    plt.savefig("plots/negative_exponential.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()


import glob
print("\nPlots generated:")
plot_files = glob.glob('plots/*.png')
for i, plot_file in enumerate(sorted(plot_files)):
    print(f"{i+1}. {plot_file}")