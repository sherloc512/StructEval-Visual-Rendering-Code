#!/usr/bin/env python
# Extracted from bland_altman.html

import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for saved plots
os.makedirs('plots', exist_ok=True)


# Code Block 1
import pandas as pd

# Peak expiratory flow rate measurements made using a Wright peak flow meter
# and a mini Wright peak flow meter.
# - https://www-users.york.ac.uk/~mb55/datasets/pefr.dct
# - https://www-users.york.ac.uk/~mb55/datasets/datasets.htm
# - https://www-users.york.ac.uk/~mb55/meas/ba.pdf
df = pd.DataFrame({
    'Wright Mini': [
        512, 430, 520, 428, 500, 600, 364, 380, 658,
        445, 432, 626, 260, 477, 259, 350, 451
    ],
    'Wright Large': [
        494, 395, 516, 434, 476, 557, 413, 442, 650,
        433, 417, 656, 267, 478, 178, 423, 427
    ]
})


# Code Block 2
import pandas as pd

# Peak expiratory flow rate measurements made using a Wright peak flow meter
# and a mini Wright peak flow meter.
# - https://www-users.york.ac.uk/~mb55/datasets/pefr.dct
# - https://www-users.york.ac.uk/~mb55/datasets/datasets.htm
# - https://www-users.york.ac.uk/~mb55/meas/ba.pdf
df = pd.DataFrame({
    'Wright Mini': [
        512, 430, 520, 428, 500, 600, 364, 380, 658,
        445, 432, 626, 260, 477, 259, 350, 451
    ],
    'Wright Large': [
        494, 395, 516, 434, 476, 557, 413, 442, 650,
        433, 417, 656, 267, 478, 178, 423, 427
    ]
})


# Code Block 3
ax = plt.axes()


# Code Block 4
import matplotlib.pyplot as plt

ax = plt.axes()
ax.scatter(df['Wright Large'], df['Wright Mini'])
ax.set_title('Peak Expiratory Flow Rate')
ax.set_ylabel('Mini Meter (L/min)')
ax.set_xlabel('Large Meter (L/min)')


# Code Block 5
import matplotlib.pyplot as plt

ax = plt.axes()
ax.scatter(df['Wright Large'], df['Wright Mini'])
ax.set_title('Peak Expiratory Flow Rate')
ax.set_ylabel('Mini Meter (L/min)')
ax.set_xlabel('Large Meter (L/min)')


# Code Block 6
plt.savefig("plots/bland_altman_extracted_plot_6.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 7
# Get the axis limits
left, right = ax.get_xlim()
# Plot the line of equality
ax.plot([0, right], [0, right])
# Set the axis limits
ax.set_ylim(0, right)
ax.set_xlim(0, right)

plt.savefig("plots/bland_altman_extracted_plot_7.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 8
# Get the axis limits
left, right = ax.get_xlim()
# Plot the line of equality
ax.plot([0, right], [0, right])
# Set the axis limits
ax.set_ylim(0, right)
ax.set_xlim(0, right)

plt.savefig("plots/bland_altman_extracted_plot_8.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 9
ax = plt.axes()
ax.scatter(df['Wright Large'], df['Wright Mini'], c='k', s=20, alpha=0.6, marker='o')
ax.set_title('Peak Expiratory Flow Rate')
ax.set_ylabel('Mini Meter (L/min)')
ax.set_xlabel('Large Meter (L/min)')
# Get axis limits
left, right = ax.get_xlim()
# Reference line
ax.plot([0, right], [0, right], c='grey', ls='--', label='Line of Equality')
# Set axis limits
ax.set_ylim(0, right)
ax.set_xlim(0, right)
# Set aspect ratio
ax.set_aspect('equal')
# Legend
ax.legend(frameon=False)

plt.savefig("plots/bland_altman_extracted_plot_9.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 10
ax = plt.axes()
ax.scatter(df['Wright Large'], df['Wright Mini'], c='k', s=20, alpha=0.6, marker='o')
ax.set_title('Peak Expiratory Flow Rate')
ax.set_ylabel('Mini Meter (L/min)')
ax.set_xlabel('Large Meter (L/min)')
# Get axis limits
left, right = ax.get_xlim()
# Reference line
ax.plot([0, right], [0, right], c='grey', ls='--', label='Line of Equality')
# Set axis limits
ax.set_ylim(0, right)
ax.set_xlim(0, right)
# Set aspect ratio
ax.set_aspect('equal')
# Legend
ax.legend(frameon=False)

plt.savefig("plots/bland_altman_extracted_plot_10.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 11
import pandas as pd
import numpy as np

means = df.mean(axis=1)
diffs = df.diff(axis=1).iloc[:, -1]
# Average difference (aka the bias)
bias = np.mean(diffs)
# Sample standard deviation
sd = np.std(diffs, ddof=1)
# Limits of agreement
upper_loa = bias + 2 * sd
lower_loa = bias - 2 * sd


# Code Block 12
import pandas as pd
import numpy as np

means = df.mean(axis=1)
diffs = df.diff(axis=1).iloc[:, -1]
# Average difference (aka the bias)
bias = np.mean(diffs)
# Sample standard deviation
sd = np.std(diffs, ddof=1)
# Limits of agreement
upper_loa = bias + 2 * sd
lower_loa = bias - 2 * sd


# Code Block 13
ax = plt.axes()
ax.scatter(means, diffs, c='k', s=20, alpha=0.6, marker='o')
ax.set_title('Bland-Altman Plot for Two Methods of Measuring PEFR')
ax.set_ylabel('Difference (L/min)')
ax.set_xlabel('Mean (L/min)')
# Get axis limits
left, right = plt.xlim()
bottom, top = plt.ylim()
# Set y-axis limits
max_y = max(abs(bottom), abs(top))
ax.set_ylim(-max_y * 1.1, max_y * 1.1)
# Set x-axis limits
domain = right - left
ax.set_xlim(left, left + domain * 1.1)

plt.savefig("plots/bland_altman_extracted_plot_13.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 14
ax = plt.axes()
ax.scatter(means, diffs, c='k', s=20, alpha=0.6, marker='o')
ax.set_title('Bland-Altman Plot for Two Methods of Measuring PEFR')
ax.set_ylabel('Difference (L/min)')
ax.set_xlabel('Mean (L/min)')
# Get axis limits
left, right = plt.xlim()
bottom, top = plt.ylim()
# Set y-axis limits
max_y = max(abs(bottom), abs(top))
ax.set_ylim(-max_y * 1.1, max_y * 1.1)
# Set x-axis limits
domain = right - left
ax.set_xlim(left, left + domain * 1.1)

plt.savefig("plots/bland_altman_extracted_plot_14.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 15
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=upper_loa, c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=lower_loa, c='grey', ls='--')

plt.savefig("plots/bland_altman_extracted_plot_15.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 16
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=upper_loa, c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=lower_loa, c='grey', ls='--')

plt.savefig("plots/bland_altman_extracted_plot_16.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 17
ax.annotate('+2×SD', (right, upper_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (-10, -27), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{bias:+4.2f}', (right, bias), (-10, -27), textcoords='offset pixels')
ax.annotate('-2×SD', (right, lower_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (-10, -27), textcoords='offset pixels')

plt.savefig("plots/bland_altman_extracted_plot_17.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 18
ax.annotate('+2×SD', (right, upper_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (-10, -27), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{bias:+4.2f}', (right, bias), (-10, -27), textcoords='offset pixels')
ax.annotate('-2×SD', (right, lower_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (-10, -27), textcoords='offset pixels')

plt.savefig("plots/bland_altman_extracted_plot_18.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 19
import scipy.stats as stats

# Sample size
n = df.shape[0]
# Variance
var = sd**2
# Standard error of the bias
se_bias = np.sqrt(var / n)
# Standard error of the limits of agreement
se_loas = np.sqrt(3 * var / n)
# Endpoints of the range that contains 95% of the Student’s t distribution
t_interval = stats.t.interval(alpha=0.95, df=n - 1)
# Confidence intervals
ci_bias = bias + np.array(t_interval) * se_bias
ci_upperloa = upper_loa + np.array(t_interval) * se_loas
ci_lowerloa = lower_loa + np.array(t_interval) * se_loas


# Code Block 20
import scipy.stats as stats

# Sample size
n = df.shape[0]
# Variance
var = sd**2
# Standard error of the bias
se_bias = np.sqrt(var / n)
# Standard error of the limits of agreement
se_loas = np.sqrt(3 * var / n)
# Endpoints of the range that contains 95% of the Student’s t distribution
t_interval = stats.t.interval(alpha=0.95, df=n - 1)
# Confidence intervals
ci_bias = bias + np.array(t_interval) * se_bias
ci_upperloa = upper_loa + np.array(t_interval) * se_loas
ci_lowerloa = lower_loa + np.array(t_interval) * se_loas


# Code Block 21
# Plot the confidence intervals
ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_bias), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--', alpha=0.5)
# Plot the confidence intervals' caps
x_range = [left - domain * 0.025, left + domain * 0.025]
ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--', alpha=0.5)
# Adjust the x- and y-axis limits
max_y = max(abs(ci_upperloa[1]), abs(ci_lowerloa[0]))
ax.set_ylim(-max_y * 1.05, max_y * 1.05)
ax.set_xlim(left - domain * 0.05, left + domain * 1.1)

plt.savefig("plots/bland_altman_extracted_plot_21.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 22
# Plot the confidence intervals
ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_bias), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--', alpha=0.5)
# Plot the confidence intervals' caps
x_range = [left - domain * 0.025, left + domain * 0.025]
ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--', alpha=0.5)
# Adjust the x- and y-axis limits
max_y = max(abs(ci_upperloa[1]), abs(ci_lowerloa[0]))
ax.set_ylim(-max_y * 1.05, max_y * 1.05)
ax.set_xlim(left - domain * 0.05, left + domain * 1.1)

plt.savefig("plots/bland_altman_extracted_plot_22.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 23
# Make figures A5 in size
A = 5
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
# Image quality
plt.rc('figure', dpi=141)
# Be able to add Latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{textgreek}')

#
# Plot
#
ax = plt.axes()
ax.scatter(means, diffs, c='k', s=20, alpha=0.6, marker='o')
ax.set_title('Bland-Altman Plot for Two Methods of Measuring PEFR')
ax.set_ylabel(r'Difference, $d$ (L/min)')
ax.set_xlabel(r'Mean, $\mu$ (L/min)')
# Set x-axis limits
left, right = plt.xlim()
domain = right - left
ax.set_xlim(left - domain * 0.05, left + domain * 1.1)
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=upper_loa, c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=lower_loa, c='grey', ls='--')
# Add the annotations
ax.annotate('+2×SD', (right, upper_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (-10, -27), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{bias:+4.2f}', (right, bias), (-10, -27), textcoords='offset pixels')
ax.annotate('-2×SD', (right, lower_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (-10, -27), textcoords='offset pixels')
# Plot the confidence intervals
ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_bias), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--', alpha=0.5)
# Plot the confidence intervals' caps
x_range = [left - domain * 0.025, left + domain * 0.025]
ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--', alpha=0.5)
# Set y-axis limits
max_y = max(abs(ci_upperloa[1]), abs(ci_lowerloa[0]))
ax.set_ylim(-max_y * 1.05, max_y * 1.05)

plt.savefig("plots/bland_altman_extracted_plot_23.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 24
# Make figures A5 in size
A = 5
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
# Image quality
plt.rc('figure', dpi=141)
# Be able to add Latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{textgreek}')

#
# Plot
#
ax = plt.axes()
ax.scatter(means, diffs, c='k', s=20, alpha=0.6, marker='o')
ax.set_title('Bland-Altman Plot for Two Methods of Measuring PEFR')
ax.set_ylabel(r'Difference, $d$ (L/min)')
ax.set_xlabel(r'Mean, $\mu$ (L/min)')
# Set x-axis limits
left, right = plt.xlim()
domain = right - left
ax.set_xlim(left - domain * 0.05, left + domain * 1.1)
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=upper_loa, c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=lower_loa, c='grey', ls='--')
# Add the annotations
ax.annotate('+2×SD', (right, upper_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (-10, -27), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{bias:+4.2f}', (right, bias), (-10, -27), textcoords='offset pixels')
ax.annotate('-2×SD', (right, lower_loa), (-10, 10), textcoords='offset pixels')
ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (-10, -27), textcoords='offset pixels')
# Plot the confidence intervals
ax.plot([left] * 2, list(ci_upperloa), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_bias), c='grey', ls='--', alpha=0.5)
ax.plot([left] * 2, list(ci_lowerloa), c='grey', ls='--', alpha=0.5)
# Plot the confidence intervals' caps
x_range = [left - domain * 0.025, left + domain * 0.025]
ax.plot(x_range, [ci_upperloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_upperloa[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[1]] * 2, c='grey', ls='--', alpha=0.5)
ax.plot(x_range, [ci_lowerloa[0]] * 2, c='grey', ls='--', alpha=0.5)
# Set y-axis limits
max_y = max(abs(ci_upperloa[1]), abs(ci_lowerloa[0]))
ax.set_ylim(-max_y * 1.05, max_y * 1.05)

plt.savefig("plots/bland_altman_extracted_plot_24.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 25
plt.savefig(<filename>)


# Code Block 26
plt.figure()


# Code Block 27
plt.close()


import glob
print("\nPlots generated:")
plot_files = glob.glob('plots/*.png')
for i, plot_file in enumerate(sorted(plot_files)):
    print(f"{i+1}. {plot_file}")
