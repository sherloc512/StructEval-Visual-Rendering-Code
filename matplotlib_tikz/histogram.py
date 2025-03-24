# Converted from: histogram.ipynb

# Markdown Cell 1
# # Data Viz with matplotlib Series 11: Histogram
# 
# ## Reference
# - Histogram
#     <https://en.wikipedia.org/wiki/Histogram>

# Cell 2
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Markdown Cell 3
# ## Histogram

# Markdown Cell 4
# A histogram is an accurate representation of the distribution of numerical data. It differs from a bar graph, in the sense that a bar graph relates two variables, but a histogram relates only one. To construct a histogram, the first step is to "bin" (or "bucket") the range of values-that is, divide the entire range of values into a series of intervals—and then count how many values fall into each interval. The bins are usually specified as consecutive, non-overlapping intervals of a variable. The bins (intervals) must be adjacent, and are often (but are not required to be) of equal size.

# Markdown Cell 5
# ### When to use it ?

# Markdown Cell 6
# - Estimating the probability distribution of a continuous variable (quantitative variable).
# - Organizing large amounts of data, and producing a visualization quickly, using a single dimension.

# Cell 7
tickets = []
for i in range(1, 1931):
    ticket = 'T'+str(i)
    tickets.append(ticket)

# Cell 8
vol = [1] * 140 + [2] * 120 + [3] * 230 + [4] * 200 + [5] * 230 + \
[6] * 180 + [7] * 160 + [8] * 170 + [9] * 130 + [10] * 100 + [11] * 60 + \
[12] * 63 + [13] * 42 + [14] * 31 + [15] * 20 + [16] * 12 + [17] * 9 + [18] * 12 + [19] * 9 + [20] * 12

# Cell 9
df = pd.DataFrame({'TicketID': tickets,
                   'Volumes': vol},
                  columns=['TicketID', 'Volumes'])

# Cell 10
df.head()

# Cell 11
df.describe(percentiles=[0.25, 0.5, 0.75, 0.85])

# Markdown Cell 12
# ### Basic histogram

# Cell 13
plt.figure(figsize=(9, 6))

plt.hist(df['Volumes'], bins=6, density=True)
plt.xlim(left=0, right=21)
plt.xticks(np.arange(21))

plt.grid(alpha=0.2)
plt.show()

# Markdown Cell 14
# This plot describes that among 1930 tickets, 11% tickets contain less than 5 products; less than 1% tickets contain less than 21 products but more than 16 products. However, if we want to the percentage of tickets that contains less than or egale to 10 products, this basic histogram cannot satisfy our need in one second. In the following cumulative histogram, we can find the answer.

# Markdown Cell 15
# ### Cumulative histogram
# A cumulative histogram is a mapping that counts the cumulative number of observations in all of the bins up to the specified bin.

# Cell 16
plt.figure(figsize=(9, 6))

plt.hist(df['Volumes'], bins=6, density=True, cumulative=True, histtype='step', linewidth=2)
plt.xlim(left=0, right=21)
plt.xticks(np.arange(21))

plt.grid(alpha=0.3)
plt.show()

# Markdown Cell 17
# Considering the same question as above: what the percentage of tickets that contain less than or egale to 10 products? According to this cumulative histogram, the answer is obvious: nearly 85% tickets contain less than or egale to 10 products.

