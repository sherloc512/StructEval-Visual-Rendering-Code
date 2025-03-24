# Converted from: Heat%20Maps%20using%20Matplotlib%20and%20Seaborn.ipynb

# Markdown Cell 1
# <h1 align="center"> Heat Maps using Matplotlib and Seaborn</h1>

# Markdown Cell 2
# youtube tutorial: https://www.youtube.com/watch?v=m7uXFyPN2Sk

# Cell 3
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cell 4
helix = pd.read_csv('Data/helix_parameters.csv')
helix.head() # just seeing that data was imported properly by outputing first 5 cells

# Cell 5
# shape of the dataframe
helix.shape

# Cell 6
# checking what the columns are
helix.columns

# Markdown Cell 7
# <h3 align='Left'>Selecting Columns (by different methods)</h3>

# Cell 8
# selecting a couple columns
couple_columns = helix[['Energy','helix 2 phase', 'helix1 phase']]
couple_columns.head()

# Cell 9
# selecting same columns a different way
helix.ix[:,['Energy','helix 2 phase', 'helix1 phase']].head()

# Cell 10
# this is essentially would be taking the average of each unique combination. 
# one important mention is notice how little the data varies from eachother.
phase_1_2 = couple_columns.groupby(['helix1 phase', 'helix 2 phase']).mean()
print phase_1_2.shape
phase_1_2.head(10)

# Cell 11
phase_1_2 = phase_1_2.reset_index()
phase_1_2.head()

# Markdown Cell 12
# <h3 align='Left'>Heat Map using Matplotlib</h3>

# Cell 13
major_ticks = np.arange(0, 200, 20)                                              
minor_ticks = np.arange(0, 180, 5)  

fig = plt.figure(figsize = (6,5))  
ax = fig.add_subplot(1,1,1) 
s = ax.scatter('helix1 phase', 'helix 2 phase', c = 'Energy',data = phase_1_2, cmap = 'Blues_r', marker = 's',s = 190)
ax.axis([phase_1_2['helix1 phase'].min()-10, phase_1_2['helix1 phase'].max()+10, phase_1_2['helix 2 phase'].min()-10, phase_1_2['helix 2 phase'].max()+10])
ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)                                           
ax.set_yticks(major_ticks)                                                                                                                                                      
ax.grid(which='both', alpha = 0.3)                                                                                                           
ax.grid(which='major', alpha=0.3) 
ax.set_xlabel('helix1 phase', fontsize=10);
ax.set_ylabel('helix 2 phase', fontsize=10);
ax.set_title('Energy from Helix Phase Angles', size = 15)

# http://stackoverflow.com/questions/13943217/how-to-add-colorbars-to-scatterplots-created-like-this
cbar = plt.colorbar(mappable = s,ax = ax)

plt.show()

# Markdown Cell 14
# <h3 align='Left'>Heat Map using Seaborn</h3>

# Cell 15
import numpy as np;
import seaborn as sns; 

# To translate into Excel Terms for those familiar with Excel
# string 1 is row labels 'helix1 phase'
# string 2 is column labels 'helix 2 phase'
# string 3 is values 'Energy'
# Official pivot documentation
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pivot.html

phase_1_2.pivot('helix1 phase', 'helix 2 phase','Energy').head()

# Cell 16
# To translate into Excel Terms for those familiar with Excel
# string 1 is row labels 'helix1 phase'
# string 2 is column labels 'helix 2 phase'
# ['Energy'] is values
phase_1_2.pivot('helix1 phase', 'helix 2 phase')['Energy'].head()

# Cell 17
# seaborn heatmap documentation
# https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html

# cmap choices: http://matplotlib.org/users/colormaps.html
plt.figure(figsize=(9,9))
pivot_table = phase_1_2.pivot('helix1 phase', 'helix 2 phase','Energy')
plt.xlabel('helix 2 phase', size = 15)
plt.ylabel('helix1 phase', size = 15)
plt.title('Energy from Helix Phase Angles', size = 15)
sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');

# Markdown Cell 18
# <h1 align="center"> Subplotting and 3D Heatmaps using Matplotlib and Seaborn </h1>

# Markdown Cell 19
# youtube tutorial: https://www.youtube.com/watch?v=NHwXkvwSd7E

# Cell 20
initial_data = pd.read_csv('Data/helix_parameters.csv')
initial_data.head() # just seeing that data was imported properly by outputing first 5 cells

# Cell 21
# checking what the columns are
initial_data.columns

# Markdown Cell 22
# <h3 align='Left'>Selecting Columns (by different methods)</h3>

# Cell 23
# selecting a couple columns
data = initial_data[['helix1 phase', 'helix 2 phase', 'helix3 phase', 'Energy']]
data2 = data
data.head()

# Markdown Cell 24
# <h3 align='Left'>Taking Average of Unique Combination</h3>

# Cell 25
# this is essentially would be taking the average of each unique combination. 
# one important mention is notice how little the data varies from eachother.
data = data.groupby(['helix1 phase', 'helix 2 phase', 'helix3 phase']).mean()
low = data2.groupby(['helix1 phase', 'helix 2 phase', 'helix3 phase']).min()
print phase_1_2.shape
data.head(10)

# Cell 26
data = data.reset_index()
data2 = data2.reset_index()
data.head()

# Cell 27
data2.head()

# Markdown Cell 28
# <h3 align='Left'>3D Matplotlib Heatmap</h3>

# Cell 29
# Main problem here is nnot very interactive or informative. Very Pretty though! 
# other problem is colorbar takes up space normally used for figure. Also not interactive. 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

#colors = cm.hsv(the_fourth_dimension/max(the_fourth_dimension))

colmap = cm.ScalarMappable(cmap=cm.Greens_r)
colmap.set_array(data[['Energy']])

# reference for cmap. note cmap and c are different!
# http://matplotlib.org/examples/color/colormaps_reference.html
ax.scatter(data[['helix1 phase']], data[['helix 2 phase']], data[['helix3 phase']], marker='s',s = 140, c=data[['Energy']], cmap='Greens_r');
cb = fig.colorbar(colmap)

ax.set_xlabel('helix1 phase');
ax.set_ylabel('helix 2 phase');
ax.set_zlabel('helix3 phase');
plt.show()
# change view angle 
# http://infamousheelfilcher.blogspot.com/2013/02/changing-viewing-angle-of-matplotlib.html
#ax.view_init(azim = 0,elev = 0)

# Markdown Cell 30
# <h3 align='Left'>Taking Average of Unique Combination</h3>

# Cell 31
# It would be a lot of redundant code to make the 
# labelsize bigger on all the subplots for each future graph
# this just graphs pylabs defaults temporarily

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': '17',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# Cell 32
# https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html
# for more seaborn stuff

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (25,25));

filt_data = initial_data[['helix1 phase', 'helix 2 phase', 'Energy']];
filt_data = filt_data.groupby(['helix1 phase', 'helix 2 phase']).mean();
filt_data = filt_data.reset_index();
pivot_0_0 = filt_data.pivot('helix1 phase', 'helix 2 phase', 'Energy');
sns.heatmap(pivot_0_0, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[0], cbar = False);

filt_data = initial_data[['helix1 phase', 'helix3 phase', 'Energy']];
filt_data = filt_data.groupby(['helix1 phase', 'helix3 phase']).mean();
filt_data = filt_data.reset_index();
pivot_0_1 = filt_data.pivot('helix1 phase', 'helix3 phase', 'Energy');
sns.heatmap(pivot_0_1, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[1], cbar = False);

filt_data = initial_data[['helix 2 phase', 'helix3 phase', 'Energy']];
filt_data = filt_data.groupby(['helix 2 phase', 'helix3 phase']).mean();
filt_data = filt_data.reset_index();
pivot_0_2 = filt_data.pivot('helix 2 phase', 'helix3 phase', 'Energy');
sns.heatmap(pivot_0_2, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[2], cbar = False);

# Markdown Cell 33
# <h3 align='Left'>Min of Each Combination</h3>

# Cell 34
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (25,25));

filt_data = initial_data[['helix1 phase', 'helix 2 phase', 'Energy']];
filt_data = filt_data.groupby(['helix1 phase', 'helix 2 phase']).min();
filt_data = filt_data.reset_index();
pivot_0_0 = filt_data.pivot('helix1 phase', 'helix 2 phase', 'Energy');
sns.heatmap(pivot_0_0, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[0], cbar = False);

filt_data = initial_data[['helix1 phase', 'helix3 phase', 'Energy']];
filt_data = filt_data.groupby(['helix1 phase', 'helix3 phase']).min();
filt_data = filt_data.reset_index();
pivot_0_1 = filt_data.pivot('helix1 phase', 'helix3 phase', 'Energy');
sns.heatmap(pivot_0_1, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[1], cbar = False);

filt_data = initial_data[['helix 2 phase', 'helix3 phase', 'Energy']];
filt_data = filt_data.groupby(['helix 2 phase', 'helix3 phase']).min();
filt_data = filt_data.reset_index();
pivot_0_2 = filt_data.pivot('helix 2 phase', 'helix3 phase', 'Energy');
sns.heatmap(pivot_0_2, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[2], cbar = False);

# Markdown Cell 35
# <h3 align='Left'>Taking Max of Unique Combination</h3>

# Cell 36
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (25,25));

filt_data = initial_data[['helix1 phase', 'helix 2 phase', 'Energy']];
filt_data = filt_data.groupby(['helix1 phase', 'helix 2 phase']).max();
filt_data = filt_data.reset_index();
pivot_0_0 = filt_data.pivot('helix1 phase', 'helix 2 phase', 'Energy');
sns.heatmap(pivot_0_0, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[0], cbar = False);

filt_data = initial_data[['helix1 phase', 'helix3 phase', 'Energy']];
filt_data = filt_data.groupby(['helix1 phase', 'helix3 phase']).max();
filt_data = filt_data.reset_index();
pivot_0_1 = filt_data.pivot('helix1 phase', 'helix3 phase', 'Energy');
sns.heatmap(pivot_0_1, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[1], cbar = False);

filt_data = initial_data[['helix 2 phase', 'helix3 phase', 'Energy']];
filt_data = filt_data.groupby(['helix 2 phase', 'helix3 phase']).max();
filt_data = filt_data.reset_index();
pivot_0_2 = filt_data.pivot('helix 2 phase', 'helix3 phase', 'Energy');
sns.heatmap(pivot_0_2, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r', ax = axes[2], cbar = False);

