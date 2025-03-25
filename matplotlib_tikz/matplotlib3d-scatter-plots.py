# Converted from: matplotlib3d-scatter-plots.ipynb

# Markdown Cell 1
# # 3D scatter plot and related plots
# 
# **'related' is referencing 2D plots where a sense of the degree of a third variable is implied by the size of the point on the 2D scatter plot**
# 
# If this notebook is not in active (runnable) form, go to [here](https://github.com/fomightez/3Dscatter_plot-binder) and press `launch binder`.
# 
# (This notebook also works in sessions launched from [here](https://github.com/fomightez/Python_basics_4nanocourse).)
# 
# ------
# 
# <div class="alert alert-block alert-warning">
# <p>If you haven't used one of these notebooks before, they're basically web pages in which you can write, edit, and run live code. They're meant to encourage experimentation, so don't feel nervous. Just try running a few cells and see what happens!.</p>
# 
# <p>
#     Some tips:
#     <ul>
#         <li>Code cells have boxes around them.</li>
#         <li>To run a code cell, click on the cell and either click the <i class="fa-play fa"></i> button on the toolbar above, or then hit <b>Shift+Enter</b>. The <b>Shift+Enter</b> combo will also move you to the next cell, so it's a quick way to work through the notebook. Selecting from the menu above the toolbar, <b>Cell</b> > <b>Run All</b> is a shortcut to trigger attempting to run all the cells in the notebook.</li>
#         <li>While a cell is running a <b>*</b> appears in the square brackets next to the cell. Once the cell has finished running the asterisk will be replaced with a number.</li>
#         <li>In most cases you'll want to start from the top of notebook and work your way down running each cell in turn. Later cells might depend on the results of earlier ones.</li>
#         <li>To edit a code cell, just click on it and type stuff. Remember to run the cell once you've finished editing.</li>
#     </ul>
# </p>
# </div>
# 
# 
# 
# 
# ----
# 
# ## 3D scatter plot 
# 
# ### Matplotlib-based
# 
# Based on [3D Scatterplot page](https://python-graph-gallery.com/370-3d-scatterplot/) from Yan Holtz's [Python Graph Gallery](https://python-graph-gallery.com/).

# Cell 2
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Dataset
df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })
 
# plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)
ax.view_init(50, 185)
plt.show()

# Markdown Cell 3
# Using `%matplotlib notebook` in the classic notebook results in an rotatable, 3D view. (See below about the difference needed for the modern JupyterLab and Jupyter Notebook 7+ tech.) Once you have a good view you can stop the interactivity by pressing on the blue 'stop' button in the upper right and then in the next cell run the following code to get the values to put in `ax.view_init()` to reproduce that view point later:
# 
# ```python
# print('ax.azim {}'.format(ax.azim))
# print('ax.elev {}'.format(ax.elev))
# ```
# Change `%matplotlib notebook` back to `%matplotlib inline` for static view. 
# 
# In the case of the modern JupyterLab and Jupyter Notebook 7+ tech, a slightly different invocation is needed and there is no blue 'stop' button. (Not sure what flavor of Jupyter tech you are using for your notebooks, then see ['Quickly Navigating the tech of the Jupyter ecosystem post-2023'](https://gist.github.com/fomightez/e873947b502f70388d82644b17196279).)    
# For modern JupyterLab and Jupyter Notebook 7+ tech:  
# In place of `%matplotlib notebook`, you install [ipympl](https://matplotlib.org/ipympl/) and use `%matplotlib ipympl`.  
# (Otherwise, if trying `%matplotlib notebook` in JupyterLab or Jupyter Notebook 7+, you'll see errors about `IPython not defined` when trying to run the cells on this page in JupyterLab.) In this session served by MyBinder, it currently defaults to opening in the classic notebook and so `ipympl` is not installed here. (If you go [here](https://github.com/fomightez/animated_matplotlib-binder), you can launch a session that has the modern Jupyter tech and `ipympl` has been installed there. You can try the code from here, in that session using `%matplotlib ipympl` for the interactive interface.)   
# In the case of using that interface and you want the values to put in `ax.view_init()`, you rotate it to where you want and make a new cell and run the `print()` lines spelled out above.
# 
# The same holds for other plots below on this page.
# 
# Another example of a Matplotlib Scatter plot, this one with lots of the features labeled.

# Cell 4
#based on correction at https://stackoverflow.com/a/78319637/8508004 & adjusting `ax.set_box_aspect()` to not cut-off the Z-axis label
%matplotlib inline
  
# importing required libraries
import matplotlib.pyplot as plt
  
# creating random dataset
xs = [14, 24, 43, 47, 54, 66, 74, 89, 12,
      44, 1, 2, 3, 4, 5, 9, 8, 7, 6, 5]
  
ys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 3,
      5, 2, 4, 1, 8, 7, 0, 5]
  
zs = [9, 6, 3, 5, 2, 4, 1, 8, 7, 0, 1, 2, 
      3, 4, 5, 6, 7, 8, 9, 0]
  
# creating figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
  
# creating the plot
plot_geeks = ax.scatter(xs, ys, zs, color='green')
  
# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
      
# displaying the plot
#ax.dist = 13 # See https://stackoverflow.com/a/77580433/8508004 about this line and the next; note the next line worked with the current example in the documentation at https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html with that example otherwise cutting off the Z-label
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Markdown Cell 5
# #### A more thorough Matplotlib example
# 
# Next example, based on [Part #3](https://jovianlin.io/data-visualization-seaborn-part-3/) of Jovian Lin's 3-part series [Data Visualization with Seaborn](https://jovianlin.io/data-visualization-seaborn-part-1/); however that section acknowledges the solution is based on Matplotlib:

# Cell 6
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
red_wine   = pd.read_csv('winequality-red.csv',   sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
wines = pd.concat([red_wine,white_wine], ignore_index=True)
print("red wines:",len(red_wine))
print("white wines:",len(white_wine))
print("wines:",len(wines))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = wines['residual sugar']
ys = wines['fixed acidity']
zs = wines['alcohol']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Fixed Acidity')
ax.set_zlabel('Alcohol')

# Fix so zlabel not cut off
#ax.dist = 13 # See https://stackoverflow.com/a/77580433/8508004 about this line and the next; note the next line worked with the current example in the documentation at https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html with that example otherwise cutting off the Z-label
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Markdown Cell 7
# The earlier parts of the code in the cell above were built-based on the earlier parts of that series.

# Markdown Cell 8
# #### 3D, maptlotlib-based examples with a legend
# 
# Based on https://stackoverflow.com/a/60621783/8508004 and then updated with [the current documentation]https://matplotlib.org/stable/users/explain/toolkits/mplot3d.html) pointing out two things:
# 
# - "3D Axes (of class `Axes3D`) are created by passing the `projection="3d"` keyword argument to Figure.add_subplot", with the code below there using `ax = fig.add_subplot(projection='3d')`.
# 
# - "Changed in version 3.2.0: Prior to Matplotlib 3.2.0, it was necessary to explicitly import the `mpl_toolkits.mplot3d` module to make the '3d' projection to `Figure.add_subplot`."

# Cell 9
%matplotlib inline
import seaborn as sns, numpy as np, pandas as pd, random
import matplotlib.pyplot as plt
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(projection='3d')

x = np.random.uniform(1,20,size=20)
y = np.random.uniform(1,100,size=20)
z = np.random.uniform(1,100,size=20)


g = ax.scatter(x, y, z, c=x, marker='o', depthshade=False, cmap='Paired')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# produce a legend with the unique colors from the scatter
legend = ax.legend(*g.legend_elements(), loc="lower center", title="X Values", borderaxespad=-10, ncol=4) # Used to work. The stuff with the zlabel and size seems to have changed so not good 
legend = ax.legend(*g.legend_elements(), loc="lower center", bbox_to_anchor=(0.12, 0., 1.0, 0.5), borderaxespad=+24.5, title="X Values", ncol=4) # works as far as displaying, but why not use 'upper' to be simpler?
legend = ax.legend(*g.legend_elements(), loc="upper center", title="X Values", ncol=4)
ax.add_artist(legend)

#ax.dist = 13 # See https://stackoverflow.com/a/77580433/8508004 about this line and the next; note the next line worked with the current example in the documentation at https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html with that example otherwise cutting off the Z-label
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Markdown Cell 10
# If you want to see the possibilities for `cmap` enter some nonsensical text as `cmap`, and it will list the possibilities. `viridis` is one

# Cell 11
%matplotlib inline
import seaborn as sns, numpy as np, pandas as pd, random
import matplotlib.pyplot as plt
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(projection='3d')

x = np.random.uniform(1,20,size=20)
y = np.random.uniform(1,100,size=20)
z = np.random.uniform(1,100,size=20)


g = ax.scatter(x, y, z, c=x, marker='o', depthshade=False, cmap='viridis')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# produce a legend with the unique colors from the scatter
legend = ax.legend(*g.legend_elements(), loc="upper center", title="X Values", ncol=4)
ax.add_artist(legend)

#ax.dist = 13 # See https://stackoverflow.com/a/77580433/8508004 about this line and the next; note the next line worked with the current example in the documentation at https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html with that example otherwise cutting off the Z-label
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Markdown Cell 12
# That last `cmap` is a gradient and if you'd prefer the legend be a color bar showing that gradient, you'd adjust the last code cell to read like this:

# Cell 13
%matplotlib inline
import seaborn as sns, numpy as np, pandas as pd, random
import matplotlib.pyplot as plt
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(9.5,7))

ax = fig.add_subplot(projection='3d')

x = np.random.uniform(1,20,size=20)
y = np.random.uniform(1,100,size=20)
z = np.random.uniform(1,100,size=20)


g = ax.scatter(x, y, z, c=x, marker='o', depthshade=False, cmap='viridis')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# produce a legend with a gradient colorbar on the right, based on https://stackoverflow.com/a/5495912/8508004
clb = fig.colorbar(g)
clb.ax.set_title('X') #adding title based on https://stackoverflow.com/a/33740567/8508004

plt.show()

# Markdown Cell 14
# Note the width is also adjusted up (`figsize=(8,8)` to `figsize=(9.5,7)`) to accomodate the additonal colorbar legend on the right side of the resulting plot figure without it overlapping the Z axis label.
# 
# 
# Additional example with a legend, based on [here](https://stackoverflow.com/a/69887341/8508004 ):

# Cell 15
#based on https://stackoverflow.com/a/69887341/8508004 & other adaptations already covered in this notebook
import random 
import pandas as pd
from matplotlib import pyplot as plt

random.seed(0)
D = [[random.random() for x in range(3)] for y in range(1000)]
df = pd.DataFrame(D,columns=['Feature_1','Feature_2','Feature_3'])
predictions = [random.randint(0,4) for x in range(1000)]
df['predictions'] = predictions

plt.rcParams["figure.figsize"]=(10,10)
plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scatter = ax.scatter(df['Feature_1'], df['Feature_2'], df['Feature_3'],
                     c=df['predictions'], s=150, cmap='rainbow')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
ax.add_artist(legend1)

ax.set_xlabel('Feature_1',fontsize=20,labelpad=10)
ax.set_ylabel('Feature_2', fontsize=20, rotation=150,labelpad=10)
ax.set_zlabel('Feature_3', fontsize=20, rotation=60,labelpad=15)

#ax.dist = 13 # See https://stackoverflow.com/a/77580433/8508004 about this line and the next; note the next line worked with the current example in the documentation at https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html with that example otherwise cutting off the Z-label
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Cell 16
# this resets the settings applied in the last plot above with `plt.rcParams` SO THEY DON'T AFFECT THE PLOTS FOLLOWING THIS POINT, based on https://stackoverflow.com/a/59474458/8508004
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# Markdown Cell 17
# ------
# 
# ## 2D, Seaborn-based approach, better?
# 
# **Use the size parameter to give a sense of a third variable's degree on a 2D plot**
# 
# Based on [Part #3](https://jovianlin.io/data-visualization-seaborn-part-3/) of Jovian Lin's 3-part series [Data Visualization with Seaborn](https://jovianlin.io/data-visualization-seaborn-part-1/):
#     
# >"The better alternative — using Seaborn + toggle the size via the s parameter:"
# 
# The earlier parts of the code below were built-based on the earlier parts of the series.

# Cell 18
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
red_wine   = pd.read_csv('winequality-red.csv',   sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
wines = pd.concat([red_wine,white_wine], ignore_index=True)
print("red wines:",len(red_wine))
print("white wines:",len(white_wine))
print("wines:",len(wines))
plt.scatter(x = wines['fixed acidity'], 
            y = wines['alcohol'], 
            s = wines['residual sugar']*25, # <== 😀 Look here!
            alpha=0.4, 
            edgecolors='w')

plt.xlabel('Fixed Acidity')
plt.ylabel('Alcohol')
plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar', y=1.05);

# Markdown Cell 19
# #### 2D Using Seaborn
# 
# However, the code isn't actually using Seaborn (or at least not current Seaborn code) despite what the source material says. Seems to use Matplpotlib still. I have added use of Seaborn below.

# Cell 20
%matplotlib inline
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
red_wine   = pd.read_csv('winequality-red.csv',   sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
wines = pd.concat([red_wine,white_wine], ignore_index=True)
print("red wines:",len(red_wine))
print("white wines:",len(white_wine))
print("wines:",len(wines))
ax = sns.scatterplot(x=wines['fixed acidity'], y=wines['alcohol'], size=wines['residual sugar'],
                     sizes=(25, 1450), alpha=0.4, legend = False, data=wines)
plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar', y=1.05);

# Markdown Cell 21
# #### 2D, seaborn-based with legend

# Cell 22
%matplotlib inline
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!curl -OL https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
red_wine   = pd.read_csv('winequality-red.csv',   sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
wines = pd.concat([red_wine,white_wine], ignore_index=True)
print("red wines:",len(red_wine))
print("white wines:",len(white_wine))
print("wines:",len(wines))
ax = sns.scatterplot(x=wines['fixed acidity'], y=wines['alcohol'], size=wines['residual sugar'],
                     sizes=(25, 1450), alpha=0.4, data=wines)
#plt.legend(loc='best')
ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1) # based on https://stackoverflow.com/a/53737271/8508004
plt.title('Wine Alcohol Content - Fixed Acidity - Residual Sugar', y=1.05);

# Markdown Cell 23
# (The size scale in this legend doesn't make much sense, but including the legend illustrates having it so that it does overlap the seaborn-based plot.)

# Markdown Cell 24
# ------
# 
# ------
# 
# 
# ### Plotly
# 
# Plotly has all sorts of options for 3D scatter plots. See [this notebook](Plotly3d-scatter-plots.ipynb) for a demo here with links to original source.
# 
# 
# ### Use widgets to adjust options
# 
# See [this notebook](3D_scatter_adjustableVIAwidgets.ipynb) for examples of how `ipywidgets` can be added to allow setting of options from a range of choices or values.

