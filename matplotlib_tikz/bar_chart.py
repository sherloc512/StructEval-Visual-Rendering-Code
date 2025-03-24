# Converted from: bar_chart.ipynb

# Markdown Cell 1
# # Data Viz with matplotlib Series 1: Bar chart
# 
# ## Reference
# 
# - Bar chart
#     <https://en.wikipedia.org/wiki/Bar_chart>
# - What to consider when creating stacked column charts
#     <https://blog.datawrapper.de/stacked-column-charts>
# - Horizontal bar chart
#     <https://datavizproject.com/data-type/bar-chart-horizontal/>

# Cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Markdown Cell 3
# ## Bar chart
# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. The bars can be plotted vertically or horizontally. 

# Markdown Cell 4
# ### When to use it ?

# Markdown Cell 5
# - Compare **categorical data**.
# - Comparisons among **discrete categories**.
# - One axis of the chart shows the specific categories being compared, and the other axis represents a measured value.

# Markdown Cell 6
# ### Example

# Cell 7
plt.figure(figsize=(9, 6))

x = np.arange(4)
turnover_k_euros = [12, 34, 31, 20]

plt.bar(x, turnover_k_euros, width=0.4)
plt.xticks(np.arange(4), ('apple', 'banana', 'cherry', 'durian'))
plt.xlabel('Product')
plt.ylabel('Turnover (k euros)')

plt.show()

# Markdown Cell 8
# This plot describes turnovers(k euros) for each fruit. Among four fruits, bananas' sales bring the largest turnover (34k euros), however, it seems that consumers don't like apple that much.

# Markdown Cell 9
# ## Grouped bar chart
# Bar graphs can also be used for more complex comparisons of data with grouped bar charts and stacked bar charts. In a grouped bar chart, for each categorical group there are two or more bars. These bars are color-coded to represent a particular grouping.

# Markdown Cell 10
# ### When to use it ?

# Markdown Cell 11
# To represent and compare **different categories of two or more groups**.

# Markdown Cell 12
# ### Example

# Cell 13
year_n_1 = (20, 25, 27, 35, 27, 27, 33)
year_n = (25, 32, 35, 40, 33, 29, 36)

ind = np.arange(7)
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(ind - width / 2, year_n_1, width, color='#1f77b4', alpha=0.5)
rects2 = ax.bar(ind + width / 2, year_n, width, color='#1f77b4')

plt.xticks(np.arange(7), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'))
plt.xlabel('Month')
plt.ylabel('Turnover (k euros)')
plt.legend((rects1[0], rects2[0]), ('year N-1', 'year N'))

plt.show()

# Markdown Cell 14
# This plot compares monthly turnover of year N to year N-1. Except for April and May, monthly turnover in year N is higher than year N-1. In the case of retailing, this kind of changes can be explained like the strategy of year N works well, or new products attract clients, or new stores of year N contribute to the turnover.

# Markdown Cell 15
# ## Stacked bar chart
# Alternatively, a stacked bar chart could be used. The stacked bar chart stacks bars that represent different groups on top of each other. The height of the resulting bar shows the combined result of the groups.

# Markdown Cell 16
# ### When to use it ?

# Markdown Cell 17
# - To compare the **totals** and **one part of the totals**.
# - If the total of your parts is crucial, stacked column chart can work well for dates.

# Markdown Cell 18
# ### Example

# Cell 19
plt.figure(figsize=(9, 6))

cheese = (20, 25, 22, 25, 27, 30, 10)
non_cheese = (10, 18, 15, 16, 18, 17, 9)

rect1 = plt.bar(np.arange(7), cheese, width=0.5, color='orangered', alpha=0.9)
rect2 = plt.bar(np.arange(7), non_cheese, bottom=cheese, width=0.5, color='#1f77b4', alpha=0.9)

plt.xticks(np.arange(7), ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'))
plt.xlabel('Weekday')
plt.ylabel('Turnover (k euros)')
plt.legend((rect1[0], rect2[0]), ('cheese', 'non-cheese'))

plt.show()

# Markdown Cell 20
# This plot presents weekdays' turnover with cheese and non-cheese products' sales. Globally, the sales of cheese products are much more than others.

# Markdown Cell 21
# ## Horizontal bar chart

# Markdown Cell 22
# The horizontal bar chart is the same as a vertical bar chart only the x-axis and y-axis are switched.

# Markdown Cell 23
# ### When to use it ?

# Markdown Cell 24
# - You need more room to fit text labels for categorical variables.
# - When you work with a big dataset, horizontal bar charts tend to work better in a narrow layout such as mobile view.

# Markdown Cell 25
# ### Example

# Cell 26
df = pd.DataFrame({'product': ['grill', 'cheese', 'red wine', 'salade', 'chicken', 'sushi', 'pizza', 'soup'],
                   'turnover': [846, 739, 639, 593, 572, 493, 428, 293]},
                  columns=['product', 'turnover'])
df.sort_values('turnover', inplace=True)
df.reset_index(inplace=True, drop=True)
df

# Cell 27
plt.figure(figsize=(9, 6))

plt.barh(np.arange(len(df['product'])), df['turnover'], align='center')
plt.yticks(np.arange(len(df['product'])), df['product'])
plt.tick_params(labelsize=12)
plt.xlabel('Turnover(k euros)', fontdict={'fontsize': 13})
plt.ylabel('Product', fontdict={'fontsize': 13})

plt.show()

# Markdown Cell 28
# This vertical bar chart describes clearly turnover for each product. Obviously, grill product is prefered by clients.

