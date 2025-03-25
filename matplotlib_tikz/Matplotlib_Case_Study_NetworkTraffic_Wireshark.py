# Converted from: Matplotlib_Case_Study_NetworkTraffic_Wireshark.ipynb

# Markdown Cell 1
# # Plotting Network Traffic Data With Matplotlib

# Markdown Cell 2
# Matplotlib is a powerful Python plotting library. With this library it is easy to produce;
# - Scatter Plots
# - Line Plots
# - Bar Charts
# - Pie Charts
# - Waffle Charts
# - Word Clouds
# - Histograms
# - Box Plots
# - Heatmaps
# - Subplots
# 
# In this study, we produce __Pie Chart__, __Histogram__ and __Line Chart__ plotting step by step. We prefer to use "network traffic data" for visualization, so we capture our computer's internet traffic data nearly 2 minute with <a href="https://www.wireshark.org/">WireShark</a> and export the as a .csv file. The data can be downloaded from <a href="https://github.com/msklc/Plotting-Network-Traffic-Data-With-Matplotlib/blob/master/network_traffic.csv">GitHub</a>.

# Markdown Cell 3
# ## Preparing Data

# Markdown Cell 4
# First, we import the required libraries and load our network traffic data

# Cell 5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('network_traffic.csv')
df.head()

# Markdown Cell 6
# To see the total column and row numbers

# Cell 7
df.shape

# Markdown Cell 8
# Than, get the summary information of data (row numbers, non-null values, data type of columns)

# Cell 9
df.info()

# Markdown Cell 10
# Convert the 'Time' column type as __datetime__ format

# Cell 11
df['Time']= pd.to_datetime(df['Time']).astype('datetime64[s]') #convert column types as datetime

# Markdown Cell 12
# Create a new column as __Domain__ from the 'Info' column with the help of __regular expression__
# 
# Then filter the data with 'non-null Domain' information

# Cell 13
df['Domain']=df['Info'].str.extract(r'((?:[a-z]*)(?:\.*)(?:[a-z]+)(?:\.)(?:[a-z]+))')
df[df['Domain'].notnull()].head()

# Markdown Cell 14
# Create a new column as __Host__ from the IP adress of 'Destination' column with the help of __socket library__
# 
# We use for loop to get the all the destination IP adresses
# 
# Also use try-except for bypass the errors
# 
# Then filter the data with 'non-Not Found' information

# Cell 15
'''
import socket
ip_host={}
for ip_adress in list(df['Destination'].unique()):
    try:
        ip_host[ip_adress]=socket.gethostbyaddr(ip_adress)[0]
    except:
        ip_host[ip_adress]='Not Found'

df['Host']=df['Destination'].apply(lambda x: ip_host[x])
df[df['Host']!='Not Found'].head()
'''

# Markdown Cell 16
# Finally, we drop the unnecessary columns

# Cell 17
df.drop(['No.','Info'], axis=1, inplace=True)
df.head() # Don't forget, some rows don't have a value in Domain column!!!

# Markdown Cell 18
# ## Understanding Data

# Markdown Cell 19
# The network traffic data size is 1.92MB (in .csv file format). We can check the duration of traffic and the number of records by;

# Cell 20
duration=df['Time'].max()-df['Time'].min()
total_record=df.shape[0]
print('The total records of {} seconds traffic data has {} rows'.format(duration,total_record))

# Markdown Cell 21
# As we know, _Source IP Address_ is the IP address of computer which is used by users. The _Destination IP Address_ is the IP address of server which the user want to reach. 
# 
# So, in network traffic data, it can be expected that _Source IP Address_ should be an unique value. But in real world, network traffic are two-way. So after a request made by users, the server send a reply. So, the _Source IP Address_ of a network traffic data include both the IP addresses of computer and server.
# 
# We can find the real _Source IP Address_ from the 'Info' column which don't have a domain adress. We already capture the domain adresses from 'Info' column with regular expression and save them as a new column 'Domain'.
# 
# We can get real _Source IP Address_ with the help of 'Domain' info as; 

# Cell 22
df[df['Domain'].notnull()]['Source'].value_counts()

# Markdown Cell 23
# __As a Result:__

# Markdown Cell 24
# - The data has 15187 rows (records) in 84 seconds!
# - The data has 2 Source IP Address, what it mean is; it was used neither a virtual machine or the internet connection was corrupted and reconnected again
# - The adress of __fe80::c:5938:664a:f8af__ is a IPv6 gateway IP address. So, it is understood that the user configured the IP v6.

# Markdown Cell 25
# ## Visualization

# Markdown Cell 26
# Now, we visualize the data with __matplotlib__ library

# Markdown Cell 27
# ### Pie Chart

# Markdown Cell 28
# Firstly, visualize the 'Protocol' column with __matplotlib pie chart__

# Cell 29
plt.figure()
x=df['Protocol'].value_counts()
plt.pie(x)
plt.show()

# Markdown Cell 30
# Add labels to the chart with __label__ parameter

# Cell 31
plt.figure()
x=df['Protocol'].value_counts()
labels=x.index
plt.pie(x,labels=labels)
plt.show()

# Markdown Cell 32
# It is seen easily that, apart from the first 6 labels (TCP, TLSv1.3, UDP, TLSv1.2, HTTP, DNS), other labels have a small count of value. So, we can limit the data with the first 6 label and grouped the rest with __Other__ label.

# Cell 33
plt.figure()
x=df['Protocol'].value_counts()[0:6] # limiting first 6 value
other_sum=sum(df['Protocol'].value_counts()[7:]) # summing the other values
x.at['Other']=other_sum #adding a new key-value pair to our list
labels=x.index
plt.pie(x,labels=labels)
plt.show()

# Markdown Cell 34
# Explode/expand the first pie slice with __explode__ parameter.

# Cell 35
plt.figure()
x=df['Protocol'].value_counts()[0:6] # limiting first 6 value
other_sum=sum(df['Protocol'].value_counts()[7:]) # summing the other values
x.at['Other']=other_sum #adding a new key-value pair to our list
labels=x.index
explode = np.append([0.05],np.zeros(len(x)-1)) # explode 1st slice
plt.pie(x,labels=labels,explode=explode)
plt.show()

# Markdown Cell 36
# Add the percentage values of data with the __autopct__ parameter.

# Cell 37
plt.figure()
x=df['Protocol'].value_counts()[0:6] # limiting first 6 value
other_sum=sum(df['Protocol'].value_counts()[7:]) # summing the other values
x.at['Other']=other_sum #adding a new key-value pair to our list
labels=x.index
explode = np.append([0.05],np.zeros(len(x)-1)) # explode 1st slice
plt.pie(x,labels=labels,explode=explode,autopct='%.02f%%')
plt.show()

# Markdown Cell 38
# Add title with fonsize of 20 with __plt.title__ function
# 
# Arrange the plot size with __figsize__ parameter

# Cell 39
plt.figure(figsize=(10,6)) #Arrange the figure size
x=df['Protocol'].value_counts()[0:6] # limiting first 6 value
other_sum=sum(df['Protocol'].value_counts()[7:]) # summing the other values
x.at['Other']=other_sum #adding a new key-value pair to our list
labels=x.index
explode = np.append([0.05],np.zeros(len(x)-1)) # explode 1st slice
plt.pie(x,labels=labels,explode=explode,autopct='%.02f%%')
plt.title('Detail Of Protocol Info', fontsize=20) # Title of chart
plt.show()

# Markdown Cell 40
# ### Histogram

# Markdown Cell 41
# Secondly, visualize the 'Protocol' column with __matplotlib histogram__
# 
# Histogram is a kind of bar chart, that shows the frequency of a data

# Cell 42
plt.figure()
plt.hist(df['Protocol'])
plt.show()

# Markdown Cell 43
# Arrange the bar chart and xticks values with __bins__ parameter

# Cell 44
plt.figure()
plt.hist(df['Protocol'],
        bins = np.arange(len(df['Protocol'].value_counts())) - 0.5)
plt.show()

# Markdown Cell 45
# Arrange the _width_ for bar chart with __rwidth__ parameter.

# Cell 46
plt.figure()
plt.hist(df['Protocol'],
        bins = np.arange(len(df['Protocol'].value_counts())) - 0.5,
        rwidth=0.5)
plt.show()

# Markdown Cell 47
# Rotate the xticks values with __plt.xticks__ function.

# Cell 48
plt.figure()
plt.hist(df['Protocol'],
        bins = np.arange(len(df['Protocol'].value_counts())) - 0.5,
        rwidth=0.5)
plt.xticks(rotation=40)

plt.show()

# Markdown Cell 49
# Add the values of data with the __plt.text__ function.
# 
# For get the x,y location of every char plot; we use the return values of plt.hist

# Cell 50
plt.figure()
counts, _, patches=plt.hist(df['Protocol'],
        bins = np.arange(len(df['Protocol'].value_counts())) - 0.5,
        rwidth=0.5)

for count, patch in zip(counts,patches):
    plt.text(x=patch.get_x()+0.05, y=patch.get_height()+50,s=str(int(count))) #x-location, y=location, value

plt.xticks(rotation=40)

plt.show()

# Markdown Cell 51
# Remove the top, left and right frames with __set.visible__ parameter
# 
# Also remove the yticks with null values

# Cell 52
plt.figure()
counts, _, patches=plt.hist(df['Protocol'],
        bins = np.arange(len(df['Protocol'].value_counts())) - 0.5,
        rwidth=0.5)

xvals = patch.get_x() # x-location
yvals = patch.get_height() # y-location
for count, patch in zip(counts,patches):
    plt.text(x=patch.get_x()+0.05, y=patch.get_height()+50,s=str(int(count)))

[plt.gca().spines[loc].set_visible(False) for loc in ['top', 'left','right']] #Remove top, left and right frame
plt.yticks([]) #disable ythicks
plt.xticks(rotation=40)


plt.show()

# Markdown Cell 53
# Add title with fonsize of 20 with __plt.title__ function
# 
# Arrange the plot size with __figsize__ parameter

# Cell 54
plt.figure(figsize=(10, 6))
counts, _, patches=plt.hist(df['Protocol'],
        bins = np.arange(len(df['Protocol'].value_counts())) - 0.5,
        rwidth=0.5)

xvals = patch.get_x() # x-location
yvals = patch.get_height() # y-location
for count, patch in zip(counts,patches):
    plt.text(x=patch.get_x()+0.05, y=patch.get_height()+50,s=str(int(count)))

[plt.gca().spines[loc].set_visible(False) for loc in ['top', 'left','right']] #Remove top, left and right frame
plt.yticks([]) #disable ythicks
plt.xticks(rotation=40)

plt.title('Detail Of Protocol Info', fontsize=20) # Title of chart

plt.show()

# Markdown Cell 55
# ### Line Chart

# Markdown Cell 56
# Thirdly, visualize the 'Length' (packet size)/'Time' column with __matplotlib line plot__

# Cell 57
plt.figure()
plt.plot(df['Time'],df['Length'])
plt.show()

# Markdown Cell 58
# As we mentioned before, the total row number is 15187! in 84 seconds. So it is not easy to show the size of packets clearly in every microseconds. So, we can prefer to visialize the time info in second level. It means, we need to sum up the total size packet for every second and grouping them. We can use __groupby__ function.

# Cell 59
plt.figure()
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum())
plt.show()

# Markdown Cell 60
# Rotate the xticks values with __plt.xticks__ function.

# Cell 61
plt.figure()
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum())

plt.xticks(rotation=40)

plt.show()

# Markdown Cell 62
# Arrange the color of line with __color__ parameter and the style of line with __linestyle__ or __ls__ parameter.

# Cell 63
plt.figure()
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum(),
        color='red',
        linestyle='--')

plt.xticks(rotation=40)

plt.show()

# Markdown Cell 64
# Add marker and arrange the marker style,color and size with __marker__,__markeredgecolor__ (or __mec__),__markerfacecolor__ (or __mfc__) and __markersize__ (or __ms__) parameters.

# Cell 65
plt.figure()
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum(),
        color='red',
        linestyle='--',
        marker='o',
        markeredgecolor='black',
        markerfacecolor='red',
        markersize=8)

plt.xticks(rotation=40)

plt.show()

# Markdown Cell 66
# Remove the top, left and right frames with __set.visible__ parameter
# 
# Also remove the yticks with null values

# Cell 67
plt.figure()
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum(),
        color='red',
        linestyle='--',
        marker='o',
        markeredgecolor='black',
        markerfacecolor='red',
        markersize=8)

[plt.gca().spines[loc].set_visible(False) for loc in ['top', 'left','right']] #Remove top, left and right frame
plt.yticks([]) #disable ythicks
plt.xticks(rotation=40)

plt.show()

# Markdown Cell 68
# Fill the area under the line with __fill_between__ function and arrange the color and transparancy of the filling with __facecolor__ and __alpha__ parameters.

# Cell 69
plt.figure()
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum(),
        color='red',
        linestyle='--',
        marker='o',
        markeredgecolor='black',
        markerfacecolor='red',
        markersize=8)

##Fill the area under the line
plt.gca().fill_between(df.groupby(by='Time')['Length'].sum().index, # x-location
                       df.groupby(by='Time')['Length'].sum(), 0,  #y-up location and y-down location
                       facecolor='blue', alpha=0.15) #parameter of color and transparency

[plt.gca().spines[loc].set_visible(False) for loc in ['top', 'left','right']] #Remove top, left and right frame
plt.yticks([]) #disable ythicks
plt.xticks(rotation=40)
  
plt.show()

# Markdown Cell 70
# Add title with fonsize of 20 with __plt.title__ function
# 
# Arrange the plot size with __figsize__ parameter

# Cell 71
plt.figure(figsize=(10,6))
plt.plot(df.groupby(by='Time')['Length'].sum().index,df.groupby(by='Time')['Length'].sum(),
        color='red',
        linestyle='--',
        marker='o',
        markeredgecolor='black',
        markerfacecolor='red',
        markersize=8)

##Fill the area under the line
plt.gca().fill_between(df.groupby(by='Time')['Length'].sum().index, # x-location
                       df.groupby(by='Time')['Length'].sum(), 0,  #y-up location and y-down location
                       facecolor='blue', alpha=0.15) #parameter of color and transparency

[plt.gca().spines[loc].set_visible(False) for loc in ['top', 'left','right']] #Remove top, left and right frame
plt.yticks([]) #disable ythicks
plt.xticks(rotation=40)
plt.title('Packet Traffic Size in Seconds', fontsize=20) # Title of chart
  
plt.show()

# Markdown Cell 72
# __Don't forget!__
# 
# This study is only prepared for a small demostration of the power of python matplotlib library. For detail information please visit <a href="https://matplotlib.org/">Matplotlib Web Site</a>

