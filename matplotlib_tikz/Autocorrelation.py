# Converted from: Autocorrelation.ipynb

# Markdown Cell 1
# # Autocorelation: What it is, why it matters, what it tells us, and how to use it. 

# Markdown Cell 2
# ### The accompanying blog post can be found here:
# ### NOAA dataset: 
# ``curl https://s3.amazonaws.com/noaa.water-database/NOAA_data.txt -o NOAA_data.txt
# influx -import -path=NOAA_data.txt -precision=s -database=NOAA_water_database``

# Markdown Cell 3
# ### All of my data is stored in InfluxDB. I am using the Python CL to query the data and perform autocorrelation analysis. To learn more about how to use the InfluxDB-Python CL please take a look at this post:

# Cell 4
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from influxdb import InfluxDBClient
from pandas.plotting import autocorrelation_plot
from scipy.stats import linregress
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# Markdown Cell 5
# <h2>1) Does our water temp data have any Autocorrelation? </h2>

# Markdown Cell 6
# <p>Connect to the client and use an influxql query to retreive and plot data.</p>

# Cell 7
client = InfluxDBClient(host='localhost', port=8086)
h2O = client.query('SELECT mean("degrees") AS "h2O_temp" FROM "NOAA_water_database"."autogen"."h2o_temperature"  GROUP BY time(12h) LIMIT 60')
h2O_points = [p for p in h2O.get_points()]
h2O_df = pd.DataFrame(h2O_points)
h2O_df['time_step'] = range(0,len(h2O_df['time']))
h2O_df.plot(kind='line',x='time_step',y='h2O_temp')
plt.show()

# Markdown Cell 8
# <p>You can use .autocorr() to determine R_xx (autocorrelation) for comparisons of the data(i) with data(i-lag).</p> 

# Cell 9
shift_1 = h2O_df['h2O_temp'].autocorr(lag=1)
shift_2 = h2O_df['h2O_temp'].autocorr(lag=2)
print(shift_1)
print(shift_2)

# Markdown Cell 10
# <p>An autocorrelation plot is used to asses the randomness of the time series data. Random data has an R_k value close to 0 for all time lags (k) or shifts. Solid lines correspond to 95% confidence interval while dashed lines correspond to 99% confidence interval. </p>

# Cell 11
autocorrelation_plot(h2O_df['h2O_temp'])
plt.show()

# Cell 12
plot_acf(h2O_df['h2O_temp'], lags=20)
plt.show()

# Markdown Cell 13
# <p> From the graph above we can clearly see that our data displays little autocorrelation.</p>

# Markdown Cell 14
# <h2> 2) What does having little autocorrelation mean? </h2>

# Markdown Cell 15
# <p> Having no or little autocorrelation tells us our data is in fact random. Knowing this is important consideration when selecting a prediction or forecasting method. </p> 

# Markdown Cell 16
# <h2> 3) What else can autocorrelation tell us? </h2>

# Markdown Cell 17
# <h3>Let's look at the water levels. This is the same dataset that was used in this  <a href="https://www.influxdata.com/blog/how-to-use-influxdbs-holt-winters-function-for-predictions/">blog series</a> on how to use the built-in Holt-Winters Prediction algorithm. The statistical assumptions for using Holt-Winters are that the data is: a) random, b)has seasonality, and c)has trend. </h3>

# Cell 19
client = InfluxDBClient(host='localhost', port=8086)
h2O_level = client.query('SELECT "water_level" FROM "NOAA_water_database"."autogen"."h2o_feet" WHERE "location"=\'santa_monica\' AND time >= \'2015-08-22 22:12:00\' AND time <= \'2015-08-28 03:00:00\'')
h2O_level_points = [p for p in h2O_level.get_points()]
h2O_level_df = pd.DataFrame(h2O_level_points)
h2O_level_df['time_step'] = range(0,len(h2O_level_df['time']))
h2O_level_df.plot(kind='line',x='time_step',y='water_level')
plt.show()

# Markdown Cell 20
# <p> Let's verify our assumption that the data isn't random and that we can in fact use a predictor like Holt Winters </p>

# Cell 21
autocorrelation_plot(h2O_level_df['water_level'])
plt.show()

# Cell 22
plot_acf(h2O_level_df['water_level'], lags=300)
plt.show()

# Cell 23
plot_pacf(h2O_level_df['water_level'], lags=300)
plt.show()

# Markdown Cell 24
# <p> We see that our data has significantly non-zero autocorrelation values so it is in fact random. We can also verify that our data has seasonality and trend. The autocorelation plot shows an oscillation, indicative of a seasonal series. To verify that our data has a trend, we can first remove the seasonality from our data and then take a look a the autocorrelation again.
# 
# Specifically the distance between each peak is 379m</p>

# Markdown Cell 25
# <h3> Removing seasonality: Differencing </h3>

# Markdown Cell 26
# <p> Differencing is a method of transforming a time series dataset. It can be used to remove seasonal components of the series as well as trend. In this example we'll be removing the seasonality. Taking a lag difference will allow us to do so. A lagged difference is defined by:</p>
#     <p>difference(t) = observation(t) - observation(t-lag)</p>

# Cell 27
from datetime import datetime 
h2O_level_df['time'].head()
h2O_level_time = [p[:-1] for p in h2O_level_df['time']]
h2O_level_time = [datetime.strptime(p,'%Y-%m-%dT%H:%M:%S') for p in h2O_level_time]

# Cell 28
h2O = h2O_level_df.copy()
h2O['time'] = pd.DataFrame(h2O_level_time)
h2O.drop(columns = ['time_step'], inplace = True)
h2O.set_index('time')
h2O.head()

# Cell 29
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
result = seasonal_decompose(h2O['water_level'], model='additive', freq=250)
result.plot()
pyplot.show()

# Cell 30
def difference(dataset, interval):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.DataFrame(diff, columns = ["water_level_diff"])
h2O_level_diff = difference(h2O_level_df['water_level'], 246)
h2O_level_diff['time_step'] = range(0,len(h2O_level_diff['water_level_diff']))
h2O_level_diff.plot(kind='line',x='time_step',y='water_level_diff')
plt.show()

# Cell 31
plot_acf(h2O_level_diff['water_level_diff'], lags=300)
plt.show()

# Cell 32
plot_pacf(h2O_level_diff['water_level_diff'], lags=300)
plt.show()

# Markdown Cell 33
# <h2> 4) Taking a Look at Correlation </h2>

# Cell 34
h2O_level_diff_df = h2O_level_df.copy()
# h2O_level_diff_df['difference'] = [h2O_level_diff_df['water_level'][p] - h2O_level_diff_df['water_level'][p-]
h2O_level_diff_df['water_level'][0]

# Cell 35
h2O_temp_array = h2O_df['h2O_temp'].values
h2O_temp_array = [float(p) for p in h2O_temp_array]
time_array = list(range(len(h2O_df['time']))) 
time_array
np.corrcoef(h2O_temp_array,time_array)[0,1]

# Cell 36
client = InfluxDBClient(host='localhost', port=8086)
temp = client.query('SELECT mean("degrees") as "air_temp" FROM "NOAA_water_database"."autogen"."average_temperature" Group By time(12h) limit 60')
temp_points = [p for p in temp.get_points()]
temp_df = pd.DataFrame(temp_points)
temp_df['time_step'] = range(0,len(temp_df['time']))
temp_df.plot(kind='line',x='time_step',y='air_temp')
plt.show()

# Markdown Cell 37
# ### 1) Does the temperature of the water have any correlation with the temperature of the air? 

# Markdown Cell 38
# <p> Let's plot the air_temp against the h2O_temp and use visual inspection to make a hypothesis </p>

# Cell 39
ax = temp_df.plot(kind='line',y='air_temp')
ax2 = h2O_df.plot(kind='line',x='time_step',y='h2O_temp')
ax.set_ylim(79.25,81)
ax2.set_ylim(64,65.75)
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=0)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
plt.show()

# Markdown Cell 40
# <p> By plotting the two together, it doesn't look like there's any sort of correlation between the two. Let's verify that assumption with a correlation plot</p>

# Cell 41
h2O_air = h2O_df.copy()
h2O_air['air_temp'] = temp_df['air_temp']
h2O_air.drop(columns=['time_step'], inplace=True)
h2O_air.head()
corr = h2O_air.corr()
corr.style.background_gradient(cmap='coolwarm')

# Markdown Cell 42
# <p> We see that there's a Pearson Correlation Coefficient (r) of 0.108 which indicates low correlation. This makes sense. For example, the water temperatures of a spring fed stream can stay constant the entire year. The water temperature doesn't vary with the temperature of the air. However, we expect to get a positive r value because the water temperature and air temperatures probably change in tandem. 

# Cell 43
h2O_air.plot(kind='scatter',x='air_temp',y='h2O_temp')
slope, intercept, r_value, p_value, std_err = linregress(h2O_air['air_temp'],h2O_air['h2O_temp'])
plt.plot(h2O_air['air_temp'], h2O_air['h2O_temp'], 'o', label='original data')
plt.plot(h2O_air['air_temp'], intercept + slope*h2O_air['air_temp'], 'r', label='fitted line')
plt.legend()
plt.show()
print("slope: %f intercept: %f r_value: %f" % (slope, intercept, r_value))

