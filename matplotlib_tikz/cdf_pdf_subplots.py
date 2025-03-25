# plot hourly action rate (HAR) distribution
import matplotlib as plt
import seaborn as sns

# settings
f, axes = plt.subplots(1, 2, figsize=(18,6), dpi=320)
axes[0].set_ylabel('fraction (PDF)')
axes[1].set_ylabel('fraction (CDF)')

# left plot (PDF) # REMEMBER TO CHANGE bins, xlim PROPERLY!!
sns.distplot(
    mydataframe.mycolumn, bins=5000, kde=True, axlabel='my variable',
    hist_kws={"normed":True}, ax=axes[0]
).set(xlim=(0,8))

# right plot (CDF) # REMEMBER TO CHANGE bins, xlim PROPERLY!!
sns.distplot(
    mydataframe.mycolumn, bins=50000, kde=False, axlabel='my variable',
    hist_kws={"normed":True,"cumulative":True,"histtype":"step","linewidth":4}, ax=axes[1],
).set(xlim=(0,8),ylim=(0,1))