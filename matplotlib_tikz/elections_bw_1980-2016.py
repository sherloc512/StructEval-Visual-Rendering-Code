import matplotlib.pyplot as plt
import numpy as np

# results for the greens and the cdu
gruene = np.array([5.3, 8.0, 7.9, 9.5, 12.1, 7.7, 11.7, 24.2, 30.3])
cdu = np.array([53.4, 51.9, 49.0, 39.6, 41.3, 44.8, 44.2, 39.0, 27.0])

fig, ax = plt.subplots()
xlabels = [1980, 1984, 1988, 1992, 1996, 2001, 2006, 2011, 2016]

plt.title("Regional Elections Baden-Wuerttemberg 1980-2016", size="x-large")
plt.ylabel("Votes in %", size="x-large")
plt.xlabel("Year", size="x-large")

# plot the data
plt.plot(cdu, "r*-", markersize=6, linewidth=1, color='black', label="CDU")
plt.plot(gruene, "r*-", markersize=6, linewidth=1, color='g', label="Gruene")

# add legend
plt.legend(loc=(0.1, 0.3))

# add x-labels
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels, rotation='vertical')

plt.show()