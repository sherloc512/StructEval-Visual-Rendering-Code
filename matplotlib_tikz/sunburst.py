import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['legend.fontsize'] = 14

group_names = ['不依赖其他条目', '依赖其他条目']
group_size = [24, 40]
subgroup_names = ['', '1个', '2个', '3个', '4个', '5个', '9个', '13个']
subgroup_size = [24, 15, 13, 5, 3, 2, 1, 1]

a, b, c = [plt.cm.RdPu, plt.cm.GnBu, plt.cm.Greys]

fig, ax = plt.subplots()
ax.axis('equal')

mypie, _ = ax.pie(subgroup_size, radius=1.3, labels=subgroup_names, labeldistance=0.85, colors=[c(0.0), b(0.7), b(0.6), b(0.5), b(0.4), b(0.3), b(0.2), b(0.1)])
plt.setp(mypie, width=0.3, edgecolor='white')

mypie2, _ = ax.pie(group_size, radius=1.3-0.3, labels=group_names, colors=[a(0.5), b(0.7)], labeldistance=0.6)
plt.setp(mypie2, width=0.4, edgecolor='white')

plt.margins(0, 0)

plt.tight_layout()
plt.savefig('sunburst.png')