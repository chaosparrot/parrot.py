import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('run.csv', skiprows=0, header=0)
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

num = 0
for column in df.drop(['winner', 'intensity', 'time'], axis=1):
	num+=1
	plt.bar(df['time'], df[column], color=palette(num), linewidth=1, alpha=0.9, label=column)
 
plt.legend(loc=1, bbox_to_anchor=(1, -0.01), ncol=4)
 
# Add titles
plt.title("Percentage", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time")
plt.ylabel("Percentage")
plt.show()
