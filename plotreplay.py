import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('run.csv', skiprows=0, header=0)
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

num = 0
bottom=0

# Add percentage plot
plt.subplot(2, 1, 1)
plt.title("Percentage distribution of predicted sounds", loc='left', fontsize=12, fontweight=0, color='black')
plt.ylabel("Percentage")

for column in df.drop(['winner', 'intensity', 'time'], axis=1):
	color = palette(num)
	
	if(column == "silence"):
		color = "w"
	
	num+=1
	plt.bar(np.arange(df['time'].size), df[column], color=color, linewidth=1, alpha=0.9, label=column, bottom=bottom)
	bottom += np.array( df[column] )
 
plt.legend(loc=1, bbox_to_anchor=(1, 1.3), ncol=4)

plt.subplot(2, 1, 2)

# Add audio subplot
plt.title('Audio', loc='left', fontsize=12, fontweight=0, color='black')
plt.ylabel('Loudness')
plt.xlabel("Time( s )")
plt.ylim(ymax=40000)
plt.plot(np.array( df['time'] ), np.array( df['intensity'] ), '-')

plt.show()
