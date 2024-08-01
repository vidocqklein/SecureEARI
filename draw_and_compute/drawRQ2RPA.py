import matplotlib.pyplot as plt
import numpy as np
#add
data = ((75.72, 74.63, 70.75), (73.36, 71.90, 70.62), (75.45, 74.39,71.11), (82.53, 81.95, 80.12), (77.94, 77.33,75.00), (85.31, 84.22,83.13))
models = ['GCN', 'GUARD', 'RGCN', 'MedianGCN', 'NoisyGNN', 'PPG']
dim = len(data[0])
w = 0.75
dimw = w / dim

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(data))
for i in range(len(data[0])):
    y = [d[i] for d in data]
    b = ax.bar(x + i * dimw, y, dimw, bottom=0.001)

ax.set_xticks(x + dimw )    #This is just to set the scale at the center of each group of data
#ax.set_yscale('log')       #Calculate the y-coordinate with log
ax.set_xticklabels(models)

# ax.set_xlabel('models')
ax.set_ylabel('Accuracy%')
# ax.legend(('ptb_rate=3.0%','ptb_rate=5.0%','ptb_rate=7.0%'),loc='upper right')
ax.legend(('ptb_rate=3.0%','ptb_rate=5.0%','ptb_rate=7.0%'), bbox_to_anchor=(0.5, 1.05), loc='center', ncol=len(('ptb_rate=3.0%','ptb_rate=5.0%','ptb_rate=7.0%')))
plt.show()
plt.savefig('plot.svg', format='svg')