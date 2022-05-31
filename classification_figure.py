import pandas as pd
data = pd.read_csv("save/cl_01_csv.csv",index_col=0)

import matplotlib.pyplot as plt
import numpy as np


zx_pooling = data.iloc[0,:].to_numpy()
general_pooling = data.iloc[1,:].to_numpy()
d1_176 = data.iloc[2,:].to_numpy()
d1_350 = data.iloc[3,:].to_numpy()
d2_76 = data.iloc[4,:].to_numpy()
d2_377 = data.iloc[5,:].to_numpy()
d_dense = data.iloc[6,:].to_numpy()
x_label = [1,2,3,4,5,6,7,8,9,10,11]

x = np.arange(len(zx_pooling))

width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, zx_pooling, width, label='ZX pooling')
rects2 = ax.bar(x + width/2, general_pooling, width, label='General pooling')
plt.ylim(50,100)
ax.plot(d1_176,label='1D 176', linestyle = ':',color='c')
ax.plot(d1_350,label='1D 350', linestyle = 'dashdot',color='y')
ax.plot(d2_76,label='2D 76', linestyle = 'solid',color='r')
ax.plot(d2_377,label='2D 377', linestyle = (0, (3, 5, 1, 5, 1, 5)),color='m')
ax.plot(d_dense,label='Dense', linestyle = 'dashed', color='g')

plt.xticks(x,x_label)
plt.xlabel('QCNN Convolution')
plt.ylabel('Accuracy(%)')



# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=5)

plt.show()