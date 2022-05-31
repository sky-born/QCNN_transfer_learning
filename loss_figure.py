from cProfile import label
import pandas as pd
data = pd.read_csv("save/loss_history.csv")

import matplotlib.pyplot as plt
import numpy as np

c_01=data.iloc[11,1:].to_numpy().astype(np.float32)
x = np.linspace(1, 200, 200)

c_01_l=data.iloc[12,1:].to_numpy().astype(np.float32)
c_01_h=data.iloc[13,1:].to_numpy().astype(np.float32)

c_23=data.iloc[26,1:].to_numpy().astype(np.float32)
c_23_l=data.iloc[27,1:].to_numpy().astype(np.float32)
c_23_h=data.iloc[28,1:].to_numpy().astype(np.float32)

c_89=data.iloc[41,1:].to_numpy().astype(np.float32)
c_89_l=data.iloc[42,1:].to_numpy().astype(np.float32)
c_89_h=data.iloc[43,1:].to_numpy().astype(np.float32)

plt.plot(x,c_01, color='#CC4F1B',label='0,1 classification')
plt.fill_between(x,c_01_l,c_01_h,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x,c_23, color='#1B2ACC',label='2,3 classification')
plt.fill_between(x,c_23_l,c_23_h,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

plt.plot(x,c_89, color='#3F7F4C',label='8,9 classification')
plt.fill_between(x,c_89_l,c_89_h,alpha=0.2, edgecolor='#3F7F4C', facecolor='#7EFF99')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()
