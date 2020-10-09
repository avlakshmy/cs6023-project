# plots the timing of different versions of the sampling algorithm for different K values

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./sampling_discrete_outputs.txt')

x_axis = df['K']
y_1 = df['t1']
y_2 = df['t2']
y_3 = df['t3']
y_4 = df['t4']

plt.figure()
plt.plot(x_axis, y_1, c='red', label='Version 2.1 A', linestyle='-', marker='o')
plt.plot(x_axis, y_2, c='green', label='Version 2.1 B', linestyle='-', marker='^')
plt.plot(x_axis, y_3, c='blue', label='Version 2.1 C', linestyle='-', marker='s')
plt.plot(x_axis, y_4, c='black', label='Version 2.1 D', linestyle='-', marker='*')
plt.xlabel("K values")
plt.ylabel("Elapsed time (s)")
plt.legend(loc='upper left')
plt.savefig('./sampling_discrete_plot.png')
