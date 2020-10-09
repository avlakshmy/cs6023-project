# plots the timing of different versions of the LDA sampling algorithm for different K values

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import pandas as pd

df = pd.read_csv('./lda_sampling_c_outputs.txt')
df1 = pd.read_csv('./lda_sampling_a1_b1_outputs.txt')

x_axis = df['K']
y_axis = df['t']

y_1 = df1['t1']
y_2 = df1['t2']

plt.figure()
plt.plot(x_axis, y_1, c='red', label='Version 2.2 A1', linestyle='-', marker='s', markersize=8)
plt.plot(x_axis, y_2, c='green', label='Version 2.2 B1', linestyle='-', marker='o', markersize=6)
plt.plot(x_axis, y_axis, c='blue', label='Version 2.2 C', linestyle='-', marker='^', markersize=6)
plt.xlabel("K values")
plt.ylabel("Elapsed time (s)")
plt.legend(loc='lower right')
plt.savefig('./lda_sampling_a1_b1_c_plot.png')
