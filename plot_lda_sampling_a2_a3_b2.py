# plots the timing of different versions of the LDA sampling algorithm for different K values

import matplotlib.pyplot as plt
import pandas as pd

df2 = pd.read_csv('./lda_sampling_a2_b2_outputs.txt')
df3 = pd.read_csv('./lda_sampling_a3_outputs.txt')

x_axis = df2['K']
y_1_par = df2['t1']
y_2_par = df2['t2']
y_1_par_par = df3['t1']

plt.figure()
plt.plot(x_axis, y_1_par, c='red', label='Version 2.2 A2', linestyle='-', marker='s', markersize=6)
plt.plot(x_axis, y_1_par_par, c='blue', label='Version 2.2 A3', linestyle='-', marker='s', markersize=6)
plt.plot(x_axis, y_2_par, c='green', label='Version 2.2 B2', linestyle='-', marker='o', markersize=6)

plt.xlabel("K values")
plt.ylabel("Elapsed time (s)")
plt.legend(loc='lower right')
plt.savefig('./lda_sampling_a2_a3_b2_plot.png')
