import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig

Cudf = pd.read_csv('Data/NMR data/CuSO4widepicclean.csv')

# print(Cudf)

filtered_signal = sig.savgol_filter(Cudf['VOLT'], window_length=int(len(Cudf['VOLT']) / 15), polyorder=3)
plt.scatter(Cudf['Time'], Cudf['VOLT'])
plt.plot(Cudf['Time'], filtered_signal, c='red')
plt.xlabel('Time (s)')
plt.ylabel('Height (V)')

filtered_signal = np.array(filtered_signal)
time = np.array(Cudf['Time'])

data = np.column_stack((time, filtered_signal))
peaks = sig.argrelmax(data, order=1500)
neg_peaks = sig.argrelmin(data, order=1500)

net_peaks = np.append(peaks, neg_peaks)

plt.scatter(time[net_peaks], filtered_signal[net_peaks], c='purple')
plt.show()


