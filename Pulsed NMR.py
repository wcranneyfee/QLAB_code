import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from fontTools.ttx import parseOptions
from scipy.optimize import curve_fit
from SDOM_analysis import SDOM_analysis
from linear_regression import linear_regression
import scipy

"""This script finds T1, T2 and T2*"""
mu_N = scipy.constants.physical_constants['nuclear magneton'][0]
h = scipy.constants.h


def exponentialFunc(x,a,b):
    y = -a*(1-2*np.exp(-x/b))
    return y


def exponentialFunc2(x,a,b,c):
    y = a*np.exp(-x/b) + c
    return y

def find_T1(filepath, fignum):

    plt.figure(fignum)
    df = pd.read_csv(filepath)
    fit = curve_fit(exponentialFunc, df['Delay Time (ms)'], df['Voltage (V)'], p0=[-25, 2.1],
                    check_finite=True)

    popt, pcov = fit[0], fit[1]

    x_arr = np.linspace(df['Delay Time (ms)'].min(), df['Delay Time (ms)'].max(), 100000)
    y_arr = exponentialFunc(x_arr, *popt)
    plt.plot(x_arr, y_arr)
    plt.scatter(df['Delay Time (ms)'].values, df['Voltage (V)'].values, color='red')
    plt.xlabel('Delay Time (ms)')
    plt.ylabel('Voltage (V)')

    print(f"T1 = {popt[1]} ms")
    return y_arr


def find_T2(filepath, fignum, cutoff=25000, argrelmax_order=1200, min_peak_height=1, max_time=20E-3):

    plt.figure(fignum)
    df = pd.read_csv(filepath)

    # # Centering the signal at zero
    # df['1 (VOLT)'] = df['1 (VOLT)'] - np.array(df['1 (VOLT)'])[-1]

    # removing the left edge of the signal
    # min_voltage = df['1 (VOLT)'].min()
    # min_ind = df[df['1 (VOLT)'] == min_voltage].index[0] - 300
    # df = df.iloc[min_ind:]

    # # removing the right edge of the signal
    # df.drop(df.tail(cutoff).index, inplace=True)

    windowlength = int(len(df['1 (VOLT)']) / 1000)
    if windowlength % 2 == 0:
        windowlength = windowlength - 1

    filtered_signal = sig.savgol_filter(df['1 (VOLT)'], window_length=windowlength, polyorder=5)
    plt.scatter(df['Time (s)'], df['1 (VOLT)'])
    plt.plot(df['Time (s)'], filtered_signal, c='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (V)')

    filtered_signal = np.array(filtered_signal)
    time = np.array(df['Time (s)'])

    data = np.column_stack((time, filtered_signal))
    peaks = sig.argrelmax(data, order=argrelmax_order)
    peaks = peaks[0]  # returns index of peaks
    peaks = peaks[0::2]  # we only want the echo peaks, which occur between b pulses

    intlist = []
    for i, peak in enumerate(peaks):
        if filtered_signal[peak] < min_peak_height:
            intlist.append(i)

    peaks = np.delete(peaks, intlist)

    plt.scatter(time[peaks], filtered_signal[peaks], c='purple')

    fit = curve_fit(exponentialFunc2, time[peaks], filtered_signal[peaks], p0=[4.6, .04, 0])
    popt, pcov = fit[0], fit[1]
    perr = np.sqrt(np.diag(pcov))

    x_arr = np.linspace(0, max_time, 1000)
    y_arr = exponentialFunc2(x_arr, *popt)

    label = f"y = ({popt[0]: .2e} +/- {perr[0]: .2e})exp(-x/({popt[0]: .2e} +/- {perr[0]: .2e})"
    plt.plot(x_arr, y_arr, linestyle='--', label=label, color='orange')
    plt.legend()

    print(f"T2 = {popt[1]*1000} ms")
    print(popt, perr)

if __name__ == '__main__':
    fignum = 1
    # find_T1('Data/Pulsed NMR Data/Water/WaterT1.csv', fignum)
    # fignum += 1

    # find_T1("Data/Pulsed NMR Data/Mineral Oil/Mineral Oil T1.csv", fignum)
    # fignum += 1

    # find_T1("Data/Pulsed NMR Data/Light Mineral Oil/Light Mineral Oil T1.csv", fignum)
    # fignum += 1

    # find_T1("Data/Pulsed NMR Data/Glycerin/glycerin_T1.csv", fignum)
    # fignum += 1

    # find_T1("Data/Pulsed NMR Data/point1MCuSO4/01MCuSO4_T1.csv", fignum)
    # fignum += 1

    # find_T1("Data/Pulsed NMR Data/wonMCuSO4/wonMCuSO4_T1.csv", fignum)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/Mineral Oil/MineralOilT2.csv", fignum, 25000)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/Water/WaterT2Data_Cleaned.csv", fignum, 0, argrelmax_order=12000)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/Light Mineral Oil/Light Mineral Oil Trace.csv", fignum, 25000)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/Glycerin/Glycerin T2 Trace.csv", fignum, argrelmax_order=500)
    # fignum += 1

    find_T2("Data/Pulsed NMR Data/point1MCuSO4/01MCuSO4_trace.csv", fignum, argrelmax_order=10000)
    fignum += 1

    # find_T2("Data/Pulsed NMR Data/wonMCuSO4/wonMCuSO4_T2_trace.csv", fignum, 25000)

    for fig in range(1, fignum):
        plt.figure(fig)
        plt.savefig(f"figures/Pulsed NMR figs/figure{fig}.png")
    plt.show()