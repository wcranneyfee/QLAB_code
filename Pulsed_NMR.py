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


def find_T2(filepath, fignum, cutoff=25000, argrelmax_order=1200, min_peak_height=1, windowlength_divisor=1000, min_time=0, drop_more_indices=False):

    plt.figure(fignum)
    df = pd.read_csv(filepath)

    # removing half the points so savgolfilt doesn't take as long
    odd_indices = np.array(range(1, len(df), 2))
    df.drop(odd_indices, inplace=True)
    df = df.reset_index()

    if drop_more_indices:
        odd_indices = np.array(range(1, len(df), 2))
        df.drop(odd_indices, inplace=True)
        df = df.reset_index(drop=True)

    print(len(df))

    # Centering the signal at zero
    df['1 (VOLT)'] = df['1 (VOLT)'] - np.array(df['1 (VOLT)'])[-1]

    # removing the left edge of the signal

    min_index = 0
    for index, row in df.iterrows():
        if row['Time (s)'] > min_time:
            min_index = index
            break

    left_indices = np.linspace(0, min_index, min_index+1)
    df.drop(left_indices, inplace=True)
    df = df.reset_index(drop=True)

    # removing the right edge of the signal
    df.drop(df.tail(cutoff).index, inplace=True)

    # setting a window length
    windowlength = int(len(df['1 (VOLT)']) / windowlength_divisor)
    if windowlength % 2 == 0:
        windowlength = windowlength - 1

    print(windowlength)

    filtered_signal = sig.savgol_filter(df['1 (VOLT)'], window_length=windowlength, polyorder=3)
    filtered_signal = np.array(filtered_signal)
    time = np.array(df['Time (s)'])

    plt.scatter(time, df['1 (VOLT)'])
    plt.plot(df['Time (s)'], filtered_signal, c='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (V)')

    data = np.column_stack((time, filtered_signal))
    peaks = sig.argrelmax(data, order=argrelmax_order)
    peaks = peaks[0]  # returns index of peaks

    intlist = []
    for i, peak in enumerate(peaks):
        if filtered_signal[peak] < min_peak_height:
            intlist.append(i)

    peaks = np.delete(peaks, intlist)

    plt.scatter(time[peaks], filtered_signal[peaks], c='purple')

    fit = curve_fit(exponentialFunc2, time[peaks], filtered_signal[peaks], p0=[4.6, .04, 0])
    popt, pcov = fit[0], fit[1]
    perr = np.sqrt(np.diag(pcov))

    x_arr = np.linspace(min_time, time[-1], 1000)
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
    #
    # find_T1("Data/Pulsed NMR Data/Mineral Oil/Mineral Oil T1.csv", fignum)
    # fignum += 1
    #
    # find_T1("Data/Pulsed NMR Data/Light Mineral Oil/Light Mineral Oil T1.csv", fignum)
    # fignum += 1
    #
    # find_T1("Data/Pulsed NMR Data/Glycerin/glycerin_T1.csv", fignum)
    # fignum += 1
    #
    # find_T1("Data/Pulsed NMR Data/point1MCuSO4/01MCuSO4_T1.csv", fignum)
    # fignum += 1
    #
    # find_T1("Data/Pulsed NMR Data/wonMCuSO4/wonMCuSO4_T1.csv", fignum)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/Mineral Oil/MineralOilT2.csv", fignum, 25000, min_time=0.01)
    fignum += 1

    find_T2("Data/Pulsed NMR Data/Water/WaterT2Data_Cleaned.csv", fignum, 0, argrelmax_order=12000, drop_more_indices=True)
    fignum += 1

    # find_T2("Data/Pulsed NMR Data/Light Mineral Oil/Light Mineral Oil Trace.csv", fignum, 25000, min_time=0.01)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/Glycerin/Glycerin T2 Trace.csv", fignum, argrelmax_order=500)
    # fignum += 1

    find_T2("Data/Pulsed NMR Data/point1MCuSO4/01MCuSO4_trace.csv", fignum, argrelmax_order=10000, drop_more_indices=True)
    # fignum += 1

    # find_T2("Data/Pulsed NMR Data/wonMCuSO4/wonMCuSO4_T2_trace.csv", fignum, 25000, windowlength_divisor=1000, min_peak_height=2, min_time=-2)

    for fig in range(1, fignum+1):
        plt.figure(fig)
        plt.savefig(f"figures/Pulsed NMR figs/figure{fig}.png")
    plt.show()