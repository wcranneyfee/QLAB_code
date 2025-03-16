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
    y = a*(1-2*np.exp(-x/b))
    return y


def exponentialFunc2(x,a,b):
    y = a*np.exp(-x/b)
    return y

def find_T1(filepath, fignum, change_units=False):

    plt.figure(fignum)
    df = pd.read_csv(filepath)

    if change_units:
        df['Delay Time (ms)'] = df['Delay Time (ms)']/1000
        units = 's'
    else:
        units = "ms"

    fit = curve_fit(exponentialFunc, df['Delay Time (ms)'], df['Voltage (V)'],
                    check_finite=True)

    popt, pcov = fit[0], fit[1]
    perr = np.sqrt(np.diag(pcov))

    x_arr = np.linspace(df['Delay Time (ms)'].min(), df['Delay Time (ms)'].max(), 1000)
    y_arr = exponentialFunc(x_arr, *popt)

    above_err = np.array(popt) + np.abs(np.array(perr))
    below_err = np.array(popt) - np.abs(np.array(perr))
    y_arr_error_above = exponentialFunc(x_arr, *above_err)
    y_arr_error_below = exponentialFunc(x_arr, *below_err)

    label = f"y = ({popt[0]: .2e} +/- {perr[0]: .2e})(1 - 2exp(-x/({popt[0]: .2e} +/- {perr[0]: .2e})))"
    plt.plot(x_arr, y_arr, label=label)
    plt.plot(x_arr, y_arr_error_above, color='green', linestyle='--', label='error')
    plt.plot(x_arr, y_arr_error_below, color='green', linestyle='--')
    plt.legend()

    plt.scatter(df['Delay Time (ms)'].values, df['Voltage (V)'].values, color='red')
    plt.xlabel(f'Delay Time ({units})')
    plt.ylabel('Voltage (V)')



    print(f"T1 = {popt[1]} +/- {perr[1]} {units}")
    return y_arr


def find_T2(filepath, fignum, cutoff=25000, argrelmax_order=1200, min_peak_height=float(1), windowlength_divisor=1000, min_time=float(0), drop_factor=2, polyorder=3):

    plt.figure(fignum)
    df = pd.read_csv(filepath)

    # removing half the points so savgolfilt doesn't take as long
    kept_indices = np.array(range(1, len(df), drop_factor))
    kept_indices = [int(round(x)) for x in kept_indices]
    df = df.iloc[kept_indices]
    df = df.reset_index(drop=True)

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

    # print(f"{len(df)} = length of df")

    # setting a window length
    windowlength = int(len(df['1 (VOLT)']) / windowlength_divisor)
    if windowlength % 2 == 0:
        windowlength = windowlength - 1

    # print(f"{windowlength} = window length")

    filtered_signal = sig.savgol_filter(df['1 (VOLT)'], window_length=windowlength, polyorder=polyorder)
    filtered_signal = np.array(filtered_signal)
    time = np.array(df['Time (s)'])

    plt.scatter(time, df['1 (VOLT)'], label='oscilloscope trace data', s=0.1)
    plt.plot(df['Time (s)'], filtered_signal, c='red', label='smoothed signal')
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
    # print(f"{len(peaks)} peaks")

    plt.scatter(time[peaks], filtered_signal[peaks], c='purple')

    fit = curve_fit(exponentialFunc2, time[peaks], filtered_signal[peaks], p0=[4.6, .04])
    popt, pcov = fit[0], fit[1]
    perr = np.sqrt(np.diag(pcov))

    x_arr = np.linspace(min_time, time[-1], 1000)
    y_arr = exponentialFunc2(x_arr, *popt)

    error_above = np.array(popt) + np.abs(np.array(perr))
    error_below = np.array(popt) - np.abs(np.array(perr))

    y_above = exponentialFunc2(x_arr, *error_above)
    y_below = exponentialFunc2(x_arr, *error_below)

    label = f"y = ({popt[0]: .2e} +/- {perr[0]: .2e})exp(-x/({popt[0]: .2e} +/- {perr[0]: .2e})"
    plt.plot(x_arr, y_arr, linestyle='--', label=label, color='orange')
    plt.plot(x_arr, y_above, color='green', linestyle='-.', label='error')
    plt.plot(x_arr, y_below, color='green', linestyle='--')
    plt.legend()

    print(f"T2 = {popt[1]*1000} ms +/- {perr[1]*1000} ms")
    # print(popt, perr)

if __name__ == '__main__':
    fignum = 1
    find_T1('../Data/Pulsed NMR Data/Water/WaterT1.csv', fignum, change_units=True)
    fignum += 1

    find_T1("../Data/Pulsed NMR Data/Mineral Oil/Mineral Oil T1.csv", fignum)
    fignum += 1

    find_T1("../Data/Pulsed NMR Data/Light Mineral Oil/Light Mineral Oil T1.csv", fignum)
    fignum += 1

    find_T1("../Data/Pulsed NMR Data/Glycerin/glycerin_T1.csv", fignum)
    fignum += 1

    find_T1("../Data/Pulsed NMR Data/point1MCuSO4/01MCuSO4_T1.csv", fignum)
    fignum += 1

    find_T1("../Data/Pulsed NMR Data/wonMCuSO4/wonMCuSO4_T1.csv", fignum)
    fignum += 1

    find_T2("../Data/Pulsed NMR Data/Mineral Oil/MineralOilT2.csv", fignum, 25000, min_time=0.01)
    fignum += 1

    find_T2("../Data/Pulsed NMR Data/Water/WaterT2Data_Cleaned.csv", fignum,
            0, argrelmax_order=1200, drop_factor=4, windowlength_divisor=3000, polyorder=5, min_peak_height=0.7)
    fignum += 1

    find_T2("../Data/Pulsed NMR Data/Light Mineral Oil/Light Mineral Oil Trace.csv", fignum, cutoff=50000, min_time=0.01)
    fignum += 1

    find_T2("../Data/Pulsed NMR Data/Glycerin/Glycerin T2 Trace.csv", fignum, argrelmax_order=500, min_peak_height=0.8)
    fignum += 1

    find_T2("../Data/Pulsed NMR Data/point1MCuSO4/01MCuSO4_trace.csv", fignum,
            argrelmax_order=2000, drop_factor=8, min_peak_height=0.1, windowlength_divisor=60, polyorder=3)
    fignum += 1

    find_T2("../Data/Pulsed NMR Data/wonMCuSO4/wonMCuSO4_T2_trace.csv", fignum,
            25000, windowlength_divisor=50, min_peak_height=1.5, drop_factor=6, argrelmax_order=1500)

    for fig in range(1, fignum+1):
        plt.figure(fig)
        plt.savefig(f"figures/Pulsed NMR figs/figure{fig}.png")
    plt.show()