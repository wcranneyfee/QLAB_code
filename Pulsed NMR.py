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


def exponentialFunc(x,a,b,c):
    y = a*np.exp(x/b) + c
    return y


def fit_and_plot_T1(filepath, fignum):
    df = pd.read_csv(filepath)
    fit = curve_fit(exponentialFunc, df['Delay Time (s)'], df['Voltage (V)'], p0=[-25, 2.1, 13],
                    check_finite=True)

    popt, pcov = fit[0], fit[1]
    print(popt)

    x_arr = np.linspace(df['Delay Time (s)'].min(), df['Delay Time (s)'].max(), 100000)
    y_arr = exponentialFunc(x_arr, *popt)
    plt.plot(x_arr, y_arr)
    plt.scatter(df['Delay Time (s)'].values, df['Voltage (V)'].values, color='red')
    plt.xlabel('Delay Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()

    return y_arr

fit_and_plot_T1('Data/Pulsed NMR Data/Water/WaterT1.csv', 1)
plt.show()