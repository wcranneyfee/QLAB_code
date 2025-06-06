import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd


def FFT_1D_func(func, stop_point, arr_length, fig_num, rand_lim):

    plt.figure(fig_num)
    t_arr = np.linspace(0, stop_point, arr_length)
    dt = t_arr[1] - t_arr[0]

    data = func(t_arr) + np.random.uniform(-rand_lim, rand_lim, arr_length)

    plt.plot(t_arr, data)
    plt.title('Raw Data')


    plt.figure(fig_num+1)
    freq_arr = fftfreq(arr_length, dt)
    plt.plot(abs(freq_arr), abs(fft(data)))
    plt.title('Transformed Data')


def FFT_1D_array(t_arr, data, fig_num):

    if len(data) != len(t_arr):
        raise ValueError("data and t_arr must have same length")

    plt.figure(fig_num)
    dt = t_arr[1] - t_arr[0]

    plt.plot(t_arr, data)
    plt.title("Raw Data")

    plt.figure(fig_num+1)
    freq_arr = fftfreq(len(t_arr), dt)
    plt.plot(abs(freq_arr), abs(fft(data)))
    plt.title('Transformed Data')



if __name__ == '__main__':

    f1 = 30
    f2 = 100
    func = lambda x: np.sin(2*np.pi*f1*x) + 2*np.cos(2*np.pi*f2*x)
    FFT_1D_func(func, 0.2, 200, 1, 1)

    df = pd.read_csv("../Data/Qlab3/AM_signal_200k_points.csv")

    FFT_1D_array(df['Time'], df['1 (VOLT)'], 3)


    plt.show()
