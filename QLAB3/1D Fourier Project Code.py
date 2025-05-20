import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import pandas as pd


def FFT_1D(func, frequency, size, rand_lim):

    plt.figure(1)
    x_arr = np.linspace(0,2*np.pi, size)
    data = func(x_arr) + np.random.uniform(-rand_lim, rand_lim, size)

    freq = [(2 * np.pi) / x for x in x_arr]
    plt.plot(x_arr, data)
    plt.title('Raw Data')


    plt.figure(2)
    plt.plot(freq, fft(data))
    plt.title('Transformed Data')


if __name__ == '__main__':

    data = pd.read_csv('../Data/Qlab3/CuO_070523(in).csv')
    y_data = data['Intensity']

    plt.figure(1)
    func = lambda x: np.sin(2*np.pi*x)
    FFT_1D(func, 2, 1000, 0.001)

    # plt.figure(3)
    # plt.scatter(data['theta/2theta'], data['Intensity'])
    # plt.title('Raw Data')
    #
    # plt.figure(4)
    # plt.scatter(data['theta/2theta'], fft(data['Intensity']))
    # plt.title('Transformed Data')

    plt.show()
