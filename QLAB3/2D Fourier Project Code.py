import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from clathrate import clathrate

def FFT_2D(arr, fig_num):

    fft_arr = fftshift(fft2(arr))

    plt.figure(fig_num)
    plt.imshow(arr)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure(fig_num + 1)
    mags = np.abs(fft_arr)
    plt.imshow(mags)
    plt.xlabel('kx')
    plt.ylabel('ky')

def build_checkerboard(n):
    return np.array([[(i + j) % 2 for j in range(n)] for i in range(n)])

def build_clathrate():

    clath = clathrate(x=0.18, y=0.3, z=)
    clathrate_coords = np.concatenate()

if __name__ == '__main__':


    FFT_2D(build_checkerboard(11), 3)
    plt.show()