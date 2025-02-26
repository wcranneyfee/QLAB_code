import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from linear_regression import linear_regression
import scipy as sp


def get_peaks(unsplit_filename, split_filename, central_peak_angle_angle_cutout):
    unsplit_df = pd.read_csv(f"Data/Zeeman Effect/{unsplit_filename}")
    split_df = pd.read_csv(f"Data/Zeeman Effect/{split_filename}")

    plt.plot(split_df['alpha'], split_df['I'], color='red', label='Split data')
    plt.plot(unsplit_df['alpha'], unsplit_df['I'], color='blue', label='Unsplit data')


    # df_net = pd.concat([unsplit_df, split_df])
    # df_net.reset_index(drop=True, inplace=True)
    # plt.scatter(df_net['alpha'], df_net['I'])

    unsplit_peaks = find_peaks(np.array(split_df['I']), width=8)
    split_peaks = find_peaks(np.array(unsplit_df['I']), width=8)

    unsplit_peaks, split_peals = unsplit_peaks[0], split_peaks[0]


    print(f"{len(unsplit_peaks + split_peaks)} peaks")

    plt.scatter(df_net['alpha'][peaks], df_net['I'][peaks], c='purple', label='Determined maxima')

    return df_net


def get_FWHM(filename):
    pass

def get_delta_alpha(center_peak, split_peaks):
    pass

def get_delta_E(a1, a2):
    B1 = np.arcsin(np.sin(a1)/1.46)
    B2 = np.arcsin(np.sin(a2)/1.46)

    deltaE = (-1240/643.8) * (np.cos(B1)/np.cos(B2) - 1)

    return deltaE


"""First, we need to plot the current vs B-field trend"""

fignum = 1
#
# plt.figure(fignum)
# fignum += 1
# BvsI = pd.read_csv('Data/Zeeman Effect/Current_vs_Field.csv')
# linear_regression(x_data=BvsI['Current [A]'], y_data=BvsI['B [mT]']/1000,x_err=BvsI['Current_err'], y_err=BvsI['B_err']/1000)
# plt.xlabel('Current [A]')
# plt.ylabel('B [T]')
# plt.show()

"""Now we need to process the signals and get the FWHM and peaks"""

plt.figure(fignum)
fignum += 1
get_peaks('6A_no_splitting.csv', '6A_splitting.csv')
plt.show()

