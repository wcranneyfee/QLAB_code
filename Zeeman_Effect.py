import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from linear_regression import linear_regression
from SDOM_analysis import SDOM_analysis


def sind(x):
    x = (np.pi / 180) * x
    return np.sin(x)


def cosd(x):
    x = (np.pi / 180) * x
    return np.cos(x)

def get_peaks(unsplit_filename, split_filename, max_peaks):
    unsplit_df = pd.read_csv(f"Data/Zeeman Effect/{unsplit_filename}")
    split_df = pd.read_csv(f"Data/Zeeman Effect/{split_filename}")

    pos_peak_list = []
    neg_peak_list = []
    for df in unsplit_df, split_df:

        df.loc[abs(df['alpha']) < 2/3, ['I']] = 0

        peaks = find_peaks((df['I']), width=1)
        peaks = peaks[0]

        if hasattr(peaks, '__len__'):
            peaks = list(peaks)
        else:
            raise ValueError(f'array peaks = {peaks} has no attribute "__len__"')

        pos_peaks = [x for x in peaks if df.iloc[x]['alpha'] > 0]
        neg_peaks = [x for x in peaks if df.iloc[x]['alpha'] <= 0]

        pos_peak_list.append(pos_peaks)
        neg_peak_list.append(neg_peaks)

    delta_E_above = []
    delta_E_below = []
    delta_E_above_err = []
    delta_E_below_err = []
    for peaks in pos_peak_list, neg_peak_list:

        if peaks is pos_peak_list:
            unsplit_peaks = peaks[0][-max_peaks:]
            split_peaks = peaks[1][-2*max_peaks:]

            unsplit_peaks.reverse()
            split_peaks.reverse()

        else:
            unsplit_peaks = peaks[0][:max_peaks]
            split_peaks = peaks[1][:2*max_peaks]

        FWHM = 0.05

        for n, unsplit_peak in enumerate(unsplit_peaks):

            alpha = unsplit_df.loc[unsplit_peak, 'alpha']
            alpha_below = split_df.loc[split_peaks[2*n], 'alpha']
            alpha_above = split_df.loc[split_peaks[(2*n)+1], 'alpha']

            delta_E_above_var = get_delta_E(alpha, alpha_above)
            delta_E_below_var = get_delta_E(alpha, alpha_below)

            delta_E_above_err.append(get_delta_E_err(FWHM, alpha, alpha_above, delta_E_above_var))
            delta_E_below_err.append(get_delta_E_err(FWHM, alpha, alpha_below, delta_E_below_var))

            delta_E_above.append(delta_E_above_var)
            delta_E_below.append(delta_E_below_var)

    print(f"delta_E_above = {delta_E_above} microelectronvolts")
    print(f"delta_E_below = {delta_E_below} microelectronvolts")

    print(f"delta_E_above_err = {delta_E_above_err} microelectronvolts")
    print(f"delta_E_below_err = {delta_E_below_err} microelectronvolts")

    return delta_E_above, delta_E_below, delta_E_above_err, delta_E_below_err


# def get_FWHM(df, peaks):
#
#     FWHM_list = []
#     for peak in peaks:
#
#         half_max = df.loc[peak, 'I'] / 2
#         max_alpha = df.loc[peak, 'alpha']
#
#         # Starting from the max value of the peak, move right until the height is less than half the max
#
#         half_max_alpha = np.nan
#         for index, row in df.iterrows():
#
#             if row['alpha'] > max_alpha:
#                 continue
#
#             if row['I'] < half_max:
#                 half_max_alpha = row['alpha']
#                 break
#
#         if half_max_alpha == np.nan:
#             raise ValueError('Could not find half max')
#
#         FWHM = 2*(max_alpha - half_max_alpha)
#
#         FWHM_list.append(FWHM)
#
#     return FWHM_list


def get_delta_E(a1, a2):

    n = 1.46
    deltaE = (10**6) * (-1240 / 643.8) * ((np.sqrt(n**2 - (sind(a2)**2))) / (np.sqrt(n**2 - (sind(a1)**2))) - 1)

    return deltaE

def get_delta_E_err(FWHM, a1, a2, deltaE):

    n = 1.46

    da1 = (np.pi / 180) * FWHM
    da2 = (np.pi / 180) * FWHM

    a1 = (np.pi / 180) * a1
    a2 = (np.pi / 180) * a2

    # dEda1 = (np.sqrt(n**2 - (np.sin(a2)**2)) * np.cos(a1)*np.sin(a1)) / ((n**2 - (np.sin(a1)**2))**1.5)
    # dEda2 = (np.cos(a2) * np.sin(a2)) / ((np.sqrt(n**2 - (np.sin(a2)**2))) * (np.sqrt(n**2 - (np.sin(a1)**2))))
    #
    # dE = (10**6) * (1240 / 643.8) * np.sqrt((dEda1 * da1)**2 + (dEda2 * da2)**2)

    b1 = np.arcsin(np.sin(a1)/n)
    b2 = np.arcsin(np.sin(a2)/n)

    db1 = (np.cos(a1) * da1) / (n * np.sqrt(1 - (np.sin(a1)/n)**2))
    db2 = (np.cos(a2) * da2) / (n * np.sqrt(1 - (np.sin(a2)/n)**2))

    dE = abs(deltaE * np.sqrt((db1/b1)**2 + (db2/b2)**2))


    return dE

def plot_DeltaE_vs_Bfield(current, points, linreg, fignum):

    B_field_err = np.array(linreg[0] * linreg[1])

    slope1_list = []
    slope2_list = []
    dslope1_list = []
    dslope2_list = []
    for n, _ in enumerate(points[0][0]):

        B_field = np.array(linreg[0] * current + linreg[3])

        if n == 0 or n == 4:
            h = len(B_field)

        elif n <= 2 or 4 < n <= 6:
            h = -2

        else:
            h = -3

        plt.figure(fignum + n)
        delta_E_above = np.array([x[0][n]for x in points])[:h]
        delta_E_below = np.array([x[1][n] for x in points])[:h]
        delta_E_above_err = np.array([x[2][n] for x in points])[:h]
        delta_E_below_err = np.array([x[3][n] for x in points])[:h]

        B_field = B_field[:h]


        slope1, dslope1, _, _ = linear_regression(x_data=B_field, y_data=delta_E_above, x_err=B_field_err, y_err=delta_E_above_err)
        slope2, dslope2, _, _ = linear_regression(x_data=B_field, y_data=delta_E_below, x_err=B_field_err, y_err=delta_E_below_err)

        plt.xlabel(r'$B_{ext}$ [T]')
        plt.ylabel(r'$\Delta E$ [$\mu$ eV]')
        plt.title(f'Peak Number {n+1}')

        plt.savefig(f'figures/Zeeman_Effect_Figures/Bfield_vs_deltaE_peak_{n+1}.png')

        slope1_list.append(slope1)
        slope2_list.append(slope2)
        dslope1_list.append(dslope1)
        dslope2_list.append(dslope2)

    return slope1_list, slope2_list, dslope1_list, dslope2_list



"""First, we need to plot the current vs B-field trend"""

fignum = 1

plt.figure(fignum)
fignum += 1

BvsI = pd.read_csv('Data/Zeeman Effect/Current_vs_Field.csv')
linreg = linear_regression(x_data=BvsI['Current [A]'], y_data=BvsI['B [mT]']/1000,x_err=BvsI['Current_err'], y_err=BvsI['B_err']/1000)
plt.xlabel('Current [A]')
plt.ylabel('B [T]')
plt.savefig('figures/Zeeman_Effect_Figures/Current_vs_Field.png')

"""Now we need to process the signals and get the FWHM and peaks"""

max_peaks = 4

point1 = get_peaks('6A_no_splitting.csv', '6A_splitting.csv', max_peaks=max_peaks)

point2 = get_peaks('5point5A_no_splitting.csv', '5point5A_splitting.csv', max_peaks=max_peaks)

point3 = get_peaks('5A_no_splitting.csv', '5A_splitting.csv', max_peaks=max_peaks)

point4 = get_peaks('4point5A_no_splitting.csv', '4point5A_splitting.csv', max_peaks=max_peaks)

point5 = get_peaks('4A_no_splitting.csv', '4A_splitting.csv', max_peaks=max_peaks)

point6 = get_peaks('3point5A_no_splitting.csv', '3point5A_splitting.csv', max_peaks=max_peaks)

point7 = get_peaks('3A_no_splitting.csv', '3A_splitting.csv', max_peaks=max_peaks)

point8 = get_peaks('2point5A_no_splitting.csv', '2point5A_splitting.csv', max_peaks=max_peaks)


points = [point1, point2, point3, point4, point5, point6, point7, point8]
current = np.array([6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5]) # A

slope1_list, slope2_list, dslope1_list, dslope2_list = plot_DeltaE_vs_Bfield(current, points, linreg, fignum)


mub_list = []
mub_err_list = []
for n, (slope1, slope2, dslope1, dslope2) in enumerate(zip(slope1_list, slope2_list, dslope1_list, dslope2_list)):
        slope1 = abs(slope1 * 10**-6)
        slope2 = abs(slope2 * 10**-6)

        dslope1 = abs(dslope1 * 10**-6)
        dslope2 = abs(dslope2 * 10**-6)

        slope = (slope1 + slope2) / 2
        dslope = 0.5 * np.sqrt(dslope1**2 + dslope2**2)

        mub_list.append(slope)
        mub_err_list.append(dslope)
        print(f"Bohr magneton {n+1} = {slope} +/- {dslope} eV/T")

plt.figure(10)
SDOM_analysis(len(mub_list), mub_list, mub_err_list, accepted_val=5.788 * 10**-5)
plt.ylabel(r"$\mu_B$ [eV/T]")
plt.savefig('figures/Zeeman_Effect_Figures/SDOM_Bohr_Magneton.png')
plt.xlabel('Peak Number')

""" Now lets just make a pretty graph of peaks for the report """
plt.figure(11)

unsplit_df = pd.read_csv("Data/Zeeman Effect/6A_splitting.csv")
split_df = pd.read_csv("Data/Zeeman Effect/6A_no_splitting.csv")

plt.xlabel('alpha (degrees)')
plt.ylabel('Intensity')
plt.title('Unsplit Peaks')


plt.plot(split_df['alpha'], split_df['I'], color='blue', linewidth=0.5)
plt.savefig('figures/Zeeman_Effect_Figures/6A_Unsplit_Peaks.png')

plt.figure(12)

plt.xlabel('alpha (degrees)')
plt.ylabel('Intensity')
plt.title('Split Peaks')

plt.plot(unsplit_df['alpha'], unsplit_df['I'], color='red', linewidth=0.5)
plt.savefig('figures/Zeeman_Effect_Figures/6A_Split_Peaks.png')

plt.figure(13)

plt.xlabel('alpha (degrees)')
plt.ylabel('Intensity')
plt.title('Combined Peaks at 6A')

plt.plot(unsplit_df['alpha'], unsplit_df['I'], color='blue', linewidth=0.5, label='Unsplit')
plt.plot(split_df['alpha'], split_df['I'], color='red', linewidth=0.5, label='Split')
plt.legend()
plt.savefig('figures/Zeeman_Effect_Figures/Combined_Peaks.png')

plt.figure(14)

plt.xlabel('alpha (degrees)')
plt.ylabel('Intensity')

plt.title('Combined Peaks at 3A')
split_df = pd.read_csv("Data/Zeeman Effect/3A_splitting.csv")
unsplit_df = pd.read_csv("Data/Zeeman Effect/3A_no_splitting.csv")
plt.plot(unsplit_df['alpha'], unsplit_df['I'], color='blue', linewidth=0.5, label='Unsplit')
plt.plot(split_df['alpha'], split_df['I'], color='red', linewidth=0.5, label='Split')
plt.legend()

plt.show()

