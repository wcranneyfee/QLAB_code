import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import math


def linfunc(x, a, b):
    y = a*x + b
    return y


def FWHM_to_STD(x, input_type):
    if input_type == 'FWHM':
        out = x/(2*math.log(2))

    elif input_type == 'STD':
        out = x*2*math.log(2)

    else:
        raise ValueError('input type not recognized')

    return out


fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

centroids = np.array([3126.6, 3511.29, 351.4, 826.75, 976.96, 178.35])  # counts

FWHMs = np.array([103, 104, 44.3385, 42, 68.777, 31.2])  # counts

errors = [FWHM_to_STD(x, 'FWHM') for x in FWHMs]


energies = {'Co-60 1': 1121,
            'Co-60 2': 1332,
            'Co-57': 122,
            'Ba 1': 302,
            'Ba 2': 356,
            'Am': 59}  # kEV

Cs137 = {'centroid': 1777.54, 'FWHM': 112.4891}

plt.figure(1)
plt.scatter(centroids, list(energies.values()), marker='o', color='black')
plt.errorbar(centroids, list(energies.values()), xerr=errors, linestyle='', color='black', capsize=3)

afit, pcov = curve_fit(linfunc, centroids, list(energies.values()), sigma=errors, absolute_sigma=True)
slope = afit[0]
inter = afit[1]

perr = np.sqrt(np.diag(pcov))

d_slope = perr[0]
d_inter = perr[1]

print(d_inter, d_slope)

centroids_sorted = np.array(sorted(centroids))

yfit1 = inter + slope*centroids_sorted
yfit2 = inter+d_inter + (slope + d_slope)*centroids_sorted
yfit3 = inter-d_inter + (slope - d_slope)*centroids_sorted

ax1.plot(centroids_sorted, yfit1, linestyle='--', color='red', label=rf"y={slope:.2e})x + {inter:.2e}")
ax1.plot(centroids_sorted, yfit2, linestyle='-.', color='green', label='uncertainty')
ax1.plot(centroids_sorted, yfit3, linestyle='-.', color='green')

Cs137_peak = slope*Cs137['centroid']
Cs137_err = FWHM_to_STD(Cs137['FWHM'], 'FWHM')

Cs137_peak_err = np.sqrt((d_slope/slope)**2 + (Cs17_err/Cs137['centroid'])**2)*Cs137_peak

plt.errorbar(Cs137['centroid'], Cs137_peak, yerr=Cs137_peak_err, linestyle='', capsize=3, color='black')
plt.scatter(Cs137['centroid'], Cs137_peak, marker='*', edgecolors='black', s=200, color='gold')

energies['Cs'] = Cs137_peak
centroids = np.append(centroids,Cs137['centroid'])


ax1.set_xlabel('Voltage Bins')
ax2.set_xlabel('Radioactive Sources')

ax2.set_xticks(centroids, list(energies.keys()))
ax1.set_ylabel('Energy [keV]')

for centroid in centroids:
    plt.axvline(x=centroid, linestyle='--', color='black', linewidth='0.7')

print(f"energy = voltage bin * {slope:.2e} +/- {d_slope:.2e}")
print(f"Cs137 photo peak is at {Cs137_peak} +/- {Cs137_peak_err} keV")

ax1.legend()
plt.savefig('figures/lab7fig.png')
plt.show()
s