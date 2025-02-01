import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression
from SDOM_analysis import SDOM_analysis
from math import sqrt

electron_charge = 1.602176634E-19  # coulombs
speed_of_light = 299792458  # m/s
h_accepted = 6.62607015E-34  # J/Hz

# KEmax = hf - w
# qVstop = hf - w
# h = (qVstop+w)/f

wavelength = np.array([640E-9, 575E-9, 540E-9, 464E-9, 437E-9, 404E-9])  # m
wavelength_err = 5E-9  # m

Vstop = np.array([-0.419, -0.619, -0.696, -1.032, -1.182, -1.364])  # volts
Vstop_err = 0.01  # volts

KEmax = Vstop*-1.60217663E-19  #
KEmax_err = Vstop_err*1.60217663E-19

frequency = speed_of_light/wavelength
frequency_err = (wavelength_err/wavelength)*frequency

plt.figure(1)
plt.xlabel('frequency [Hz]')
plt.ylabel('KE max [J]')

slope, d_slope, work_func, work_func_err = linear_regression(frequency, KEmax, frequency_err, KEmax_err)

work_func = abs(work_func)
work_func_err = abs(work_func_err)

h_calc = (KEmax+work_func)/frequency

top = KEmax+work_func
top_err = sqrt(KEmax_err**2 + work_func_err**2)

h_calc_err = []
for f, df, h, t in zip(frequency, frequency_err, h_calc, top):
    val = sqrt((top_err/t)**2 + (df/f)**2)*h
    h_calc_err.append(val)

plt.figure(2)
plt.ylabel('h [J/Hz]')
SDOM_analysis(frequency, h_calc, h_calc_err)

plt.figure(3)

voltage = np.array([3.00, 2.00, 1.00, 0, -1.00])
current = np.array([14.0, 13.1, 10.6, 2.0, 0.0052])

voltage_err = 0.01
current_err = 0.01

plt.scatter(voltage, current, s=100, color='black')
# plt.errorbar(voltage, current, xerr=voltage_err, yerr=current_err, fmt='.', capsize=2)
plt.xlabel('Voltage [V]')
plt.ylabel('Current [nA]')

plt.show()
