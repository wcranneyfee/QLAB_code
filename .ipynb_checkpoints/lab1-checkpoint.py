import numpy as np
import matplotlib.pyplot as plt
import scipy
import statistics
from math import sqrt
from scipy.stats import t

distances = np.array([177, 226, 312, 372, 455, 517, 550])  # centimeters
distances = distances * 0.01  # meters
distances = distances * 2  # this is because the light travels back and forth across the experiment

times = np.array([11, 14, 20, 24, 30, 34, 36])  # nanoseconds
times = times * 1e-9  # seconds

times_err = 2e-9  # seconds
distances_err = 0.05  # meters

plt.figure(1)
fit = scipy.stats.linregress(times, distances)
print(fit)

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.32, len(distances)-2)

print(f"slope (68%): {fit.slope} +/- {ts*fit.stderr}")
print(f"intercept (68%): {fit.intercept} +/- {ts*fit.intercept_stderr}")

# print(f"c = {1/fit.slope} +/- {(ts*fit.stderr)}")

x_fit_range = np.linspace(3, 11, 2)
y_fit_range = x_fit_range*fit.slope + fit.intercept

plt.errorbar(distances, times, xerr=distances_err, yerr=times_err, fmt='.', color='black', label='data', capsize=4)
plt.xlabel(r'$\Delta$x [m]')
plt.ylabel(r'$\Delta$t [s]')

# print(f"c = {1/fit.slope} m/s")

plt.plot(x_fit_range, y_fit_range, color='red', linestyle='--',
         label=f"y={round(fit.slope, 12)}x + {round(fit.intercept, 12)}")
plt.legend()


plt.figure(2)
measurement_indices = np.linspace(1, 7, 7)

speeds_of_light = distances / times

speed_of_light_error = []
for distance, time, speed in zip(distances, times, speeds_of_light):
    err = sqrt((distances_err/distance)**2 + (times_err/time)**2) * speed
    speed_of_light_error.append(err)

plt.errorbar(measurement_indices, speeds_of_light, yerr=speed_of_light_error, fmt='.', capsize=4, color='black',
             label='data')

mean_speed = np.mean(speeds_of_light)
print (f"mean speed = {mean_speed}")

stdev = statistics.stdev(speeds_of_light)
SDOM = stdev/sqrt(len(speeds_of_light))
print(f"SDOM = {SDOM} [m/s]")

plt.hlines(mean_speed, xmin=1, xmax=7, color='red', linestyle='--',
           label=r'$\bar{x}$ = ' + f'311,000,000 [m/s]')
plt.hlines(mean_speed+SDOM, xmin=1, xmax=7, color='blue', linestyle='--',
           label=r'$\bar{x}$ +/- $\sigma_x$')
plt.hlines(mean_speed-SDOM, xmin=1, xmax=7, color='blue', linestyle='--')

plt.xlabel('Measurement Index (i)')
plt.ylabel('Measurement of c [m/s]')


plt.legend()
plt.show()
