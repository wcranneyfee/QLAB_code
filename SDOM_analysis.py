import numpy as np
import matplotlib.pyplot as plt
import statistics
from math import sqrt


def SDOM_analysis(points_num, val, val_err, accepted_val=0):
    measurement_indices = np.linspace(1, points_num, points_num)

    plt.errorbar(measurement_indices, val, yerr=val_err, fmt='.', capsize=4, color='black',
                 label='data')

    mean_val = np.mean(val)
    print(f"mean = {mean_val:2e}")

    stdev = statistics.stdev(val)
    SDOM = stdev/sqrt(len(val))
    print(f"SDOM = {SDOM:2e}")
    print(f"STDEV = {stdev:2e}")

    plt.hlines(mean_val, xmin=1, xmax=points_num+1, color='red', linestyle='-',
               label=fr'$\bar{{x}}$ =' + f"{mean_val:.2e}")
    plt.hlines(mean_val+SDOM, xmin=1, xmax=points_num+1, color='blue', linestyle='-.',
               label=r'$\bar{x}$ +/- $\sigma_\bar{x}$')
    plt.hlines(mean_val-SDOM, xmin=1, xmax=points_num+1, color='blue', linestyle='-.')

    plt.hlines(mean_val+stdev, xmin=1, xmax=points_num+1, color='orange', linestyle='--',
               label=r'$\bar{x}$ +/- $\sigma_x$')
    plt.hlines(mean_val-stdev, xmin=1, xmax=points_num+1, color='orange', linestyle='--')

    if accepted_val != 0:
        plt.hlines(accepted_val, xmin=1, xmax=points_num + 1, color='green', linestyle=':', label='accepted value')

    plt.xlabel('Measurement Index (i)')

    plt.legend()

    return mean_val, stdev, SDOM


if __name__ == '__main__':
    x_data = [1, 2, 3, 4]
    val = [2.40, 2.45, 2.4, 2.45]

    SDOM_analysis(x_data, val, val_err=0)
