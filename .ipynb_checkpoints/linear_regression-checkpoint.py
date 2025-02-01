import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linearFunc(x, intercept, slope):
    y = intercept + (slope * x)
    return y


def linear_regression(x_data, y_data, x_err, y_err):
    a_fit, pcov = curve_fit(linearFunc, x_data, y_data, sigma=y_err, absolute_sigma=True, full_output=False)

    inter = a_fit[0]
    slope = a_fit[1]
    perr = np.sqrt(np.diag(pcov))

    d_inter = perr[0]
    d_slope = perr[1]

    plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='k.', label='Data')

    yfit = inter + slope*x_data
    yfit2 = (inter+d_inter) + (slope+d_slope)*x_data
    yfit3 = (inter-d_inter) + (slope-d_slope)*x_data

    plt.plot(x_data, yfit, label=f"y={slope:.2e}+/-{d_slope:.1e}x + "
                                 f"{inter:.2e}+/-{d_inter:.1e}", color='r', linestyle='-')

    plt.plot(x_data, yfit2, color='g', linestyle='--', label='uncertainty')

    plt.plot(x_data, yfit3, color='g', linestyle='--')

    plt.legend()

    print(f'slope = {slope:.2e}, with uncertainty {d_slope:.1e}')
    print(f'intercept = {inter:.2e}, with uncertainty {d_inter:.1e}')

    return slope, d_slope, inter, d_inter







