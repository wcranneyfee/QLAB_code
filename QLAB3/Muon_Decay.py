import matplotlib.pyplot as plt
from linear_regression import linear_regression
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def exp_func(x,a,b,c):
    return a*np.exp(-x/b) + c


def analyze_muon_decay(filepath, channel_to_time, is_utf=False):

    if is_utf:
        df = pd.read_csv(filepath, encoding='utf-8')
    else:
        df = pd.read_csv(filepath)

    max_cols = df.idxmax(axis=0)

    df.drop(df.head(max_cols[1]-1).index, axis=0, inplace=True)

    df['Delay Time'] = df['Channel']*channel_to_time # microseconds
    # df['Delay Time'] = df['Delay Time']/1000

    plt.scatter(df['Delay Time'], df['Data'])

    popt, pcov = curve_fit(exp_func, df['Delay Time'], df['Data'], absolute_sigma=False)

    perr = np.sqrt(np.diag(pcov))

    total_decays = np.sum(df['Data'])

    avg_decay_time = 0
    for time, mult  in zip(df['Delay Time'], df['Data']):
        avg_decay_time += (mult * time) / total_decays

    y_arr = exp_func(df['Delay Time'], *popt)
    y_above = exp_func(df['Delay Time'], *(popt + perr))
    y_below = exp_func(df['Delay Time'], *(popt - perr))

    plt.plot(df['Delay Time'], y_arr, color='r', label='Best Fit')
    plt.plot(df['Delay Time'], y_above, linestyle='--', color='g', label='Error')
    plt.plot(df['Delay Time'], y_below, linestyle='--', color='g')

    plt.xlabel(r"Delay Time ($\mu s$)")
    plt.ylabel('Counts')

    print(f"the mean decay time is {avg_decay_time} +/- {perr[2]}")
    print(f"the decay constant for {filepath} is {popt[1]} +/- {perr[1]} microseconds")

    return popt[1], perr[1]


def analyze_muon_detections(filepath, channel_to_time = 1.0):
    df = pd.read_csv(filepath)

    df['Channel'] = df['Channel']*channel_to_time
    plt.plot(df['Channel'], df['Data'])

    plt.plot(df['Channel'], (np.ones(len(df['Channel'])) * np.mean(df['Data'])),
             label=f'mean = {np.mean(df['Data']) / channel_to_time:.1f}')
    return np.mean(df['Data'])


plt.figure(1)
Diego_slope = .039
# First, we need to convert the voltage bin in the MCA to voltage
df = pd.read_csv('../Data/Qlab3/MCA_vs_TAC_voltage.csv')
slope, dslope, _, _ = linear_regression(df['Voltage Bin'], df['Delay Time (us)'], df['Voltage Bin Error'],
                                        df['Delay Time Error'], supress_err_fit=True)

print(slope)

print(f"the slope is {slope} channels per microsecond")

plt.ylabel('Voltage Bin Number')
plt.xlabel(r'Delay Time ($\mu s$)')
plt.savefig("../figures/Muon Decay Figs/channel_to_decay_time_conversion.png")

plt.figure(2)
tau_5_day, tau_5_day_err = analyze_muon_decay('../Data/Qlab3/5_day_decay_experiment_laurenandwill.csv', slope)
plt.title('5 Day Experiment')
plt.annotate(text=fr"$\tau = {tau_5_day:.1f} +/- 0.1 \mu s$", xy=(3,35), fontsize=25, weight='bold')
plt.savefig("../figures/Muon Decay Figs/5_day_decay_experiment.png")

plt.figure(3)
tau_77_day, tau_77_day_err = analyze_muon_decay('../Data/Qlab3/77dayexperiment_from_lab_manual.csv', slope)
plt.title('77 Day Experiment')
plt.annotate(text=fr"$\tau = {tau_77_day:.2f} +/- 0.01 \mu s$", xy=(4,350), fontsize=25, weight='bold')
plt.savefig("../figures/Muon Decay Figs/77_day_decay_experiment.png")

plt.figure(4)
mean = analyze_muon_detections('../Data/Qlab3/1hr_lifetime_experiment_LaurenandWill.csv')
plt.xlabel('Seconds')
plt.ylabel('Counts')
print(f"We found approximately {mean:.2f} muons detected per second.")
plt.legend()
plt.savefig("../figures/Muon Decay Figs/one_hour_muon_detection.png")

plt.figure(5)
tau_2_day, tau_2_day_err = analyze_muon_decay('../Data/Qlab3/April_10_2day_decay_experiment.csv', Diego_slope)
plt.title('April 10 2 Day Experiment')
plt.annotate(text=fr"$\tau = {tau_2_day:.1f} +/- {tau_2_day_err:.1f} \mu s$", xy=(2.2,17), fontsize=25, weight='bold')
plt.savefig("../figures/Muon Decay Figs/April_10_2_day_decay_experiment.png")

plt.figure(6)
mean = analyze_muon_detections('../Data/Qlab3/April_17_2day_lifetime_Diego_19_77.csv', channel_to_time=19.77)
plt.xlabel('Seconds')
plt.ylabel('Counts')
print(f"We found approximately {mean/19.77:.1f} muons detected per second.")
plt.legend()
plt.savefig("../figures/Muon Decay Figs/2_day_muon_detection.png")

plt.figure(7)
tau_2_day_2nd_exp, tau_2d_2nd_exp = analyze_muon_decay('../Data/Qlab3/April_17_2day_decay_experiment_Diego.csv', Diego_slope)
plt.title('April 17 2 Day Decay Experiment')
plt.annotate(text=fr"$\tau = {tau_2_day_2nd_exp:.1f} +/- {tau_2_day_err:.1f} \mu s$", xy=(2.2,17), fontsize=25, weight='bold')
plt.savefig("../figures/Muon Decay Figs/April_17_2_day_decay_experiment.png")

plt.show()
