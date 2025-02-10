import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.optimize import curve_fit
from SDOM_analysis import SDOM_analysis
from linear_regression import linear_regression
import scipy

"""This script finds T1, T2 and T2*"""
mu_N = scipy.constants.physical_constants['nuclear magneton'][0]
h = scipy.constants.h


def exponentialFunc(x,a,b):
    y = np.exp(a*x) + b
    return y

def fit_and_plot_T1(filepath):
    df = pd.read_csv(filepath)
    popt, pcov = curve_fit(exponentialFunc, df[''])