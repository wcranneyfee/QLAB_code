import pandas as pd
import numpy as np
from SDOM_analysis import SDOM_analysis
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/Qlab1/lab5data(in).csv')



bkg = np.array(df['HCBackground(Counts/10s)'])


bkg = bkg[0:10]
bkg = bkg/10
print(len(bkg))
print(bkg)

SDOM_analysis(len(bkg), bkg, 0)
plt.show()


