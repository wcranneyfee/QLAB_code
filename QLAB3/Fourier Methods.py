import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Data/Qlab3/LaurenWillFMData.csv")

plt.plot(df['Time (s)'], df['1 (VOLT)'])
print(df.size)
plt.show()

