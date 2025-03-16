import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linfunc(x, a, v_o): return x*a + v_o

t = [10, 20, 30, 50, 70, 80]
v = [18, 22, 26, 40, 50, 68]
plt.scatter(t,v)
plt.xlabel('t')
plt.ylabel('v')

popt, pcov = curve_fit(linfunc, t, v)
perr = np.sqrt(np.diag(pcov))

x = np.linspace(0, max(t), 100)
y = linfunc(x, *popt)
y_above = linfunc(x, popt[0]+perr[0], popt[1]+perr[1])
y_below = linfunc(x, popt[0]-perr[0], popt[1]-perr[1])
plt.plot(x, y)
plt.plot(x, y_above, color='red', linestyle='--')
plt.plot(x, y_below, color='red', linestyle='--')

print(f" a = {popt[0]} +/- {perr[0]}")
print(f" v_0 = {popt[1]} +/- {perr[1]}")
plt.show()

plt.figure(2)
v_0 = np.array([8, 16, 24, 40, 64])
v_f = np.array([6, 18, 33, 71, 143])

plt.scatter(v_0, v_f)
plt.xlabel(r"$v_0$")
plt.ylabel(r"$v_f$")

plt.savefig("figures/Computer_AssessmentQ5.png")

plt.figure(3)

def f(x): return 31.82 + (1.02 * np.log(x-1874))

x = np.linspace(1875, 2017, 100)

plt.plot(x, f(x))
plt.xlabel('Year')
plt.ylabel('Predicted Speed')
plt.savefig("figures/Computer_AssessmentQ6.png")
