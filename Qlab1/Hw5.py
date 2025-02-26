from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def linfunc(x,a,b):
    y = a*x + b
    return y


# Problem 8.1
plt.figure(1)
x = np.array([1,3,5])
y = np.array([6,5,1])

a_fit, pcov = curve_fit(linfunc, x, y)

slope = a_fit[0]
inter = a_fit[1]

y_fit = x*slope + inter

plt.scatter(x,y)
plt.plot(x, y_fit, linestyle='--', color='red', label=f"y = {slope:.2e}x + {inter:.2e}")
plt.title('Problem 8.1')
plt.legend()
plt.savefig('figures/problem8-1')

# Problem 8.7
plt.figure(2)

load = np.array([200, 300, 400, 500, 600, 700, 800, 900])  # g
length = np.array([5.1, 5.5, 5.9, 6.8, 7.4, 7.5, 8.6, 9.4])  # cm

load = load/1000  # kg
length = length/100  # m

a_fit2, pcov2 = curve_fit(linfunc, load, length)

slope2 = a_fit2[0]
inter2 = a_fit2[1]

yfit = load * slope2 + inter2

plt.scatter(load, length)
plt.plot(load, yfit, linestyle='--', color='red', label=f"y = {slope:.2e}x + {inter:.2e}")
plt.xlabel('load [g]')
plt.ylabel('length [cm]')
plt.title('Problem 8.7')

k = 9.81/slope2
print(f"k = {k} N/m")
plt.legend()
plt.savefig('figures/problem8-7')

# Problems 8.8 and 8.14
plt.figure(3)

x = np.array([-4, -2, 0, 2, 4])  # seconds
y = np.array([13, 25, 34, 42, 56])  # position

a_fit, pcov = curve_fit(linfunc, x, y)

perr = np.sqrt(np.diag(pcov))

slope = a_fit[0]
inter = a_fit[1]

dslope = perr[0]
dinter = perr[1]

yfit = x*slope + inter
plt.scatter(x,y)
plt.plot(x, yfit, linestyle='--', color='red', label=f"y = {slope:.2e}x + {inter:.2e}")
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('position [cm]')
plt.title('Problem 8.8b')
plt.savefig('figures/problem8-8')

print(f"v = {slope} +/- {dslope} cm/s")

# Problem 8.15
plt.figure(4)
x = np.array([1,2,3,4,5,6])
y = np.array([5.0, 14.4, 23.1, 32.3, 41.0, 50.4])

popt, pcov = curve_fit(linfunc, x, y)

perr = np.sqrt(np.diag(pcov))

slope, inter = popt
dslope = perr[0]
dinter = perr[1]

yfit = x*slope + inter

plt.scatter(x,y)
plt.plot(x, yfit, linestyle='--', color='red', label=f"y = {slope:.2e}x + {inter:.2e}")
plt.legend()
print(f"B = {slope} +/- {dslope} cm")
plt.xlabel('node number')
plt.ylabel('position [cm]')
plt.title('Problem 8.15')
plt.savefig('figures/problem8-15')
plt.show()
