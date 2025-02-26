import numpy as np
import matplotlib.pyplot as plt
import math

"""Code for problem 5.32"""


def Poisson(mean, vals):

    if hasattr(vals, '__len__'):
        P = [np.exp(-mean)*(mean**val)/math.factorial(int(val)) for val in vals]

    else:
        P = np.exp(-mean)*(mean**vals)/math.factorial(int(vals))

    return P

# Problem 5.32

arr = np.array([8.16, 8.14, 8.12, 8.16, 8.18, 8.10, 8.18, 8.18, 8.18, 8.24, 8.16, 8.14, 8.17, 8.18,
                8.21, 8.12, 8.12, 8.17, 8.06, 8.10, 8.12, 8.10, 8.14, 8.09, 8.16, 8.16, 8.21, 8.14, 8.16, 8.13])

std_arr = np.std(arr, ddof=1)
print(f"standard deviation = {std_arr}")

trials = {}
for n in range(10):
    trials[f"trial{n+1}"] = arr[3*n:(3*n)+3]

# print(trials)

avgs = {}
for n, (key, val) in enumerate(trials.items()):
    avg = np.sum(val)/len(val)
    avgs[f"avg{n+1}"] = avg

avgs_arr = np.array(list(avgs.values()))

std_means = np.std(avgs_arr, ddof=1)

print(f"standard deviation of averages = {std_means}")

plt.figure(1)
plt.hist(arr, bins=10)
plt.title('histogram of all 30 data points for problem 5.32')
plt.savefig('figures/problem5-32a')

plt.figure(2)
plt.hist(avgs_arr, bins=10)
plt.title('histogram of all 10 averages for problem 5.32')
plt.savefig('figures/problem5-32b')

# Problem 11.2

x_ax = np.linspace(0, 6, 7)

poisson_arr = []
poisson_arr2 = []
for n in x_ax:
    n = int(n)
    poisson_arr.append(Poisson(1, n))
    poisson_arr2.append(Poisson(2, n))

fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter(x_ax, poisson_arr)
ax1.set_xlabel(r'$\nu$')
ax1.set_ylabel('Probability')

ax2.scatter(x_ax, poisson_arr2)
ax2.set_xlabel(r'$\nu$')
ax2.set_ylabel('Probability')

plt.title('Problem 11.2')
plt.savefig('figures/problem11-2')

# Problem 11.3

Problem113 = Poisson(1.5, np.linspace(0, 5))

# Problelm 11.5

Problem115 = Poisson(3, np.linspace(0, 5, 6))

plt.figure(4)
plt.scatter(np.linspace(0,5,6), Problem115, label='Poisson Distribution')

ndecays = np.linspace(0,9,10)
times = np.array([5, 19, 23, 21, 14, 12, 3, 2, 1, 0])

N = np.sum(times)
fraction = times/N

plt.scatter(ndecays, fraction, label='Histogram')
plt.title('Problem 11.5')
plt.ylabel('Probability')
plt.xlabel('Measurement Index')
plt.legend()
plt.savefig('figures/problem11-5')

# Problem 11.12
n = np.linspace(0,10,11)
print(n)
prob_v_under_10 = np.sum(Poisson(16, n))
print(prob_v_under_10)

