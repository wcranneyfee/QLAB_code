import numpy as np
import pandas as pd
import math
import circle_fit as cf
from SDOM_analysis import SDOM_analysis
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from statistics import mean
from linear_regression import linear_regression

currents = np.array([0.340, 0.420, 0.479, 0.538, 0.588])  # amps
dI = 0.001  # amps
dV_a = 5  # volts
N = 131  # loops
radius_of_helmholz_coil = 10.1E-2  # m
d_helmholz_radius = 0.25E-2  # m
mu_o = 4*math.pi*10**-7

helmholz_diameter = radius_of_helmholz_coil*2
d_helmholz_diameter = 2*d_helmholz_radius


def uncertainty_of_q_exp_n(q, dq, n):
    d_q_squared = dq/q
    d_q_squared = d_q_squared * 2 * q

    return d_q_squared


def electron_mass_ratio(V_a, B, R):

    em_ratio = (2*V_a)/(B**2 * R**2)
    em_ratio_err = []

    return em_ratio, em_ratio_err


def B_field_strength(I):

    B = (16 * mu_o * N * I) / (math.sqrt(125) * 2 * radius_of_helmholz_coil)
    B_err = []

    return B, B_err


y_offset = {'X': [2, 4, 6, 8, 10],
            'Y': [-0.05, -0.1, -0.1, -0.15, -0.15]}  # cm

voltages = np.array([1000, 1500, 2000, 2500, 3000])  # volts

df = pd.read_csv('../Data/Qlab1/Lab3forCSV(Sheet1).csv')

for index, row in df.iterrows():
    for offset_index, x_position in enumerate(y_offset['X']):
        if row['X'] == x_position:
            df.loc[index, 'Y'] = abs(row['Y'] - y_offset['Y'][offset_index])

df_normal = df.loc[df['ArcDirection'] == 'Normal']
df_reversed = df.loc[df['ArcDirection'] == 'Reversed']

for x_position in y_offset['X']:
    for voltage in voltages:
        normal_y_row = df_normal.loc[(df_normal['X'] == x_position) & (df_normal['HV'] == voltage)]
        reversed_y_row = df_reversed.loc[(df_reversed['X'] == x_position) & (df_reversed['HV'] == voltage)]

        normal_y = normal_y_row['Y'].values[0]
        reversed_y = reversed_y_row['Y'].values[0]

        y_avg = (normal_y + reversed_y)/2

        df_normal.loc[normal_y_row.index, 'Y'] = y_avg

df = df_normal

y_list = []
x_center_list = []
y_center_list = []
radius_list = []
for voltage in voltages:
    filt = (df['HV'] == voltage)

    only_one_voltage = df.loc[filt]
    x_vals = only_one_voltage['X'].values/100   # converting to meters
    y_vals = only_one_voltage['Y'].values/100  # converting to meters
    err = only_one_voltage['dHV'].values[0]/100   # converting to meters

    XY = list(zip(x_vals, y_vals))

    y_list.append(y_vals)

    xc, yc, r, sigma = cf.standardLSQ(XY)

    x_center_list.append(xc)
    y_center_list.append(yc)
    radius_list.append(r)

# print(f"radii = {radius_list}")

e_over_m_list = []
for radius, voltage, current in zip(radius_list, voltages, currents):

    B_field = B_field_strength(current)[0]
    e_over_m = electron_mass_ratio(voltage, B_field, radius)[0]

    e_over_m_list.append(e_over_m)

    # print('%E' % Decimal(e_over_m))

xc_avg = mean(x_center_list)
yc_avg = mean(y_center_list)
radius_avg = mean(radius_list)

dXc = np.std(x_center_list)
dYc = np.std(y_center_list)
dR = np.std(radius_list)

d_e_over_m_list = []
for V_a, em, I, R in zip(voltages, e_over_m_list, currents, radius_list):
    d_e_over_m = math.sqrt((dV_a/V_a)**2 + 4*(dI/I)**2 + 4*(dR/R)**2 + 4*(d_helmholz_diameter/helmholz_diameter)**2)*em
    d_e_over_m_list.append(d_e_over_m)

for em, de in zip(e_over_m_list, d_e_over_m_list):
    print(f"e/m = {em:.2e} +/- {de:.1e}")

plt.figure(1)
mean_e_over_m, stdev_val, SDOM_val = SDOM_analysis(len(voltages), e_over_m_list, d_e_over_m_list, accepted_val=1.75882000838E11)
plt.ylabel('e/m')

fig = plt.figure(2)
ax = plt.subplot()


V1000 = plt.scatter(y_offset['X'], y_list[0]*100, color='red', marker='o')
V1500 = plt.scatter(y_offset['X'], y_list[1]*100, color='green', marker='D')
V2000 = plt.scatter(y_offset['X'], y_list[2]*100, color='purple', marker='x')
V2500 = plt.scatter(y_offset['X'], y_list[3]*100, color='fuchsia', marker='s')
V3000 = plt.scatter(y_offset['X'], y_list[4]*100, color='orange', marker='*')

center = (xc_avg*100, yc_avg*100)
center2 = ((xc_avg+dXc)*100, (yc_avg+dYc)*100)
center3 = ((xc_avg-dXc)*100, (yc_avg-dYc)*100)

circle = Circle(center, radius_avg*100, facecolor='none', edgecolor='red', label='best circular fit')
circle2 = Circle(center2, (radius_avg+dR)*100, facecolor='none', edgecolor='blue', linestyle='--')
circle3 = Circle(center3, (radius_avg-dR)*100, facecolor='none', edgecolor='blue', linestyle='--', label='uncertainty')

plt.legend([circle, circle2, V1000, V1500, V2000, V2500, V3000],
           ['averaged least squares fit', 'average uncertainty', r'$V_a$ = 1000V', r'$V_a$ = 1500V', r'$V_a$ = 2000V',
            r'$V_a$ = 2500V', r'$V_a$ = 3000V'])

ax.add_patch(circle)
ax.add_patch(circle2)
ax.add_patch(circle3)

plt.xlim(0, 11)
plt.ylim(0, max(y_list[0])+2.4)

plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')


plt.figure(3)
x_points = currents**2

d_x_points = [dI/current for current in currents]
d_x_points = np.array(d_x_points)
d_x_points = d_x_points*2*x_points

y_points = voltages
d_y_points = dV_a

slope, d_slope, y_int, d_y_int = linear_regression(x_points, y_points, d_x_points, d_y_points)
plt.xlabel(r'$I^2 [A^2]$')
plt.ylabel(r'$V_a [V]$')

print(f"avg radius = {radius_avg}")

proportionality_constant = (125 * 2 * helmholz_diameter**2) / (16**2 * mu_o**2 * N**2 * radius_avg**2)
lin_reg_charge_ratio = slope * proportionality_constant


d_D_squared = uncertainty_of_q_exp_n(helmholz_diameter, d_helmholz_diameter, 2)

d_radius_squared = uncertainty_of_q_exp_n(radius_avg, dR, 2)

d_D_squared_over_radius_squared = (math.sqrt((d_D_squared/helmholz_diameter**2)**2 +
                                             (d_radius_squared/radius_avg**2)**2) *
                                   (helmholz_diameter**2 / radius_avg**2))

d_proportionality_constant = ((125 * 2) / (16**2 * mu_o**2 * N**2)) * d_D_squared_over_radius_squared

d_lin_reg_charge_ratio = (math.sqrt((d_slope/slope)**2 * (d_proportionality_constant/proportionality_constant)**2) *
                          lin_reg_charge_ratio)

print(f"e/m from linear regression = {lin_reg_charge_ratio:.2e} +/- {d_lin_reg_charge_ratio:.2e}")

print(mean_e_over_m+2*stdev_val)

# plt.show()

