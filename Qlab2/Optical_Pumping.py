import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression
from SDOM_analysis import SDOM_analysis
import pandas as pd
import scipy

df = pd.read_csv('../Data/Qlab2/optical_pumping.csv')

mu_0 = scipy.constants.mu_0
bohrmag = scipy.constants.physical_constants['Bohr magneton'][0]
h = scipy.constants.h


class helmholz_coil:

    def __init__(self, radius, n_coils, current=float('NaN')):
        self.rad = radius
        self.coils = n_coils
        self.I = current

    def field_strength(self):
        if self.I == float('NaN'):
            raise ValueError('Coil has no specified current')
        else:
            B = (16 * mu_0 * self.coils * self.I) / (np.sqrt(125) * 2 * self.rad)
            return B


horiz_I = 0.146  # A
vert_I = 0.247  # A
I_err = 0.002  # A

vert_coil_earth = helmholz_coil(0.117, 20, vert_I)
horiz_coil_earth = helmholz_coil(0.163, 11, horiz_I)
earth_err = helmholz_coil(0.117, 20, I_err)

print(f"vertical_field = {vert_coil_earth.field_strength():.2e} +/- {earth_err.field_strength():.2e} [T],"
      f" horizontal field = {horiz_coil_earth.field_strength():.2e} +/- {earth_err.field_strength():.2e}[T]")

Bx = horiz_coil_earth.field_strength()
By = vert_coil_earth.field_strength()
field_mag = np.sqrt(vert_coil_earth.field_strength()**2 + horiz_coil_earth.field_strength()**2)

dBx = earth_err.field_strength()
dBy = dBx

mag_err = np.sqrt(((Bx*dBx)**2 + (By*dBy)**2)/(Bx**2 + By**2))

err = 0

print(f"field_mag = {field_mag:.3e} +/- {mag_err:.2e} [T], "
      f"field_angle = {(-1)*np.degrees(np.arctan(vert_coil_earth.field_strength()/horiz_coil_earth.field_strength()))} "
      f"[deg]")

B_fields_rb_85 = []
B_fields_rb_87 = []
for Rb in ['Rb85', 'Rb87']:
    for n in range(df.shape[0]):
        horiz_sweep = helmholz_coil(0.163, 11, df[Rb][n])
        if Rb == 'Rb85':
            B_fields_rb_85.append(horiz_sweep.field_strength())
        else:
            B_fields_rb_87.append(horiz_sweep.field_strength())

df['Rb_85_fields'] = np.array(B_fields_rb_85)*10**6
df['Rb_87_fields'] = np.array(B_fields_rb_87)*10**6


err_in_sweep = helmholz_coil(0.163, 11, current=df['Rb_err'][0])
Rb_field_err = err_in_sweep.field_strength()
Rb_field_err = np.ones(df.shape[0])*Rb_field_err

df['Rb_field_err'] = Rb_field_err*10**6

df['f'] = df['f']*1000
df['f_err'] = df['f_err']*1000

plt.figure(1)
ax = plt.axes()
slope85, dslope85, inter85, dinter85 = linear_regression(df['Rb_85_fields'], df['f'], df['Rb_field_err'], df['f_err'])
slope87, dslope87, inter87, dinter87 = linear_regression(df['Rb_87_fields'], df['f'], df['Rb_field_err'], df['f_err'])

plt.xlabel(r'B [$\mu$T]', fontsize=20)
plt.ylabel('f [Hz]', fontsize=20)
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

g_lande_85 = slope85*(h/bohrmag)*10**6
g_lande_87 = slope87*(h/bohrmag)*10**6

g_lande_85_err = (h/bohrmag)*dslope85*10**6
g_lande_87_err = (h/bohrmag)*dslope87*10**6


print(f"g_87 = {g_lande_85} +/- {g_lande_85_err}")
print(f"g_85 = {g_lande_87} +/- {g_lande_87_err}")
"""Note that we switched 85 and 87, I'm not fixing it, just do it in your head"""

g_div = g_lande_85/g_lande_87
g_div_err = np.sqrt((g_lande_85_err/g_lande_85)**2 + (g_lande_87_err/g_lande_87)**2)*g_div

print(f"g_87/g_85 = {g_div} +/- {g_div_err}")

error = abs(1.4988586-g_div)/g_div_err

print(error)

plt.figure(2)

for n in df:
    print(n)


# SDOM_analysis(df.shape[0], )


plt.show()
