import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression
import pandas as pd
import scipy

df = pd.read_csv('Data/optical_pumping.csv')

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

vert_coil_earth = helmholz_coil(0.117,20, vert_I)
horiz_coil_earth = helmholz_coil(0.163, 11, horiz_I)

print(f"vertical_field = {vert_coil_earth.field_strength():.2e} [T],"
      f" horizontal field = {horiz_coil_earth.field_strength():.2e} [T]")

print(f"field_mag = {np.sqrt(vert_coil_earth.field_strength()**2 + horiz_coil_earth.field_strength()**2):.2e} [T], "
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

df['Rb_85_fields'] = B_fields_rb_85
df['Rb_87_fields'] = B_fields_rb_87


err_in_sweep = helmholz_coil(0.163, 11, current=df['Rb_err'][0])
Rb_field_err = err_in_sweep.field_strength()
Rb_field_err = np.ones(df.shape[0])*Rb_field_err

df['Rb_field_err'] = Rb_field_err

df['f'] = df['f']*1000
df['f_err'] = df['f_err']*1000

print(df[['Rb_87_fields', 'Rb_85_fields']])

df['Rb_85_fields'] = np.array(df['Rb_85_fields']) - horiz_coil_earth.field_strength()
df['Rb_87_fields'] = np.array(df['Rb_87_fields']) - horiz_coil_earth.field_strength()

print(df[['Rb_87_fields', 'Rb_85_fields']])

fig, ax = plt.subplots()
slope85, dslope85, inter85, dinter85 = linear_regression(df['Rb_85_fields'], df['f'], df['Rb_field_err'], df['f_err'])
slope87, dslope87, inter87, dinter87 = linear_regression(df['Rb_87_fields'], df['f'], df['Rb_field_err'], df['f_err'])
plt.xlabel('B [T]')
plt.ylabel('f [Hz]')
ax.get_legend().remove()

g_lande_85 = slope85*(h/bohrmag)
g_lande_87 = slope87*(h/bohrmag)
print(g_lande_85, g_lande_87)

print(inter85, inter87)

# print(g_lande_85, g_lande_87)
plt.show()
