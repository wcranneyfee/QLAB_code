import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from linear_regression import linear_regression
from scipy import interpolate

Pb_rho = 1.13e4  # kg m^-3
Al_rho = 2699  # kg m^-3
Cs_peak = 0.6617  # MeV
Co_lower_peak = 1.1732  # MeV
Co_upper_peak = 1.3325  # MeV

CoAl = pd.read_csv('../Data/Qlab1/Lab 7 - Co with Aluminum(Sheet1).csv')
CoPb = pd.read_csv('../Data/Qlab1/Lab 7 - Co with Lead(Sheet1).csv')
CsPb = pd.read_csv('../Data/Qlab1/Lab 7 - Cs with Lead(Sheet1).csv')
CsAl = pd.read_csv('../Data/Qlab1/Lab 7 - Cs with Aluminum(Sheet1).csv')
NIST_data = pd.read_csv('../Data/NIST_data.csv')

dfs = [CoAl, CoPb, CsPb, CsAl]
Co_dfs = [CoAl, CoPb]
Cs_dfs = [CsAl, CsPb]

for df in dfs:
    df['dx'] = df['dx']/1000
    df['x'] = df['x']/1000
    df['net_rate1'] = df['Anet1'] / df['t']
    df['gross_rate1'] = df['Agross1'] / df['t']

    df['net_rate1_err'] = np.sqrt(df['Anet1'])/df['t']
    df['gross_rate1_err'] = np.sqrt(df['Agross1'])/df['t']

    if 'Anet2' in df.columns:

        df['net_rate2'] = df['Anet2']/df['t']
        df['gross_rate2'] = df['Agross2']/df['t']

        df['net_rate2_err'] = np.sqrt(df['Anet2']) / df['t']
        df['gross_rate2_err'] = np.sqrt(df['Agross2']) / df['t']


d_err = np.array(CoPb['dx'])

fig1 = plt.figure(1)
plt.scatter(CsAl['x'], CsAl['net_rate1'], marker='.', label='Cs', s=150)
plt.scatter(CoAl['x'], CoAl['net_rate1'], marker='*', label='Co lower', s=150)
plt.scatter(CoAl['x'], CoAl['net_rate2'], marker='^', label='Co upper', s=150)

plt.xlabel('thickness [m]')
plt.ylabel('net rate [counts/s]')
plt.legend()
plt.title('Attenuation in Aluminum')

fig2 = plt.figure(2)
plt.scatter(CsPb['x'], CsPb['net_rate1'], marker='.', label='Cs', s=150)
plt.scatter(CoPb['x'], CoPb['net_rate1'], marker='*', label='Co 1', s=150)
plt.scatter(CoPb['x'], CoPb['net_rate2'], marker='^', label='Co 2', s=150)

plt.xlabel('thickness [m]')
plt.ylabel('net rate [counts/s]')
plt.legend()
plt.title('Attenuation in Lead')

fig3 = plt.figure(3)
x_data = CsPb['x']
y_data = np.array([math.log(n) for n in CsPb['net_rate1']])

x_err = d_err
y_err = CsPb['net_rate1_err']/CsPb['net_rate1']

slope, dslope, inter, dinter = linear_regression(x_data, y_data, x_err, y_err)

plt.title('Cs attenuation in Pb at 0.6617MeV')
plt.xlabel('thickness [m]')
plt.ylabel(r"$\ln{(R)}$")

I_o = math.exp(inter)
dI_o = math.exp(inter) * dinter
mu = -slope/Pb_rho
dmu = dslope/Pb_rho
print('CsPb')
print(f"alpha = {-slope} +/- {dslope} m^-1")
print(f"I_o = {I_o} +/- {dI_o} counts s^-1")
print(f"mu = {mu*10} +/- {dmu*10} cm^2 g^-1")

fig4 = plt.figure(4)
x_data = CsAl['x']
y_data = np.array([math.log(n) for n in CsAl['net_rate1']])

x_err = d_err
y_err = CsAl['net_rate1_err']/CsAl['net_rate1']

slope, dslope, inter, dinter = linear_regression(x_data, y_data, x_err, y_err)
plt.title('Cs Attenuation in Al at 0.6617MeV')
plt.xlabel('thickness [m]')
plt.ylabel(r'$\ln{(I)}$')

I_o = math.exp(inter)
dI_o = math.exp(inter) * dinter
mu = -slope/Al_rho
dmu = dslope/Al_rho
print('CsAl')
print(f"alpha = {-slope} +/- {dslope} m^-1")
print(f"I_o = {I_o} +/- {dI_o} counts s^-1")
print(f"mu = {mu*10} +/- {dmu*10} cm^2 g^-1")

fig5 = plt.figure(5)
x_data = CoPb['x']
y_data = np.array([math.log(n) for n in CoPb['net_rate1']])

x_err = d_err
y_err = CoPb['net_rate1_err']/CoPb['net_rate1']

slope, dslope, inter, dinter = linear_regression(x_data, y_data, x_err, y_err)
plt.title('Co attenuation in Pb at 1.1732MeV')
plt.xlabel('thickness [m]')
plt.ylabel('ln(I)')

I_o = math.exp(inter)
dI_o = math.exp(inter) * dinter
mu = -slope/Pb_rho
dmu = dslope/Pb_rho
print('CoPb lower')
print(f"alpha = {-slope} +/- {dslope} m^-1")
print(f"I_o = {I_o} +/- {dI_o} counts s^-1")
print(f"mu = {mu*10} +/- {dmu*10} cm^2 g^-1")

fig6 = plt.figure(6)
x_data = CoPb['x']
y_data = np.array([math.log(n) for n in CoPb['net_rate2']])

x_err = d_err
y_err = CoPb['net_rate2_err']/CoPb['net_rate2']

slope, dslope, inter, dinter = linear_regression(x_data, y_data, x_err, y_err)
plt.title('Co attenuation in Pb at 1.3325MeV')
plt.xlabel('thickness [m]')
plt.ylabel(r'$\ln{(I)}$')

I_o = math.exp(inter)
dI_o = math.exp(inter) * dinter
mu = -slope/Pb_rho
dmu = dslope/Pb_rho
print('CoPb upper')
print(f"alpha = {-slope} +/- {dslope} m^-1")
print(f"I_o = {I_o} +/- {dI_o} counts s^-1")
print(f"mu = {mu*10} +/- {dmu*10} cm^2 g^-1")

fig7 = plt.figure(7)
x_data = CoAl['x']
y_data = np.array([math.log(n) for n in CoAl['net_rate1']])

x_err = d_err
y_err = CoAl['net_rate1_err']/CoAl['net_rate1']

slope, dslope, inter, dinter = linear_regression(x_data, y_data, x_err, y_err)
plt.title('Co attenuation in Al at 1.1732MeV')
plt.xlabel('thickness [m]')
plt.ylabel(r'$\ln{(I)}$')

I_o = math.exp(inter)
dI_o = math.exp(inter) * dinter
mu = -slope/Al_rho
dmu = dslope/Al_rho
print('CoAl lower')
print(f"alpha = {-slope} +/- {dslope} m^-1")
print(f"I_o = {I_o} +/- {dI_o} counts s^-1")
print(f"mu = {mu*10} +/- {dmu*10} cm^2 g^-1")

fig8 = plt.figure(8)
x_data = CoAl['x']
y_data = np.array([math.log(n) for n in CoAl['net_rate2']])

x_err = d_err
y_err = CoAl['net_rate2_err']/CoAl['net_rate2']

slope, dslope, inter, dinter = linear_regression(x_data, y_data, x_err, y_err)
plt.title('Co attenuation in Al at 1.3325MeV')
plt.xlabel('thickness [m]')
plt.ylabel(r'$\ln{(I)}$')

I_o = math.exp(inter)
dI_o = math.exp(inter) * dinter
mu = -slope/Al_rho
dmu = dslope/Al_rho
print('CoAl upper')
print(f"alpha = {-slope} +/- {dslope} m^-1")
print(f"I_o = {I_o} +/- {dI_o} counts s^-1")
print(f"mu = {mu*10} +/- {dmu*10} cm^2 g^-1")

plt.figure(9)

x = NIST_data['Energy']
y = NIST_data['muPb']

plt.scatter(x, y, label='Data')
plt.xlabel('Photon Energy [MeV]')
plt.ylabel(r'$\mu$ [$cm^2 g^{-1}$]')
plt.title('Aluminum')

f = interpolate.interp1d(x, y)
plt.plot(x, y, 'o', x, f(x), '-', label='interpolation')

print(f"predicted CsPb mu value = {f(Cs_peak)} cm^2/g")
print(f"predicted CoPb lower mu value = {f(Co_lower_peak)} cm^2/g")
print(f"predicted CoPb upper mu value = {f(Co_upper_peak)} cm^2/g")
plt.xlabel('Photon Energy [MeV]')
plt.ylabel(r'$\mu$ [$cm^2 g^{-1}$]')
plt.title('Lead')

plt.figure(10)
x = NIST_data['Energy']
y = NIST_data['muAl']
plt.scatter(x, y, label='Data')

f = interpolate.interp1d(x, y)
plt.plot(x, y, 'o', x, f(x), '-', label='interpolation')
print(f"predicted CsAl mu value = {f(Cs_peak)} cm^2/g")
print(f"predicted CoAl lower mu value = {f(Co_lower_peak)} cm^2/g")
print(f"predicted CoAl upper mu value = {f(Co_upper_peak)} cm^2/g")
plt.xlabel('Photon Energy [MeV]')
plt.ylabel(r'$\mu$ [$cm^2 g^{-1}$]')
plt.title('Aluminum')

figs = [plt.figure(n) for n in range(10)]

for i, fig in enumerate(figs):
    plt.figure(i+1)
    plt.savefig(f'figures/fig{i+1}.png')
