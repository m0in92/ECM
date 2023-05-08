import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def soc(soc_init, delta_time, current, capacity):
    return soc_init - delta_time * current / (3600 * capacity)

df = pd.read_excel('../data/CALCE_A123/A1-007-OCV20-20120614.xlsx', sheet_name="Sheet1")
df = df[df.index>500]

df_discharge = df[df['Current(A)'] < 0]
t_discharge = df_discharge['Step_Time(s)'].to_numpy()
I_discharge = df_discharge['Current(A)'].to_numpy()
V_discharge = df_discharge['Voltage(V)'].to_numpy()

df_charge = df[df['Current(A)']>0]
t_charge = df_charge['Step_Time(s)'].to_numpy()
I_charge = df_charge['Current(A)'].to_numpy()
V_charge = df_charge['Voltage(V)'].to_numpy()

del df

# calc. soc at discharge
capacity = 1.1
soc_init = 1.0
soc_discharge = [soc_init]
for k in range(1,len(I_discharge)):
    delta_time = t_discharge[k] - t_discharge[k-1]
    soc_current = soc(soc_init=soc_init, delta_time=delta_time, current=-I_discharge[k-1], capacity=capacity)
    soc_discharge.append(soc_current)
    soc_init = soc_current
soc_discharge = np.array(soc_discharge)

# calc. soc at charge
capacity = 1.1
soc_init = 0.0
soc_charge = [soc_init]
for k in range(1,len(I_charge)):
    delta_time = t_charge[k] - t_charge[k-1]
    soc_current = soc(soc_init=soc_init, delta_time=delta_time, current=-I_charge[k-1], capacity=capacity)
    soc_charge.append(soc_current)
    soc_init = soc_current
soc_charge = np.array(soc_charge)

# save to csv files.
df_discharge['SOC'] = soc_discharge
df_charge['SOC'] = soc_charge
df_charge.to_csv('../data/CALCE_A123/df_discharge.csv')
df_charge.to_csv('../data/CALCE_A123/df_charge.csv')

# plot OCV vs. SOC
plt.plot(soc_discharge, V_discharge, label="discharge")
plt.plot(soc_charge, V_charge, label="charge")
plt.xlabel('SOC')
plt.ylabel('OCV [V]')
plt.legend()
plt.show()