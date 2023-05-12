import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('../data/NASA/00001.csv')
print(df.columns)
t = df['Time']
I = df['Current_measured']
V = df['Voltage_measured']
T = df['Temperature_measured']

plt.plot(t, I)
plt.plot(t, T)
plt.plot(t, V)
plt.show()