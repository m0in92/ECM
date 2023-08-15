import pickle

import pandas as pd


df_discharge = pd.read_csv('../../data/CALCE_A123/df_discharge.csv')
lst_soc_discharge = df_discharge['SOC'].to_list()
lst_V_discharge = df_discharge['Voltage(V)'].to_list()

df_charge = pd.read_csv('../../data/CALCE_A123/df_charge.csv')
lst_soc_charge = df_discharge['SOC'].to_numpy()
lst_V_charge = df_discharge['Voltage(V)'].to_list()

with open("soc_discharge", "wb") as file_soc_discharge:
    pickle.dump(lst_soc_discharge, file_soc_discharge)

with open("V_discharge", "wb") as file_V_discharge:
    pickle.dump(lst_V_discharge, file_V_discharge)

with open("soc_charge", "wb") as file_soc_charge:
    pickle.dump(lst_soc_charge, file_soc_charge)

with open("V_charge", "wb") as file_V_charge:
    pickle.dump(lst_V_charge, file_V_charge)
