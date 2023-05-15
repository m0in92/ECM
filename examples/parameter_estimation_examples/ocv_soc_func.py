import pandas as pd
import scipy.interpolate as sci_interp


df_discharge = pd.read_csv('../../data/CALCE_A123/df_discharge.csv')
soc_discharge = df_discharge['SOC'].to_numpy()
V_discharge = df_discharge['Voltage(V)'].to_numpy()

df_charge = pd.read_csv('../../data/CALCE_A123/df_charge.csv')
soc_charge = df_discharge['SOC'].to_numpy()
V_charge = df_discharge['Voltage(V)'].to_numpy()

del df_discharge, df_charge

def OCV_discharge():
    return sci_interp.interp1d(soc_discharge, V_discharge, fill_value="extrapolate")

def SOC_discharge():
    return sci_interp.interp1d(V_discharge, soc_discharge, fill_value="extrapolate")

def OCV_charge():
    return sci_interp.interp1d(soc_charge, V_charge, fill_value="extrapolate")

def SOC_charge():
    return sci_interp.interp1d(V_charge, soc_charge,fill_value="extrapolate")

def OCV_func(soc):
    return (OCV_discharge()(soc) + OCV_charge()(soc))/2

def SOC_func(ocv):
    return (SOC_charge()(ocv) + SOC_discharge()(ocv)) / 2
