import os
import pickle

import scipy


with open(os.path.join(os.path.dirname(__file__), "soc_discharge"), "rb") as file_soc_discharge:
    soc_discharge = pickle.load(file_soc_discharge)

with open(os.path.join(os.path.dirname(__file__), "V_discharge"), "rb") as file_V_discharge:
    V_discharge = pickle.load(file_V_discharge)

with open(os.path.join(os.path.dirname(__file__), "soc_charge"), "rb") as file_soc_charge:
    soc_charge = pickle.load(file_soc_charge)

with open(os.path.join(os.path.dirname(__file__), "V_charge"), "rb") as file_V_charge:
    V_charge = pickle.load(file_V_charge)


def func_OCV(soc):
    func_OCV_discharge = scipy.interpolate.interp1d(soc_discharge, V_discharge, fill_value="extrapolate")
    func_OCV_charge = scipy.interpolate.interp1d(soc_charge, V_charge, fill_value="extrapolate")
    return (func_OCV_discharge(soc) + func_OCV_charge(soc))/2


def func_SOC(ocv):
    func_soc_discharge = scipy.interpolate.interp1d(V_discharge, soc_discharge, fill_value="extrapolate")
    func_soc_charge = scipy.interpolate.interp1d(V_charge, soc_charge,fill_value="extrapolate")
    return (func_soc_discharge(ocv) + func_soc_charge(ocv)) / 2


def func_eta(i):
    return 1 if i<=0 else 0.9995

