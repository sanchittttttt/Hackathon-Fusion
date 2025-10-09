# metrics.py
import numpy as np

def pKd_to_Kd(pKd):
    return 10**(-pKd) * 1e9  # nM

def Kd_to_Ki(Kd_nM):
    return Kd_nM

def Kd_to_IC50(Kd_nM):
    return Kd_nM

def Kd_to_EC50(Kd_nM):
    return Kd_nM

def pKd_to_DeltaG(pKd, temperature=298):
    R = 8.314
    Kd_M = 10 ** (-pKd)
    deltaG = -R * temperature * np.log(Kd_M)
    return deltaG  # Joules/mole
