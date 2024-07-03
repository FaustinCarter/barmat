# coding=utf-8
import math as ma

import scipy.interpolate as si
import numpy as np

__all__ = [ 'deltar_bcs',
            'deltar_cos']

def deltar_bcs(temp):
    r"""Return the reduced BCS gap deltar = delta(T)/delta(T=0).

    Parameters
    ----------
    temp : float
        The reduced temperature temp = T/Tc, where Tc is the superconducting critical temperature.

    Returns
    -------
    dr : float
        The reduced superconducting energy gap delta(T)/delta(T=0).

    Note
    ----
    Tabulated data from Muhlschlegel (1959).
    Low-temperature analytic formula from Gao (2008).
    """

    #The BCS gap factor
    bcs = 1.764

    #These data points run from t=0.18 to t=1
    #in steps of 0.02 from Muhlschlegel (1959)
    delta_calc = [1.0, 0.9999, 0.9997, 0.9994, 0.9989,
           0.9982, 0.9971, 0.9957, 0.9938, 0.9915,
           0.9885, 0.985,  0.9809, 0.976,  0.9704,
           0.9641, 0.9569, 0.9488, 0.9399, 0.9299,
           0.919,  0.907,  0.8939, 0.8796, 0.864,
           0.8471, 0.8288, 0.8089, 0.7874, 0.764,
           0.7386, 0.711,  0.681,  0.648,  0.6117,
           0.5715, 0.5263, 0.4749, 0.4148, 0.3416,
           0.2436, 0.0]

    t_calc = np.linspace(0.18, 1, len(delta_calc))

    if 1 > temp >= 0.3:
        #interpolate data from table
        dr = float(si.interp1d(t_calc, delta_calc, kind='cubic')(temp))
    elif temp >= 1 or temp < 0:
        dr = 0.0
    elif temp < 0.005:
        dr = 1.0
    else:
        #This expression does a nice job of smoothely connecting the table to zero temp
        #Taken from Gao 2008
        dr = ma.exp(-ma.sqrt(2*bcs*temp)*ma.exp(-bcs/temp))

    return dr

def deltar_cos(temp):
    r"""Return the approximated reduced BCS gap deltar = delta(T)/delta(T=0).

    Parameters
    ----------
    temp : float
        The reduced temperature temp = T/Tc, where Tc is the superconducting critical temperature.

    Returns
    -------
    dr : float
        The reduced superconducting energy gap delta(T)/delta(T=0).

    Note
    ----
    Cosine approximation for gap taken from Pöpel (1989)."""

    if temp < 1:
        return ma.sqrt(ma.cos(0.5*ma.pi*temp**2))
    else:
        return 0
