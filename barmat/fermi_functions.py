# coding=utf-8
from __future__ import division
import math as ma

import scipy.constants as sc
import numba

#Eventually more complex/realistic Fermi Functions may live here. For now, just
#the standard vanilla version is available.

#TODO: Actually implement this function in the kernel.py code.

@numba.jit("float64(float64, float64, string)")
def fermi_fun(en, temp, units = 'joules'):
    """Calculate the Fermi Function given some energy and some temperature.

    Parameters
    ----------
    en : float
        Energy relative to the fermi energy (E-Ef) in units of Joules or eV.

    temp : float
        Temperature in units of Kelvin.

    Keyword Arguments
    -----------------
    units : string
        Select units of energy. Acceptable values are ``'joules' or 'eV' or
        'reduced'``. Default is ``'joules'``. Reduced units means both values
        are unitless, so onus is on the user to ensure that ``en/temp`` gives
        the desired result.

    Returns
    -------
    ffun : float
        The Fermi Function at en and temp."""

    known_units = [ 'Joules', 'joules', 'j', #Joules
                    'eV', 'ev', 'e', #Electron Volts
                    'reduced', 'r']  #Unitless proxys

    assert units in known_units, "Unknown units requested."

    #Convert temperature to joules
    if units in ['joules', 'j']
        kbt = sc.k*temp

    #Or eV if desired
    elif units in ['eV', 'ev', 'e']:
        kbt = sc.k*temp/sc.e

    #Or allow for unitless quantities
    elif units in ['reduced', 'r']:
        kbt = temp

    if en == 0:
        ffun = 0.5
    elif temp == 0:
        if en < 0:
            ffun = 1
        elif en > 0:
            ffun = 0
        else:
            raise ValueError("Energy must be a real number.")
    elif t > 0:
        #Using a tanh here instead of the more traditional formulation because
        #the function is better behaved for very large or small values of en/kbt
        ffun = 0.5*(1-ma.tanh(0.5*en/kbt))

    elif t < 0:
        raise ValueError("Temperature must be >= 0.")

    return ffun
