# coding=utf-8
from __future__ import division
import math as ma

import scipy.constants as sc

#Eventually more complex/realistic Fermi Functions may live here. For now, just
#the standard vanilla version is available.

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
        Select units of energy. Acceptable values are ``'joules' or 'eV'``.
        Default is ``'joules'``. Temperature units still must be in Kelvin.

    Returns
    -------
    ffun : float
        The Fermi Function at en and temp."""

    assert units in ['joules', 'j', 'eV', 'ev', 'e'], "Unknown units requested."

    #Convert temperature Kelvin to Joules
    kbt = sc.k*temp

    #Or eV if desired
    if units in ['eV', 'ev', 'e']:
        kbt /= sc.e

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
