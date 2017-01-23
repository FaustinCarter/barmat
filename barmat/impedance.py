# coding=utf-8
from __future__ import division
import cmath as cm

import numpy as np
import scipy.constants as sc
import scipy.integrate as sint

from .tools import do_integral, init_from_physical_data
from .kernel import cmplx_kernel
from .gap_functions import deltar_bcs

__all__ = [ 'get_Z',
            'cmplx_impedance']

def get_Z(input_vector, tc, vf, london0, axis='temperature', **kwargs):
    r"""Return Z at several temperatures or frequencies.

    Parameters
    ----------
    input_vector : list-like
        List of either reduced temperatures (T/Tc) or reduced frequencies
        (h*f/delta0), where Tc is the superconducting critical temperature, h is
        Planck's constant and delta0 is the zero-temperature superconducting
        energy gap.

    tc : float
        Critical temperature in Kelvin.

    vf : float
        Fermi velocity in m/s.

    london0 : float
        London penetration depth at zero temperature in m.

    axis : 'string' (optional)
        Acceptable values are ``'temperature' or 'frequency' or 'mean free
        path'``. Specifies what the units of the ``input_vector`` parameter are.
        Default is ``'temperature'``.

    Keyword Arguments
    -----------------
    mfp, tr, fr : float (required)
        If ``axis == 'temperature'`` must specify reduced frequency value fr =
        h*f/delta0 and mean-free-path in meters. If ``axis == 'frequency'`` must specify a reduced
        temperature value tr = T/Tc and mean-free-path in meters. If ``axis == 'mean free path'`` must
        specify both fr and tr.

    bcs : float (optional)
        The constant that gives the zero-temperature superconducting energy gap
        delta0 according to the equation delta0 = bcs*kB*Tc, where kB is
        Boltzmann's constant and Tc is the superconducting critical temperature.
        Default value is the Bardeen-Cooper-Schrieffer value of 1.76.

    gap : python function (optional)
        Python function that gives a value for the reduced superconducting
        energy gap deltar(T) = delta(T)/delta0, where delta is the
        superconducting energy gap and delta0 = delta(T=0). Function signature
        is float(float) with return value between zero and one. Default is
        tabulated values from Muhlschlegel (1959) via the deltar_bcs function.

    output_depths : bool (optional)
        Sets the output units. False returns complex impedance in Ohms. True
        converts the complex impedance to a skin-depth (real part) and a
        superconducting penetration depth (imaginary part), both in meters.
        Default is False.

    boundary : string (optional)
        Options are ``'diffuse'``/``'d'`` or ``'specular'``/``'s'``. Determines
        whether the impedance calculation assumes diffuse or specular scattering
        at the boundaries of the superconductor. Default is ``'diffuse'``.

    Returns
    -------
    zs : numpy array
        The complex impedance calculated at each value of input_vector.

    Note
    ----
    fr = 0 may return slightly wrong number for specular boundary conditions.
    Not sure why."""

    allowed_axes = ['temperature',
                    't',
                    'frequency',
                    'f',
                    'mean free path',
                    'mfp',
                    'l']

    assert axis in allowed_axes, "Invalid axis."

    #This is the value that specifies delta0 = bcs*kB*Tc
    bcs = kwargs.pop('bcs', 1.76)
    delta0 = bcs*sc.k*tc/sc.e

    #Kwargs to pass to the cmplx_impedance function
    zs_kwargs = {}

    #Allow for passing in a custom gap function
    gap = kwargs.pop('gap', None)
    if gap is not None:
        zs_kwargs['gap'] = gap

    verbose = kwargs.pop('verbose', False)
    zs_kwargs['verbose'] = verbose
    #Optionally can output penetration/skin depth in meters instead of Ohms
    output_depths = kwargs.pop('output_depths', False)
    zs_kwargs['output_depths'] = output_depths

    #See if specular reflection is requested
    boundary = kwargs.pop('boundary', 'diffuse')
    assert boundary in ['diffuse', 'd', 'specular', 's'], "Invalid boundary type."
    zs_kwargs['boundary'] = boundary

    if axis in ['mfp', 'mean free path', 'l']:
        assert 'fr' in kwargs, "Must supply reduced frequency"
        fr = kwargs['fr']

        assert 'tr' in kwargs, "Must supply reduced temperature"
        tr = kwargs['tr']

        mfps = input_vector

        zs = []

        for mfp in mfps:
            #Convert physical data to params
            params_dict = init_from_physical_data(tc, vf, london0, mfp, bcs)

            #Add the parameters from physical data into the kwargs dict
            zs_kwargs.update(params_dict)

            #Calculate the next impedance
            zs.append(cmplx_impedance(tr, fr, tc, **zs_kwargs))

        #Convert to numpy array
        zs = np.asarray(zs)



    else:
        #Convert physical data to params
        params_dict = init_from_physical_data(tc, vf, london0, mfp, bcs)

        #Add the parameters from physical data into the kwargs dict
        zs_kwargs.update(params_dict)

        if axis in ['temperature', 't']:
            assert 'fr' in kwargs, "Must supply reduced frequency"
            fr = kwargs['fr']

            trs = input_vector
            zs = np.array([cmplx_impedance(tr, fr, tc, **zs_kwargs) for tr in trs])

        if axis in ['frequency', 'f']:
            assert 'tr' in kwargs, "Must supply reduced temperature"
            tr = kwargs['tr']

            frs = np.asarray(input_vector)
            zs = np.array([cmplx_impedance(tr, fr, tc, **zs_kwargs) for fr in frs])

    return zs


def cmplx_impedance(tr, fr, tc, x0, x1, vf, **kwargs):
    r"""Calculate the complex surface impedance (Z) of a superconductor at a
    given temperature and frequency.

    Parameters
    ----------
    tr : float
        Reduced temperature tr = T/Tc, where T is temperature and Tc is the
        superconducting critical temperature.

    fr : float
        Reduced frequency fr = h*f/delta0 where h is Planck's constant, f is the
        frequency in Hz, and delta0 is the zero-temperature superconducting
        energy gap.

    x0 : float
        Mean free path (mfp) divided by the BCS coherence length (ksi0). ksi0 =
        hbar*vf/(pi*delta0), where hbar is Planck's constant divided by 2*pi, vf
        is the Fermi velocity, and delta0 is the zero-temprature superconducting
        energy gap.

    x1 : float
        Mean free path (mfp) divided by the zero-temperature London penetration depth
        (london0).

    vf : float
        Fermi velocity in m/s.


    Keyword Arguments
    -----------------
    bcs : float (optional)
        The constant that gives the zero-temperature superconducting energy gap
        delta0 according to the equation delta0 = bcs*kB*Tc, where kB is
        Boltzmann's constant and Tc is the superconducting critical temperature.
        Default value is the Bardeen-Cooper-Schrieffer value of 1.76.

    boundary : string (optional)
        Options are ``'diffuse'``/``'d'`` or ``'specular'``/``'s'``. Determines
        whether the impedance calculation assumes diffuse or specular scattering
        at the boundaries of the superconductor. Default is ``'diffuse'``.

    gap : python function (optional)
        Python function that gives a value for the reduced superconducting
        energy gap deltar(T) = delta(T)/delta0, where delta is the
        superconducting energy gap and delta0 = delta(T=0). Function signature
        is float(float) with return value between zero and one. Default is
        tabulated values from Muhlschlegel (1959) via the deltar_bcs function.

    output_depths : bool (optional)
        Sets the output units. False returns complex impedance in Ohms. True
        converts the complex impedance to a skin-depth (real part) and a
        superconducting penetration depth (imaginary part), both in meters.
        Default is False.

    verbose : bool (optional)
        Whether to print out some debugging information. Very much a work in
        progress. Default is False.

    Returns
    -------
    Z: The complex surface impedance in Ohms.

    Note
    ----
    fr = 0 may return slightly wrong number for specular boundary conditions.
    Not sure why."""

    #Optionally can output penetration/skin depth in meters instead of Ohms
    output_depths = kwargs.pop('output_depths', False)

    verbose = kwargs.pop('verbose', False)
    boundary = kwargs.pop('boundary', 'diffuse')

    gap = kwargs.pop('gap', deltar_bcs)

    dr = gap(tr)
    bcs = kwargs.pop('bcs', 1.76)
    delta0 = bcs*sc.k*tc/sc.e

    #Calculate the prefactor. Mostly this doesn't matter since we care about ratios.
    if output_depths:
        prefactor = x0*vf*sc.hbar/(delta0*sc.e)
    else:
        prefactor = fr*x0*sc.mu_0*vf



    k = lambda x : cmplx_kernel(tr, fr, x, x0, x1, dr, bcs, verbose=verbose)

    if (boundary == 'diffuse') or (boundary == 'd'):
        reInvZint = lambda x : cm.log(1+k(x)/x**2).real
        imInvZint = lambda x : cm.log(1+k(x)/x**2).imag

        # reInvZ, reInvZerr, _, _ = do_integral(reInvZint, 0, np.inf, verbose=verbose)
        # imInvZ, imInvZerr, _, _ = do_integral(imInvZint, 0, np.inf, verbose=verbose)

        reInvZ, reInvZerr = sint.quad(reInvZint, 0, np.inf)
        imInvZ, imInvZerr = sint.quad(imInvZint, 0, np.inf)


        invZ = reInvZ + 1j*imInvZ

        Z = 1.0/invZ

    elif (boundary == 'specular') or (boundary == 's'):
        reZint = lambda x : (1/(x**2+k(x))).real
        imZint = lambda x : (1/(x**2+k(x))).imag

        # reZ, reZerr, _, _ = do_integral(reZint, -np.inf, np.inf, verbose=verbose)
        # imZ, imZerr, _, _ = do_integral(imZint, -np.inf, np.inf, verbose=verbose)

        reZ, reZerr = sint.quad(reZint, -np.inf, np.inf)
        imZ, imZerr = sint.quad(imZint, -np.inf, np.inf)


        Z = (reZ + 1j*imZ)/np.pi**2

    return 1.0*prefactor*1j*Z
