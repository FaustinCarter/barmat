# coding=utf-8
import cmath as cm
import math as ma

import numpy as np
import scipy.constants as sc
import scipy.integrate as sint

from .tools import do_integral, init_from_physical_data
from .kernel import cmplx_kernel
from .gap_functions import deltar_bcs, deltar_cos

from multiprocessing import cpu_count


#Check to seeif multicore will work
num_cores = cpu_count()

if num_cores > 1:
    try:
        from joblib import Parallel, delayed
        HAS_JOBLIB = True
    except:
        HAS_JOBLIB = False
else:
    HAS_JOBLIB = False

__all__ = [ 'get_Zvec',
            'cmplx_impedance']

def get_Zvec(input_vector, tc, vf, london0, axis='temperature', **kwargs):
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

    gap : python function or string (optional)
        Python function that gives a value for the reduced superconducting
        energy gap deltar(T) = delta(T)/delta0, where delta is the
        superconducting energy gap and delta0 = delta(T=0). Function signature
        is float(float) with return value between zero and one. Default is
        tabulated values from Muhlschlegel (1959) via the deltar_bcs function.
        Optionally, one may pass the string ``'cos'`` to use the built-in cosine
        approximation of the gap.


    output_depths : bool (optional)
        Sets the output units. False returns complex impedance in Ohms. True
        converts the complex impedance to a skin-depth (real part) and a
        superconducting penetration depth (imaginary part), both in meters.
        Default is False.

    boundary : string (optional)
        Options are ``'diffuse'``/``'d'`` or ``'specular'``/``'s'``. Determines
        whether the impedance calculation assumes diffuse or specular scattering
        at the boundaries of the superconductor. Default is ``'diffuse'``.

    verbose : int
        Whether to print out some debugging information. Very much a work in
        progress. Default is 0.

        * 0: Don't print out any debugging information
        * 1: Print out minimal debugging information
        * 2: Print out full output from quad routines

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
        if gap == 'cos':
            zs_kwargs['gap'] = deltar_cos
        else:
            zs_kwargs['gap'] = gap

    verbose = kwargs.pop('verbose', 0)
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

        #Do some basic parallelization (speeds up by factor of two-ish)
        if HAS_JOBLIB:
            def get_zVec_inner(mfp):
                #Convert physical data to params
                params_dict = init_from_physical_data(tc, vf, london0, mfp, bcs)

                # Add the parameters from physical data into the kwargs dict
                zs_kwargs.update(params_dict)

                #Calculate the next impedance
                return cmplx_impedance(tr, fr, tc, **zs_kwargs)

            zs = Parallel(n_jobs = num_cores)(delayed(get_zVec_inner)(mfp) for mfp in mfps)

        else:

            zs = []

            for mfp in mfps:
                #Convert physical data to params
                params_dict = init_from_physical_data(tc, vf, london0, mfp, bcs)

                # Add the parameters from physical data into the kwargs dict
                zs_kwargs.update(params_dict)

                #Calculate the next impedance
                zs.append(cmplx_impedance(tr, fr, tc, **zs_kwargs))

        #Convert to numpy array
        zs = np.asarray(zs)



    else:
        assert 'mfp' in kwargs, "Must supply mean free path"
        mfp = kwargs['mfp']
        #Convert physical data to params
        params_dict = init_from_physical_data(tc, vf, london0, mfp, bcs)

        # Add the parameters from physical data into the kwargs dict
        zs_kwargs.update(params_dict)

        if axis in ['temperature', 't']:
            assert 'fr' in kwargs, "Must supply reduced frequency"
            fr = kwargs['fr']

            trs = input_vector

            if HAS_JOBLIB:
                zs = Parallel(n_jobs = num_cores)(delayed(cmplx_impedance)(tr, fr, tc, **zs_kwargs) for tr in trs)
            else:
                zs = np.array([cmplx_impedance(tr, fr, tc, **zs_kwargs) for tr in trs])

        if axis in ['frequency', 'f']:
            assert 'tr' in kwargs, "Must supply reduced temperature"
            tr = kwargs['tr']

            frs = np.asarray(input_vector)

            if HAS_JOBLIB:
                zs = Parallel(n_jobs = num_cores)(delayed(cmplx_impedance)(tr, fr, tc, **zs_kwargs) for fr in frs)
            else:
                zs = np.array([cmplx_impedance(tr, fr, tc, **zs_kwargs) for fr in frs])

    return zs


def cmplx_impedance(tr, fr, tc, xk, xm, vf, **kwargs):
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

    xk : float
        BCS coherence length (ksi0) divided by mean free path (mfp). ksi0 =
        hbar*vf/(pi*delta0), where hbar is Planck's constant divided by 2*pi, vf
        is the Fermi velocity, and delta0 is the zero-temprature superconducting
        energy gap.

    xm : float
        Mean free path (mfp) divided by the zero-temperature London penetration
        depth (london0).

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

    gap : python function or string (optional)
        Python function that gives a value for the reduced superconducting
        energy gap deltar(T) = delta(T)/delta0, where delta is the
        superconducting energy gap and delta0 = delta(T=0). Function signature
        is float(float) with return value between zero and one. Default is
        tabulated values from Muhlschlegel (1959) via the deltar_bcs function.
        Optionally, one may pass the string ``'cos'`` to use the built-in cosine
        approximation of the gap.

    output_depths : bool (optional)
        Sets the output units. False returns complex impedance in Ohms. True
        converts the complex impedance to a skin-depth (real part) and a
        superconducting penetration depth (imaginary part), both in meters.
        Default is False.

    verbose : int
        Whether to print out some debugging information. Very much a work in
        progress. Default is 0.

        * 0: Don't print out any debugging information
        * 1: Print out minimal debugging information
        * 2: Print out full output from quad routines

    Returns
    -------
    Z: The complex surface impedance in Ohms.

    Note
    ----
    fr = 0 may return slightly wrong number for specular boundary conditions.
    Not sure why."""

    #Optionally can output penetration/skin depth in meters instead of Ohms
    output_depths = kwargs.pop('output_depths', False)

    units = 'Ohms'
    if output_depths:
        units = 'meters'

    verbose = kwargs.pop('verbose', 0)
    boundary = kwargs.pop('boundary', 'diffuse')

    gap = kwargs.pop('gap', None)
    if gap is None:
        gap = deltar_bcs
    elif gap == 'cos':
        gap = deltar_cos
    else:
        #TODO: check user-supplied function for correct sig
        pass

    dr = gap(tr)
    bcs = kwargs.pop('bcs', 1.76)
    delta0 = bcs*sc.k*tc/sc.e

    #Calculate the prefactor. Mostly this doesn't matter since we care about ratios.
    if output_depths:
        #Units here are meters
        prefactor = vf*sc.hbar/(xk*delta0*sc.e)
    else:
        #Units here are Ohms
        prefactor = fr*sc.mu_0*vf/xk


    k = lambda x : cmplx_kernel(tr, fr, x, xk, xm, dr, bcs, verbose=verbose)
    # reK = lambda x : cmplx_kernel(tr, fr, x, x0, x1, dr, bcs, verbose=verbose).real
    # imK = lambda x : cmplx_kernel(tr, fr, x, x0, x1, dr, bcs, verbose=verbose).imag

    #For passing useful debugging info
    param_vals_string = "tr=%s, fr=%s, xk=%s, xm=%s, dr=%s, bcs=%s" % (tr, fr, xk, xm, dr, bcs)

    if (boundary == 'diffuse') or (boundary == 'd'):
        #Now separate the integrand into the real and imaginary parts, otherwise
        #ma.log chokes on trying to figure out the right branch cut
        #
        #ln(1+K/x**2) = 0.5*ln((x**2+ReK)**2+ImK**2)-2ln(x) + i*atan(ImK/(x**2+ReK))

        # reInvZint_a = lambda x : 0.5*ma.log((x**2+reK(x))**2+imK(x)**2)-2*ma.log(x)
        # reInvZint_b = lambda x : 0.5*ma.log((x**2+reK(x))**2+imK(x)**2)
        #
        # imInvZint = lambda x : ma.atan2(imK(x),(x**2+reK(x))) #Use atan2 because phase matters
        #
        # reInvZ_a, reInvZerr_a = do_integral(reInvZint_a, 1, np.inf,
        #                                     func_name = "Diffuse:reInvZint_a",
        #                                     extra_info=param_vals_string,
        #                                     verbose=verbose)
        #
        # reInvZ_b, reInvZerr_b = do_integral(reInvZint_b, 0, 1,
        #                                     func_name = "Diffuse:reInvZint_b",
        #                                     extra_info=param_vals_string,
        #                                     verbose=verbose)
        #
        # reInvZ = reInvZ_a + reInvZ_b + 2
        # reInvZerr = reInvZerr_a + reInvZerr_b
        #
        # imInvZ, imInvZerr = do_integral(imInvZint, 0, np.inf,
        #                                 func_name = "Diffuse:imInvZint",
        #                                 extra_info=param_vals_string,
        #                                 verbose=verbose)

        reInvZint = lambda x : cm.log(1+k(x)/x**2).real
        imInvZint = lambda x : cm.log(1+k(x)/x**2).imag

        reInvZ, reInvZerr = do_integral(reInvZint, 0, np.inf,
                                        func_name = "Diffuse:reInvZint",
                                        extra_info=param_vals_string,
                                        verbose=verbose)


        imInvZ, imInvZerr = do_integral(imInvZint, 0, np.inf,
                                        func_name = "Diffuse:imInvZint",
                                        extra_info=param_vals_string,
                                        verbose=verbose)



        invZ = reInvZ + 1j*imInvZ

        #Do an error calc
        invZ2 = reInvZ**2 + imInvZ**2

        if invZ2 == 0:
            Zerr = 0
        else:
            reZerr = 1/invZ2**2 * ma.sqrt(((invZ2-2*reInvZ)*reInvZerr)**2 + (2*imInvZ*imInvZerr)**2)
            imZerr = 1/invZ2**2 * ma.sqrt(((2*imInvZ-invZ2)*imInvZerr)**2 + (2*reInvZ*reInvZerr)**2)

            Zerr = reZerr+1j*imZerr

        Z = 1.0/invZ

    elif (boundary == 'specular') or (boundary == 's'):
        reZint = lambda x : (1/(x**2+k(x))).real
        imZint = lambda x : (1/(x**2+k(x))).imag

        reZ, reZerr = do_integral(reZint, -np.inf, np.inf,
                                    func_name="Specular:reZint",
                                    extra_info=param_vals_string,
                                    verbose=verbose)

        imZ, imZerr = do_integral(imZint, -np.inf, np.inf,
                                    func_name="Specular:imZint",
                                    extra_info=param_vals_string,
                                    verbose=verbose)

        Zerr = (reZerr+1j*imZerr)/np.pi**2

        Z = (reZ + 1j*imZ)/np.pi**2

    Z *= prefactor*1j
    Zerr *= prefactor*1j

    if verbose > 0:
        print("Z = %s %s" % (Z, units))
        if (Z.real == 0) and (Z.imag == 0):
            print("No error, identically zero")
        else:
            print("fractional error in real part:", Zerr.real/abs(Z.real))
            print("fractional error in imag part:", Zerr.imag/abs(Z.imag))
        print("\n")

    return Z
