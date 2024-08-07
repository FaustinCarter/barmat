# coding=utf-8
import warnings

import numba
import scipy.integrate as sint
import scipy.constants as sc
from scipy import LowLevelCallable
import ctypes as ct
import pprint

def get_delta0(tc, bcs=1.76):
    "Calculate delta0 from Tc and a custom value of the BCS constant."
    return bcs*sc.k*tc/sc.e

def get_tc(delta0, bcs=1.76):
    "Calculate Tc from delta0 and a custom value of the BCS constant."
    return delta0*sc.e/(bcs*sc.k)

def get_bcs(tc, delta0):
    "Calculate the BCS constant given a Tc and a delta0."
    return delta0*sc.e/(sc.k*tc)

def get_ksi0(vf, delta0):
    "Calculate the BCS coherence length from the Fermi velocity and delta0."
    return vf*sc.hbar/(sc.pi*delta0*sc.e)

def get_vf(ksi0, delta0):
    "Calculate the Fermi velocity from the BCS coherence length and delta0."
    return ksi0*sc.pi*delta0*sc.e/sc.hbar

def init_from_physical_data(tc, vf, london0, mfp, bcs=1.76, ksi0=None, delta0=None, verbose=False):
    r"""Calculate some unitless quantities needed for the computation from
    physical data.

    Parameters
    ----------
    tc : float
        Critical temperature in Kelvin.

    vf : float
        Fermi velocity in m/s.

    mfp : float
        Mean free path in m.

    london0 : float
        London penetration depth at zero temperature in m.

    bcs : float (optional)
        The constant that gives the zero-temperature superconducting energy gap
        delta0 according to the equation delta0 = bcs*kB*Tc, where kB is
        Boltzmann's constant and Tc is the superconducting critical temperature.
        Default value is the Bardeen-Cooper-Schrieffer value of 1.76.

    ksi0 : float (optional)
        The BCS coherence length in m. ksi0 = hbar*vf/(pi*delta0), where hbar is
        Planck's constant divided by 2*pi, vf is the Fermi velocity, and delta0
        is the zero-temprature superconducting energy gap.

    delta0 : float (optional)
        The zero-temperature superconducting energy gap in electron volts.
        delta0 = bcs*kB*Tc, where kB is Boltzmann's constant and Tc is the
        superconducting critical temperature. Default value is the
        Bardeen-Cooper-Schrieffer value of 1.76.

    Keyword Arguments
    -----------------
    verbose : int
        Whether to print out some debugging information. Very much a work in
        progress. Default is 0.

        * 0: Don't print out any debugging information
        * 1: Print out minimal debugging information
        * 2: Print out full output from quad routines

    Returns
    -------
    output_dict : dict
        output_dict contains the following keys: bcs, vf, x0, x1. These are all
        necessary inputs to impedance.cmplx_impedance."""

    if delta0 is None:
        delta0 = bcs*sc.k*tc/sc.e

    if ksi0 is None:
        ksi0 = vf*sc.hbar/(sc.pi*delta0*sc.e)
        ksi0_calc = None
    else:
        ksi0_calc = vf*sc.hbar/(sc.pi*delta0*sc.e)

    #Reduced lengths
    xk = ksi0/london0
    xm = mfp/london0

    if verbose > 1:
        if ksi0_calc is not None:
            print("calculated ksi0 = " + str(ksi0_calc*1e9)+" nm")
            print("supplied ksi0 = " + str(ksi0*1e9)+" nm")
        else:
            print("calculated ksi0 = " + str(ksi0*1e9)+" nm")

        print("calculated delta0 = " + str(delta0*1e6) + " ueV")
        print("xk = ksi0/london0 = " + str(xk))
        print("xm = mfp/london0 = " + str(xm))
        print("xk/xm = ksi0/mfp = " + str(xk/xm))

    output_dict = {'bcs':bcs,
                    'vf':vf,
                    'xk':xk,
                    'xm':xm}

    return output_dict

def do_integral(int_fun, a, b, iargs=None, **kwargs):
    """Wrapper around scipy.integrate.quad to handle error reporting"""

    #Parse the kwargs
    func_name=kwargs.pop('func_name', None)
    extra_info=kwargs.pop('extra_info', None)
    iargs_format_strings = kwargs.pop('iargs_format_strings',None)
    verbose = kwargs.pop('verbose', 0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=sint.IntegrationWarning)

        #First run the integral with full_output = 0, which will generate a warning if there's an issue
        try:
            if iargs is not None:
                result, result_err = sint.quad(int_fun, a, b, iargs)
            else:
                result, result_err = sint.quad(int_fun, a, b)

        #If there is an issue, rerun it with full_output = 1, which will suppress the warning,
        #but will generate some useful debugging information
        except sint.IntegrationWarning:
            if iargs is not None:
                result, result_err, output_dict, error_msg = sint.quad(int_fun, a, b, iargs, full_output=1)
            else:
                result, result_err, output_dict, error_msg = sint.quad(int_fun, a, b, full_output=1)

            if func_name is None:
                try:
                    func_name = int_fun.__name__
                except AttributeError:
                    func_name = 'Unknown name'

            if verbose > 0:
                print(func_name)
                if extra_info is not None:
                    print(extra_info)
                if iargs is not None:
                    print("iargs = ", iargs)
                print(result, result_err)
                print(error_msg)
                if verbose > 1:
                    pprint.pprint(output_dict)

    return (result, result_err)

def wrap_for_numba(func):
    """Uses numba to create a C-callback out of a function.

    Parameters
    ----------
    func : python function
        Signature float(float[:])

    Returns
    -------
    new_cfunc : numba.cfunc object
        Signature float(int, pointer-to-array).

    Note
    ----
    The ``__name__`` and ``argtypes`` attributes of new_cfunc are modified here.
    """


    def c_func(n, a):
        """Simple wrapper to convert (int, pointer-to-array) args to (list) args.

        Parameters
        ----------
        n : C-language integer

        a : C-language pointer to array of type double.

        Returns
        -------
        func : python function
            Function signature is float(float[:])"""

        #unpack the pointer into an array
        args = numba.carray(a,n)

        return func(args)

    #Function signature required by numba
    #arguments are integer denoting length of array and pointer to array
    c_sig = numba.types.double(numba.types.intc, numba.types.CPointer(numba.types.double))

    #Use numba to create a c-callback
    new_cfunc = numba.cfunc(c_sig)(c_func)

    return LowLevelCallable(new_cfunc.ctypes)
