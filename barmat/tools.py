from __future__ import division
import numba
import scipty.integrate as sint
import ctypes as ct
import scipy.constants as sc

def get_delta0(tc, bcs=1.76):
    return bcs*sc.k*tc/sc.e

def get_tc(delta0, bcs=1.76):
    return delta0*sc.e/(bcs*sc.k)

def get_bcs(tc, delta0):
    return delta0*sc.e/(sc.k*tc)

def get_ksi0(vf, delta0):
    return vf*sc.hbar/(sc.pi*delta0*sc.e)

def get_vf(ksi0, delta0):
    return ksi0*sc.pi*delta0*sc.e/sc.hbar

def init_from_physical_data(tc, vf, london0, mfp, bcs=1.76, ksi0=None, delta0=None, verbose=False):
    #Derived quantities
    if delta0 is None:
        delta0 = bcs*sc.k*tc/sc.e

    if ksi0 is None:
        ksi0 = vf*sc.hbar/(sc.pi*delta0*sc.e)
        ksi0_calc = None
    else:
        ksi0_calc = vf*sc.hbar/(sc.pi*delta0*sc.e)

    x0 = mfp/ksi0
    x1 = mfp/london0

    if verbose:
        if ksi0_calc is not None:
            print "calculated ksi0 = " + str(ksi0_calc*1e9)+" nm"
            print "supplied ksi0 = " + str(ksi0*1e9)+" nm"
        else:
            print "calculated ksi0 = " + str(ksi0*1e9)+" nm"

        print "calculated delta0 = " + str(delta0*1e6) + " ueV"
        print "x0 = mfp/ksi0 = " + str(x0)
        print "x1 = mfp/london0 = " + str(x1)
        print "x1/x0 = ksi0/london0 = " + str(x1/x0)

    return {'bcs':bcs,
            'vf':vf,
            'x0':x0,
            'x1':x1}

def do_integral(int_fun, a, b, iargs=None, verbose=False):
    """Wrapper around scipy.integrate.quad to handle error reporting"""
    if verbose:
        full_output = 1
    else:
        full_output = 0

    if iargs is not None:
        int_tup = sint.quad(int_fun, a, b, iargs, full_output = full_output)
    else:
        int_tup = sint.quad(int_fun, a, b, full_output = full_output)

    if len(int_tup) == 3:
        int_tup = tuple(list(int_tup)+[''])
    if len(int_tup) == 2:
        int_tup = tuple(list(int_tup)+[{}]+[''])
    else:
        result, error, id_dict, error_msg = int_tup
        if verbose:
            try:
                funcName = int_fun.__name__
            except AttributeError:
                funcName = 'Unknown name'

            if iargs is not None:
                print funcName+": x=%s, xop=%s, tr=%s, fr=%s, dr=%s" % iargs
            else:
                print funcName
            print "Result: %s, error: %s" % (result, error)
            print error_msg

    return int_tup

def wrap_for_numba(func):
    """Uses numba to create a C-callback out of a function.
    Also includes a hack to fix a bug in the way SciPy parses input args in quad.
    This will probably break in future versions of SciPy."""


    def c_func(n, a):
        """Simple wrapper to convert (int, pointer-to-array) args to (list) args."""

        #unpack the pointer into an array
        args = numba.carray(a,n)

        return func(args)

    #Function signature required by numba
    #arguments are integer denoting length of array and pointer to array
    c_sig = numba.types.double(numba.types.intc, numba.types.CPointer(numba.types.double))

    #Use numba to create a c-callback
    new_cfunc = numba.cfunc(c_sig)(c_func)

    #This is a hack to address a bug in scipy.integrate.quad
    new_cfunc.ctypes.argtypes = (ct.c_int, ct.c_double)

    #This is because for some ungodly reason numba doesn't include this...
    new_cfunc.ctypes.__name__ = new_cfunc.__name__

    return new_cfunc
