from __future__ import division
import numpy as np
import cmath as cm
from tools import do_integral, init_from_physical_data
from kernel import cmplx_kernel
from gap_functions import deltar_bcs

def get_Z(input_vector, tc, vf, mfp, london0, axis='temperature', **kwargs):
    """Return Z along any axis requested."""

    allowed_axes = ['temperature',
                    'frequency']

    assert axis in allowed_axes, "Invalid axis."

    #This is the value that specifies delta0 = bcs*kB*Tc
    bcs = kwargs.pop('bcs', 1.76)

    #Initialize values
    #zs_kwargs contains x0, x1, vf, bcs
    zs_kwargs = init_from_physical_data(tc, vf, london0, mfp, bcs)

    #Allow for passing in a custom gap function
    gap = kwargs.pop('gap', None)
    if gap is not None:
        zs_kwargs['gap'] = gap

    #Optionally can output penetration/skin depth in meters instead of Ohms
    output_depths = kwargs.pop('output_depths', False)

    if axis == 'temperature':
        assert 'fr' in kwargs, "Must supply reduced frequency"
        fr = kwargs['fr']

        trs = input_vector
        zs = np.array([cmplx_impedance(tr, fr, **zs_kwargs) for tr in trs])

        if output_depths:
            zs /= (sc.mu_0*fr*delta0*sc.e/sc.hbar)

    if axis == 'frequency':
        assert 'tr' in kwargs, "Must supply reduced temperature"
        tr = kwargs['tr']

        frs = np.asarray(input_vector)
        zs = np.array([cmplx_impedance(tr, fr, **zs_kwargs) for fr in frs])

        if output_depths:
            zs /= (sc.mu_0*frs*delta0*sc.e/sc.hbar)

    return zs


def cmplx_impedance(tr, fr, x0, x1, vf, bcs, boundary='diffuse', gap = deltar_bcs, verbose=False):
    """Calculate the complex surface impedance of a superconductor.

    Arguments
    ---------

    x0: mean free path / BCS coherence length
    x1: mean free path / London penetration depth at T=0
    tr: temperature / critical temperature
    fr: planck's const * frequency / BCS energy gap at T=0
    dr: BCS energy gap at T = temperature / BCS energy gap at T=0
    vf: Fermi velocity

    Returns
    -------
    Z: The complex impedance."""

    #Calculate the prefactor. Mostly this doesn't matter since we care about ratios.
    prefactor = fr*x0*sc.mu_0*vf

    dr = gap(tr)

    k = lambda x : cmplx_kernel(tr, fr, x, x0, x1, dr, bcs, verbose=False)

    if (boundary == 'diffuse') or (boundary == 'd'):
        reInvZint = lambda x : cm.log(1+k(x)/x**2).real
        imInvZint = lambda x : cm.log(1+k(x)/x**2).imag

        reInvZ, reInvZerr, _, _ = do_integral(reInvZint, 0, np.inf, verbose=verbose)
        imInvZ, imInvZerr, _, _ = do_integral(imInvZint, 0, np.inf, verbose=verbose)

        invZ = reInvZ + 1j*imInvZ

        Z = 1.0/invZ

    elif (boundary == 'specular') or (boundary == 's'):
        reZint = lambda x : (1/(x**2+k(x))).real
        imZint = lambda x : (1/(x**2+k(x))).imag

        reZ, reZerr, _, _ = do_integral(reZint, -np.inf, np.inf, verbose=verbose)
        imZ, imZerr, _, _ = do_integral(imZint, -np.inf, np.inf, verbose=verbose)

        Z = (reZ + 1j*imZ)/np.pi**2

    return 1.0*prefactor*1j*Z
