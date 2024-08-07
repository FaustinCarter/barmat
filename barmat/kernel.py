# coding=utf-8
import math as ma

import numba
import numpy as np
import scipy.integrate as sint

from .volume_integrals import intR, intS
from .tools import wrap_for_numba, do_integral


@numba.jit("float64(float64[:])", nopython=True)
def reKlint1_a1(args):
    r"""Energy integral of real part of K, limits -dr to 0."""
    #u-substitution, u**2 = en+dr
    #from -dr to 0, only fires if f >= 2*dr
    #ulims are: 0, sqrt(dr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = u**2-dr
    e1 = u*ma.sqrt(2*dr-u**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    a1 = a0*e1
    a2 = a0*e2

    return 2*ma.tanh(0.5*bcs*(en+fr)/tr)*(((en**2+dr**2+en*fr)/(ma.sqrt(2*dr-u**2)*e2))*intR(a2, a1+b0, x)+u*intS(a2, a1+b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def reKlint1_a2(args):
    r"""Energy integral of real part of K, limits dr-fr to dr-0.5*fr."""
    #u-substitution, u**2 = (en+fr)-dr
    #from dr-fr to dr-0.5*fr
    #ulims are: 0, sqrt(0.5*fr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = u**2-fr+dr
    e1 = ma.sqrt(abs(en**2-dr**2))
    e2 = u*ma.sqrt(u**2+2*dr)
    a1 = a0*e1
    a2 = a0*e2

    return 2*ma.tanh(0.5*bcs*(en+fr)/tr)*(((en**2+dr**2+en*fr)/(e1*ma.sqrt(u**2+2*dr)))*intR(a2, a1+b0, x)+u*intS(a2, a1+b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def reKlint1_b(args):
    r"""Energy integral of real part of K, limits are max(0, dr-0.5*fr) to dr."""
    #u-substitution, u**2 = dr-en
    #from 0 to dr OR dr-0.5*fr to dr, depending on whether fr > 2*dr
    #ulims are: 0 to sqrt(dr) OR 0, sqrt(0.5*fr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = dr-u**2
    e1 = u*ma.sqrt(2*dr-u**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    a1 = a0*e1
    a2 = a0*e2

    return 2*ma.tanh(0.5*bcs*(en+fr)/tr)*(((en**2+dr**2+en*fr)/(ma.sqrt(2*dr-u**2)*e2))*intR(a2, a1+b0, x)+u*intS(a2, a1+b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def reKlint2_a(args):
    r"""Energy integral of real part of K, limits are dr-fr to -fr/2."""
    #u-substitution, u**2 = en+fr-dr
    #from dr-fr to -fr/2, only fires if fr > 2*dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = u**2-fr+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(e1*ma.sqrt(u**2+2*dr))

    return ma.tanh(0.5*bcs*(en+fr)/tr)*((gfun+u)*intS(a0*(e2-e1), b0, x)
                                                       -(gfun-u)*intS(a0*(e2+e1), b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def reKlint2_b(args):
    """Energy integral of real part of K, limits are -fr/2 to -dr."""
    #u-substitution, u**2 = -en-dr
    #from -fr/2 to -dr, only fires if fr > 2*dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = -u**2-dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)

    return ma.tanh(0.5*bcs*(en+fr)/tr)*((gfun+u)*intS(a0*(e2-e1), b0, x)
                                                       -(gfun-u)*intS(a0*(e2+e1), b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def reKlint3(args):
    r"""Energy integral of real part of K, limits are dr to infinity."""
    #u-substitution, u**2 = en-dr
    #from dr to infinity
    #ulims are: 0, 31.6 <-- this is big enough, rounding errors happen at big numbers

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    #Fixes the divide by zero error nicely
    #The right way to do this is to recognize that intS goes to zero like 'a' as 'a'->0.
    #This leading behavior cancels out the part that makes the integrand blow up at the lower limit.
    if (fr == 0) and (u**2/dr < 0.01):
        en = dr*(1+1e-2)
    else:
        en = u**2+dr

    e1 = ma.sqrt((en+dr)*(en-dr))
    e2 = ma.sqrt(((en+fr)+dr)*((en+fr)-dr))
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)
    tplus = ma.tanh(0.5*bcs*en/tr)+ma.tanh(0.5*bcs*(en+fr)/tr)
    tminus = ma.tanh(0.5*bcs*fr/tr)*(1-ma.tanh(0.5*bcs*en/tr)*ma.tanh(0.5*bcs*(en+fr)/tr))

    return -tplus*(gfun-u)*intS(a0*(e2+e1), b0, x) + tminus*(gfun+u)*intS(a0*(e2-e1), b0, x)

@numba.jit("float64(float64[:])", nopython=True)
def imKlint1_a(args):
    r"""Energy integral of imaginary part of K, limits are dr-fr to -dr."""
    #from dr-fr to -dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = u**2-fr+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(e1*ma.sqrt(u**2+2*dr))

    return -ma.tanh(0.5*bcs*(en+fr)/tr)*((gfun+u)*intR(a0*(e2-e1), b0, x)+(gfun-u)*intR(a0*(e2+e1), b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def imKlint1_b(args):
    r"""Energy integral of imaginary part of K, limits are dr-fr to -dr."""
    #from dr-fr to -dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = -u**2-dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)

    return -ma.tanh(0.5*bcs*(en+fr)/tr)*((gfun+u)*intR(a0*(e2-e1), b0, x)+(gfun-u)*intR(a0*(e2+e1), b0, x))

@numba.jit("float64(float64[:])", nopython=True)
def imKlint2(args):
    r"""Energy integral of imaginary part of K, limits are dr to infinity."""
    #u-substitution, u**2 = en-dr
    #from dr to infinity
    #ulims are 0, infinity

    u = args[0]
    x = args[1]
    a0 = args[2]
    b0 = args[3]
    tr = args[4]
    fr = args[5]
    dr = args[6]
    bcs = args[7]

    en = u**2+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)
    tminus = ma.tanh(0.5*bcs*fr/tr)*(1-ma.tanh(0.5*bcs*en/tr)*ma.tanh(0.5*bcs*(en+fr)/tr))

    return tminus*((gfun+u)*intR(a0*(e2-e1), b0, x)+(gfun-u)*intR(a0*(e2+e1), b0, x))


#Make C-language callbacks from each of the integrand functions for scipy quad
#Signature for each is float64(int, Pointer-to-float64-array)
c_reKlint1_a1 = wrap_for_numba(reKlint1_a1)
c_reKlint1_a2 = wrap_for_numba(reKlint1_a2)
c_reKlint1_b = wrap_for_numba(reKlint1_b)
c_reKlint2_a = wrap_for_numba(reKlint2_a)
c_reKlint2_b = wrap_for_numba(reKlint2_b)
c_reKlint3 = wrap_for_numba(reKlint3)
c_imKlint1_a = wrap_for_numba(imKlint1_a)
c_imKlint1_a = wrap_for_numba(imKlint1_a)
c_imKlint1_b = wrap_for_numba(imKlint1_b)
c_imKlint2 = wrap_for_numba(imKlint2)


def cmplx_kernel(tr, fr, x, xk, xm, dr, bcs, verbose=0):
    r"""Calculate the Kl integral over energy. Kl = K*mfp^2 where K is the Mattis-Bardeen Kernel.

    Parameters
    ----------
    tr : float
        Reduced temperature tr = T/Tc.

    fr : float
        Reduced frequency fr = h*f/delta0

    x : float
        Reduced momentum x = q*london0 where london0 is the zero-temperature
        London penetration depth and q is the coordinate resulting from a
        fourier transform of position.

    xk : float
        BCS coherence length (ksi0) divided by mean free path (mfp). ksi0 =
        hbar*vf/(pi*delta0), where hbar is Planck's constant divided by 2*pi, vf
        is the Fermi velocity, and delta0 is the zero-temprature superconducting
        energy gap.

    xm : float
        Mean free path (mfp) divided by the zero-temperature London penetration
        depth (london0).

    dr : float
        Reduced BCS energy gap dr = delta(T)/delta(T=0) where delta is the
        superconducting energy gap.

    Keyword Arguments
    -----------------
    verbose : int
        Whether to print out some debugging information. Very much a work in
        progress. Default is 0. Currently does nothing.

    Returns
    -------
    Kl : complex float
        Kl = K*london0**2 where K is the complex Mattis-Bardeen Kernel as a
        function of x = q*london0, london0 is the zero-temperature London
        penetration depth and q is the coordinate resulting from a fourier
        transform of position.

    Note
    ----
    See Mattis and Bardeen (1958), Pöpel (1989), or Gao (2008)."""

    #a0 and b0
    a0 = 1.0/(xk*np.pi)

    b0 = 1.0/xm


    #Hackish way to make sure there are no divide by zero errors.
    if tr == 0:
        #This is a fun way to get machine epsilon
        tr = 7./3 - 4./3 -1

    #If temperature is >= Tc, then skip all the integration, and
    #use the normal Kernel
    if tr >= 1:

        #Calculate the actual number out front (1/x dependance included in R and S)
        prefactor = 3*fr*a0

        reKl = intS(fr*a0, b0, x)
        imKl = intR(fr*a0, b0, x)


    else:
        #arguments to pass the integrand functions
        iargs = (x, a0, b0, tr, fr, dr, bcs,)

        #Calculate the actual number out front (1/x dependance included in R and S)
        prefactor = 3*a0

        #Now run the integrals

        if (fr == 0):
            #All the integrals blow up here, but it's ok, because only three
            #contribute. reKl1_a2 and reKl1_b have an analytic expression,
            #and reKl3 has a limit built into it.

            #reKl1_a2 = reKl1_b = Fermi*intR*0.25*pi
            reKl1 = 2*ma.tanh(0.5*bcs*dr/tr)*intR(0, b0, x)*0.5*ma.pi*dr
            reKl1err = 0
            reKl2 = 0
            reKl2err = 0
            imKl1 = 0
            imKl1err = 0
            imKl2 = 0
            imKl2err = 0

        elif (fr/dr < 2):
            #from (dr-fr) to dr (inside the gap), limits look wierd due to u-substitution
            reKl1a, reKl1aerr = do_integral(c_reKlint1_a2, 0, ma.sqrt(0.5*fr), iargs)
            reKl1b, reKl1berr = do_integral(c_reKlint1_b, 0, ma.sqrt(0.5*fr), iargs)

            reKl1 = reKl1a + reKl1b
            reKl1err = reKl1aerr + reKl1berr

            #from dr-fr to -dr (below the gap), limits look wierd due to u-substitution
            #These ones are zero if the photons aren't big enough to break pairs
            reKl2 = 0
            reKl2err = 0
            imKl1 = 0
            imKl1err = 0
        else:
            #from -dr to dr (inside the gap), limits look wierd due to u-substitution

            reKl1a, reKl1aerr = do_integral(c_reKlint1_a1, 0, ma.sqrt(dr), iargs)
            reKl1b, reKl1berr = do_integral(c_reKlint1_b, 0, ma.sqrt(dr), iargs)

            reKl1 = reKl1a + reKl1b
            reKl1err = reKl1aerr + reKl1berr

            #from dr-fr to -dr (below the gap), limits look wierd due to u-substitution
            reKl2a, reKl2aerr = do_integral(c_reKlint2_a, 0, ma.sqrt(0.5*fr-dr), iargs)
            reKl2b, reKl2berr = do_integral(c_reKlint2_b, 0, ma.sqrt(0.5*fr-dr), iargs)

            reKl2 = reKl2a+reKl2b
            reKl2err = reKl2aerr+reKl2berr

            imKl1a, imKl1aerr = do_integral(c_imKlint1_a, 0, ma.sqrt(0.5*fr-dr), iargs)
            imKl1b, imKl1berr = do_integral(c_imKlint1_b, 0, ma.sqrt(0.5*fr-dr), iargs)

            imKl1 = imKl1a+imKl1b
            imKl1err = imKl1aerr+imKl1berr




        #from dr to infinity (above the gap), limits look wierd due to u-substitution
        #Wierd things happen numerically at infinity. 31.6 is far enough. equals about 1000 gaps.
        reKl3, reKl3err = do_integral(c_reKlint3, 0, 31.6, iargs)
        imKl2, imKl2err = do_integral(c_imKlint2, 0, np.inf, iargs)

        reKl = reKl1+reKl2+reKl3
        imKl = imKl1+imKl2


#         if verbose:
#             print('x = '+str(x)+'\n')
#             print('prefactor = '+str(prefactor)+'\n')
#             print('First reK integral = '+str(reKl1)+' +/- '+str(reKl1err) + '\n')
#             print('Second reK integral = '+str(reKl2)+' +/- '+str(reKl2err) + '\n')
#             print('Third reK integral = '+str(reKl3)+' +/- '+str(reKl3err) + '\n')
#             print('First imK integral = '+str(imKl1)+' +/- '+str(imKl1err) + '\n')
#             print('Second imK integral = '+str(imKl2)+' +/- '+str(imKl2err) + '\n')

    return prefactor*(reKl+1j*imKl)
