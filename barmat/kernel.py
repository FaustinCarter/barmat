from __future__ import division
import numba
import math as ma
import numpy as np
from volume_integrals import intR, intS
from tools import wrap_for_numba


@numba.jit("float64(float64[:])")
def reKlint1_a1(args):
    #u-substitution, u**2 = en+dr
    #from -dr to 0, only fires if f >= 2*dr
    #ulims are: 0, sqrt(dr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = u**2-dr
    e1 = u*ma.sqrt(2*dr-u**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    a1 = x0p*e1
    a2 = x0p*e2

    return -2*ma.tanh(-0.5*bcs*(en+fr)/tr)*(((en**2+dr**2+en*fr)/(ma.sqrt(2*dr-u**2)*e2))*intR(a2, a1+1, x)+u*intS(a2, a1+1, x))

@numba.jit("float64(float64[:])")
def reKlint1_a2(args):
    #u-substitution, u**2 = (en+fr)-dr
    #from dr-fr to dr-0.5*fr
    #ulims are: 0, sqrt(0.5*fr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = u**2-fr+dr
    e1 = ma.sqrt(abs(en**2-dr**2))
    e2 = u*ma.sqrt(u**2+2*dr)
    a1 = x0p*e1
    a2 = x0p*e2

    return -2*ma.tanh(-0.5*bcs*(en+fr)/tr)*(((en**2+dr**2+en*fr)/(e1*ma.sqrt(u**2+2*dr)))*intR(a2, a1+1, x)+u*intS(a2, a1+1, x))

@numba.jit("float64(float64[:])")
def reKlint1_b(args):
    #u-substitution, u**2 = dr-en
    #from 0 to dr OR dr-0.5*fr to dr, depending on whether fr > 2*dr
    #ulims are: 0 to sqrt(dr) OR 0, sqrt(0.5*fr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = dr-u**2
    e1 = u*ma.sqrt(2*dr-u**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    a1 = x0p*e1
    a2 = x0p*e2

    return -2*ma.tanh(-0.5*bcs*(en+fr)/tr)*(((en**2+dr**2+en*fr)/(ma.sqrt(2*dr-u**2)*e2))*intR(a2, a1+1, x)+u*intS(a2, a1+1, x))

@numba.jit("float64(float64[:])")
def reKlint2_a(args):
    #u-substitution, u**2 = en+fr-dr
    #from dr-fr to -fr/2, only fires if fr > 2*dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = u**2-fr+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(e1*ma.sqrt(u**2+2*dr))

    return -ma.tanh(-0.5*bcs*(en+fr)/tr)*((gfun+u)*intS(x0p*(e2-e1), 1, x)
                                                       -(gfun-u)*intS(x0p*(e2+e1), 1, x))

@numba.jit("float64(float64[:])")
def reKlint2_b(args):
    #u-substitution, u**2 = -en-dr
    #from -fr/2 to -dr, only fires if fr > 2*dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = -u**2-dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)

    return -ma.tanh(-0.5*bcs*(en+fr)/tr)*((gfun+u)*intS(x0p*(e2-e1), 1, x)
                                                       -(gfun-u)*intS(x0p*(e2+e1), 1, x))

@numba.jit("float64(float64[:])")
def reKlint3(args):
    #u-substitution, u**2 = en-dr
    #from dr to infinity
    #ulims are: 0, 31.6 <-- this is big enough, rounding errors happen at big numbers

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = u**2+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)
    t1 = ma.tanh(-0.5*bcs*en/tr)
    t2 = ma.tanh(-0.5*bcs*(en+fr)/tr)

    return (t1+t2)*(gfun-u)*intS(x0p*(e2+e1), 1, x) + (t1-t2)*(gfun+u)*intS(x0p*(e2-e1), 1, x)

@numba.jit("float64(float64[:])")
def imKlint1_a(args):
    #from dr-fr to -dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = u**2-fr+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(e1*ma.sqrt(u**2+2*dr))

    return ma.tanh(-0.5*bcs*(en+fr)/tr)*((gfun+u)*intR(x0p*(e2-e1), 1, x)+(gfun-u)*intR(x0p*(e2+e1), 1, x))

@numba.jit("float64(float64[:])")
def imKlint1_b(args):
    #from dr-fr to -dr
    #ulims are: 0, sqrt(0.5*fr-dr)

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = -u**2-dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)

    return ma.tanh(-0.5*bcs*(en+fr)/tr)*((gfun+u)*intR(x0p*(e2-e1), 1, x)+(gfun-u)*intR(x0p*(e2+e1), 1, x))

@numba.jit("float64(float64[:])")
def imKlint2(args):
    #u-substitution, u**2 = en-dr
    #from dr to infinity
    #ulims are 0, infinity

    u = args[0]
    x = args[1]
    x0p = args[2]
    tr = args[3]
    fr = args[4]
    dr = args[5]
    bcs = args[6]

    en = u**2+dr

    e1 = ma.sqrt(en**2-dr**2)
    e2 = ma.sqrt((en+fr)**2-dr**2)
    gfun = (en**2+dr**2+en*fr)/(ma.sqrt(u**2+2*dr)*e2)

    return ((ma.tanh(-0.5*bcs*en/tr)-ma.tanh(-0.5*bcs*(en+fr)/tr))*((gfun+u)*intR(x0p*(e2-e1), 1, x)
                                                                +(gfun-u)*intR(x0p*(e2+e1), 1, x)))


#Make C-language callbacks from each of the integrand functions for scipy quad
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


def cmplx_kernel(tr, fr, x, x0, x1, dr, bcs, verbose=False):
    """Calculate the Kl integral over energy. Kl = K*mfp**2 where K is the Kernel from Poppel.

    Arguments:

        tr is reduced temperature T/Tc
        fr is reduced frequency h*f/delta0
        x is q*mfp where mfp is mean-free-path
        x0 is mfp/ksi0 where ksi0 is pippard coherence length
        x1 is mfp/L0 where L0 is the London penetration depth at T=0
        dr is reduced BCS energy gap delta(T)/delta0

    Keyword Arguments:

        'verbose' (True or False)
            prints out values and stuff

    Returns:
        reKl+1j*imKl"""


    #Hackish way to make sure there are no divide by zero errors.
    if tr == 0:
        #This is a fun way to get machine epsilon
        tr = 7./3 - 4./3 -1

    #If temperature is >= Tc, then skip all the integration, and
    #use the normal Kernel
    if tr >= 1:

        #Calculate the actual number out front (1/x dependance included in R and S)
        prefactor = (3/np.pi)*fr*x0*x1**2

        reKl = intS(fr*x0/np.pi, 1, x)
        imKl = intR(fr*x0/np.pi, 1, x)

    else:
        #arguments to pass the integrand functions
        iargs = (x, x0/np.pi, tr, fr, dr, bcs,)

        #Calculate the actual number out front (1/x dependance included in R and S)
        prefactor = (3/np.pi)*x0*x1**2

        #Now run the integrals

        if fr/dr < 2:
            #from (dr-fr) to dr (inside the gap), limits look wierd due to u-substitution
            reKl1a, reKl1aerr, _, _ = do_integral(c_reKlint1_a2.ctypes, 0, ma.sqrt(0.5*fr), iargs, verbose=verbose)
            reKl1b, reKl1berr, _, _ = do_integral(c_reKlint1_b.ctypes, 0, ma.sqrt(0.5*fr), iargs, verbose=verbose)

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

            reKl1a, reKl1aerr, _, _ = do_integral(c_reKlint1_a1.ctypes, 0, ma.sqrt(dr), iargs, verbose=verbose)
            reKl1b, reKl1berr, _, _ = do_integral(c_reKlint1_b.ctypes, 0, ma.sqrt(dr), iargs, verbose=verbose)

            reKl1 = reKl1a + reKl1b
            reKl1err = reKl1aerr + reKl1berr

            #from dr-fr to -dr (below the gap), limits look wierd due to u-substitution
            reKl2a, reKl2aerr, _, _ = do_integral(c_reKlint2_a.ctypes, 0, ma.sqrt(0.5*fr-dr), iargs, verbose=verbose)
            reKl2b, reKl2berr, _, _ = do_integral(c_reKlint2_b.ctypes, 0, ma.sqrt(0.5*fr-dr), iargs, verbose=verbose)

            reKl2 = reKl2a+reKl2b
            reKl2err = reKl2aerr+reKl2berr

            imKl1a, imKl1aerr, _, _ = do_integral(c_imKlint1_a.ctypes, 0, ma.sqrt(0.5*fr-dr), iargs, verbose=verbose)
            imKl1b, imKl1berr, _, _ = do_integral(c_imKlint1_b.ctypes, 0, ma.sqrt(0.5*fr-dr), iargs, verbose=verbose)

            imKl1 = imKl1a+imKl1b
            imKl1err = imKl1aerr+imKl1berr



        #from dr to infinity (above the gap), limits look wierd due to u-substitution
        #Wierd things happen numerically at infinity. 31.6 is far enough. equals about 1000 gaps.
        reKl3, reKl3err, _, _ = do_integral(c_reKlint3.ctypes, 0, 31.6, iargs, verbose=verbose)
        imKl2, imKl2err, _, _ = do_integral(c_imKlint2.ctypes, 0, np.inf, iargs, verbose=verbose)

        reKl = reKl1+reKl2+reKl3
        imKl = imKl1+imKl2


#         if verbose:
#             print 'x = '+str(x)+'\n'
#             print 'prefactor = '+str(prefactor)+'\n'
#             print 'First reK integral = '+str(reKl1)+' +/- '+str(reKl1err) + '\n'
#             print 'Second reK integral = '+str(reKl2)+' +/- '+str(reKl2err) + '\n'
#             print 'Third reK integral = '+str(reKl3)+' +/- '+str(reKl3err) + '\n'
#             print 'First imK integral = '+str(imKl1)+' +/- '+str(imKl1err) + '\n'
#             print 'Second imK integral = '+str(imKl2)+' +/- '+str(imKl2err) + '\n'

    return prefactor*(reKl+1j*imKl)
