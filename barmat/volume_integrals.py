from __future__ import division
import numba
import math as ma

@numba.jit("float64(float64, float64, float64)")
def intR(a, b, x):
#Calculate the R integral from Popel divided by x
#x = q*l where l = mean free path

    z2 = a**2+b**2

    if x == 0:
        r = b/(3.0*z2) #This is really r/x

    elif x < 0.001*ma.sqrt(z2):
        r = b/(3.0*z2) #This is really r/x

    elif x > 2000*z2:
        r = (ma.pi*(1+(b**2-a**2)/x**2)/4-b/x)/x #This is really r/x

    else:
        #calculate all the terms of r
        r = (1/x**2)*(-0.5*b*x+0.25*a*b*ma.log((z2+x**2+2*a*x)/(z2+x**2-2*a*x))+
        0.25*(x**2+b**2-a**2)*ma.atan2(2*b*x,(z2-x**2)))/x #This is really r/x

    return r

@numba.jit("float64(float64, float64, float64)")
def intS(a, b, x):
#Calculate the S integral from Popel divided by x
#x = q*l where l = mean free path

    z2 = a**2+b**2

    if x == 0:
        s = a/(3.0*z2) #This is really s/x

    elif x < 0.001*ma.sqrt(z2):
        s = a/(3.0*z2) #This is really s/x

    elif x > 2000*z2:
        s = (a/x - a*b*ma.pi/(2*x**2))/x #This is really s/x

    #calculate all the terms of s
    else:
        s = (1/x**2)*(0.5*(a*x)+
            0.125*(x**2+b**2-a**2)*ma.log((z2+x**2+2*a*x)/(z2+x**2-2*a*x))-
            0.5*b*a*ma.atan2(2*b*x,(z2-x**2)))/x #This is really s/x

    return s