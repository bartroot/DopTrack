from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
from math import *
import matplotlib.pyplot as plt


def Tanh(tt, a, b, c, d):
    return -a*np.tanh((tt-d)/c)+b

def fitTanh(t, pks, dt, dispFig):
    print('fittan')
    # define the function to fit
     
    # initial guess
    a0 = (np.max(pks)-np.min(pks))/2
    b0 = np.mean(pks)
    c0 = 100/dt
    d0 = np.mean(t)
    p00 = [a0, b0, c0, d0]
    lower = [0.5*a0, 0.7*b0, 20, d0-500*dt] 
    upper = [1.5*a0, 1.3*b0, 200, d0+500*dt]
    bounds = (lower, upper)
    ftol = 10**-8
    xtol = 10**-8
    max_nfev = 10000

    # prepare curve data
    # non-linear least squares
    fitresult, covar = curve_fit(Tanh, t, pks, bounds=bounds, p0=p00, loss='soft_l1',
                                method='trf', ftol=ftol, xtol=xtol, max_nfev=max_nfev)

    if dispFig:
        ### plot it        
        plt.scatter(t,pks)
        plt.plot(t, Tanh(t, *fitresult), color='r')
        plt.title("Tanh fit")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.show()
    return fitresult, covar


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def fourier4(x,a0,a1,a2,a3,a4,b1,b2,b3,b4,p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p)


def fourier5(x,a0,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p) + \
            a5 * np.cos(5*x*p) + b5 * np.sin(5*x*p)


def fourier3(x,a0,a1,a2,a3, b1,b2,b3,p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p)
           

def p3(x, a0, a1, a2, a3, p):
    return a0 + a1 * (x - p) + a2 * (x - p)**2 + a3 * (x - p)**3


def p5(x, a0, a1, a2, a3, a4, a5, p):
    return a0 + a1 * (x - p) + a2 * (x - p)**2 + a3 * (x - p)**3 + \
    a4 * (x - p)**4 + a5 * (x - p)**5


def p0(x):
    return x * 0


def resFit(rest, residu, dispFig):
    print('residual fit')
    # define the function to fit
    #plt.plot(t, fff(t,*p00))
    #plt.show()
    #print(f(*p0))
    ftol = 10**-8
    xtol = 10**-8
    max_nfev = 6000
    p00 = [0,0,0,0,0,0,0,0,0,0.8]
    a = (rest)/np.std(rest)
    # non-linear least squares
    i =1
    for fn in [fourier4, fourier3, fourier5, p3, p5]:
        print(i)
        i+=1
        try:
            fitresult, covar = curve_fit(fn, a, residu, #p0=p00,
                method='trf', loss='soft_l1', max_nfev=max_nfev)
            fitresult[-1] = (fitresult[-1])/np.std(rest)
            print('Score:',np.mean(rest-fn(rest, *fitresult)))
            if np.abs(np.mean(rest-fn(rest, *fitresult))) > 10000:
                raise Exception('Did not converge')
            if dispFig:
                plt.scatter(rest, residu)
                plt.plot(rest, fn(rest, *fitresult), color='r')
                plt.title("Residue fit")
                plt.ylabel("Normalized frequency difference ()")
                plt.xlabel("Time (s)")
                plt.show()            
            return fitresult, covar, fn
        except:
            continue

    #This return statement only occurs when no match is found. 
    return [], 0, p0