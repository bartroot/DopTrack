import scipy.signal as sc
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from math import *
import matplotlib.pyplot as plt


def image2tf(data, mask, t, freq, window, dispFig):
    dispFig = 1
    sideBand = 600
    est_freq = 2000
    print(data.shape)
    mData = np.mean(data)
    #If = data*mask+(1-mask)*mData
    If = np.multiply(data,mask) + np.multiply(1-mask, mData)
    _, loc, us = filterSatData(If, window, satSignal(window, sideBand))
    #select usable time area for fitting
    print("i2tf")
    tu = t[us==1]
    ploc = loc[us==1]
    #remove outliers and fit tanh to retrieve carrier frequency
    tu, ploc = removeOutliers(tu, ploc, est_freq, 2*sideBand)
    tFit, _ = fitTanh(tu, ploc, window ,dispFig)
    x = np.arange(freq.size)
    fc = interp1d(x, freq, fill_value=tFit[1])
    TCA = tFit[3]
    #crop to first length estimate
    lEst = tFit[2]*2
    ploc = ploc[(tu>TCA-lEst) & (tu<TCA+lEst)]
    tu = tu[(tu>TCA-lEst) & (tu<TCA+lEst)]

    residu = ploc - Tanh(tu, *tFit)
    rest, residu = removeOutliers(tu, residu, 0, 200*window)

    # Fit disturbance model on residu
    rfit, _ = resFit(rest, residu, dispFig)
    fit = Tanh(t, *tFit) + fourier4(t, *rfit)
    plt.plot(t, fit)
    plt.show()
    print(data.shape)


    #window = 100*window
    mask = np.zeros((t.size, freq.size))

    for j in range(t.size):
        line = [0 if (i < fit[j] - window) | (i > fit[j] + window) else 1 for i in range(freq.size)]
        mask[j,:] = line

    #filter data and find peaks
    Ic = np.multiply(If, mask) + np.multiply(1-mask, mData)
    #_, loc2, us2 = filterSatData(Ic, window, sc.gaussian(100*window))


    
def removeOutliers(t, y, mean, band):
    y = y[(y>mean-band) & (y<mean+band)]
    t = t[(y>mean-band) & (y<mean+band)]
    return t, y
    

def filterSatData(data,window, _filter):
    sz = data.shape
    noisefilt = sc.gaussian(int(14/window)+1, 2.5)
    print("filtersatdata")
    noisefilt = noisefilt/np.sum(noisefilt)
    #mask = sc.gaussian(100*window, 2.5)
    avgline = np.mean(np.mean(data,axis=1))
    rdata=np.zeros(sz)
    peak=np.zeros((sz[0]))
    peaks=np.zeros((sz[0]))
    for i, row in enumerate(data):        
        avgdata=row-avgline

        avgdata[avgdata<5*np.std(avgdata)]=0
        avgdata=np.convolve(avgdata,noisefilt,mode='same')
        p= np.convolve(avgdata, _filter,mode='same')
        rdata[i][:]=avgdata
        pks = np.max(avgdata)
        loc = np.argmax(avgdata)        
        if pks:
            peaks[i]=pks
            peak[i]=loc
#    %define usable data space
    rpeaks = peak

    return rdata, rpeaks, usable(peaks, window, sz)

def Tanh(tt, a, b, c, d):
    return -a*np.tanh((tt-d)/c)+b

def fitTanh(t, pks, dt, dispFig):
    print('fittan')
    # define the function to fit

    # initial guess
    a0 = (np.max(pks)-np.min(pks))/2
    b0 = np.mean(pks)
    c0 = 10/dt
    d0 = np.mean(t)
    p00 = [a0, b0, c0, d0]
    
    #plt.plot(t, fff(t,*p00))
    #plt.show()
    #print(f(*p0))
    lower = [0.5*a0, 0.7*b0, 20, d0-500*dt] 
    upper = [1.5*a0, 1.3*b0, 200, d0+500*dt]
    bounds = (lower, upper)
    ftol = 10**-8
    xtol = 10**-8
    max_nfev = 6000

    # prepare curve data
    # non-linear least squares
    fitresult, covar = curve_fit(Tanh, t, pks, bounds=bounds, 
                                ftol=ftol, xtol=xtol, max_nfev=max_nfev)
    
    print("Estimated frequency: ", fitresult[1])
    print("Estimated midpoint: ", fitresult[3])

    if dispFig:
        ### plot it
        plt.scatter(t,pks)
        plt.plot(t, Tanh(t, *fitresult))
        plt.show()
    return fitresult, covar
    


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def usable(peaks, window, sz):
    #peaks[peaks<(np.mean(peaks)-np.std(peaks)*0.5)]=0
    peaks[peaks>0]=1
    peaks = moving_average(peaks, int(200/window))
    peaks[peaks>0.2] = 1
    peaks[peaks<=0.2] = 0
    peaks = np.diff(peaks)
    d1 = np.nonzero(peaks>0.5)[0]
    d2 = np.nonzero(peaks<-0.5)[-1]
    if not d1.any() or d1>(sz[0]/2):
        d1=1
# Als d2 kleiner dan de helft of geen waarde ->gelijk aan shape
    if not d2.any() or d2<(sz[0]/2):
        d2=sz[0]
    usable=np.arange(0,sz[0])
    usable[usable>d2]=0
    usable[usable<d1]=0
    usable[usable>0]=1
    return usable

def satSignal(freqStep, sideBand):
    ##Hardcoded for delfi
    dp = sideBand * freqStep
    dz = round(dp/4)
    l = round(dp*1.2)
    h = np.zeros((1,l*2))
    h[0,l-1] = 1.2
    h[0,l-dp-1] = 1
    h[0,l+dp-1] = 1
    sig = moving_average(h, int(freqStep*21)*2+1)
    sig = moving_average(sig, int(freqStep*11)*2+1)
    sig = moving_average(sig, int(freqStep*9)*2+1)
    return sig

def fourier4(x,a0,a1,a2,a3,a4,b1,b2,b3,b4,p):
    return a0 + a1 * np.cos(1*x*p) + b1 * np.sin(1*x*p) + \
            a2 * np.cos(2*x*p) + b2 * np.sin(2*x*p) + \
            a3 * np.cos(3*x*p) + b3 * np.sin(3*x*p) + \
            a4 * np.cos(4*x*p) + b4 * np.sin(4*x*p)

def resFit(rest, residu, dispFig):
    print('resfit')
    # define the function to fit
    #plt.plot(t, fff(t,*p00))
    #plt.show()
    #print(f(*p0))
    ftol = 10**-8
    xtol = 10**-8
    max_nfev = 6000

    # prepare curve data
    # non-linear least squares
    fitresult, covar = curve_fit(fourier4, rest, residu, 
        ftol=ftol, xtol=xtol)
    
    if dispFig:
        ### plot it
        plt.plot(rest, residu)
        plt.plot(rest, fourier4(rest, *fitresult), color='r')
        plt.show()
    return fitresult, covar
    
