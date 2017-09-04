import scipy.signal as sc
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from math import *
import matplotlib.pyplot as plt


def image2tf(data, mask, t, freq, window, dispFig):
    """
    Function that analyses the spectogram of the satellite signal to find the 
    transmission frequency of the satellite. 
    """
    #dispFig = 0
    sideBand = 600
    #est_freq = 6000
    #est_shift = 4000
    
    mData = np.mean(data)
    If = np.multiply(data,mask) + np.multiply(1-mask, mData)

    # select usable time area for fitting
    
    _, loc, us = filterSatData(If, window, satSignal(window, sideBand))
    print("i2tf")
    tu = t[us==1]
    ploc = loc[us==1]
    # fit tanh to retrieve carrier frequency
    
    tFit, _ = fitTanh(tu[100:], ploc[100:], window ,dispFig)
    x = np.arange(freq.size)
    fc = (interp1d(x, freq))(tFit[1])
    TCA = tFit[3]
    #crop to first length estimate
    lEst = tFit[2]*2
    ploc = ploc[(tu>TCA-lEst) & (tu<TCA+lEst)]
    tu = tu[(tu>TCA-lEst) & (tu<TCA+lEst)]

    residu = ploc - Tanh(tu, *tFit)
    rest, residu = removeOutliers(tu, residu, 0, 200*window)

    # Fit disturbance model on residu
    rfit, _, fn = resFit(rest, residu, dispFig)
    fit = Tanh(t, *tFit) + fn(t, *rfit)
    if dispFig:
        plt.plot(t, fit)
        plt.show()

    #window = 100*window
    mask = np.zeros((t.size, freq.size))

    for j in range(t.size):
        line = [0 if (i < fit[j] - 100*window) | (i > fit[j] + 100*window) else 1 for i in range(freq.size)]
        mask[j,:] = line

    print(np.mean(mask))
    #filter data and find peaks
    Ic = np.multiply(If, mask) + np.multiply(1-mask, mData)

    _, loc2, us2 = filterSatData(Ic, window, sc.gaussian(100*window,2.5))
    tu2 = t[us2==1]
    ploc2 = loc2[us2==1]
    #remove outliers and fit tanh to retrieve carrier frequency
    
    tFit2, _ = fitTanh(tu, ploc, window ,dispFig)

    TCA = tFit2[3]
    fc = (interp1d(x, freq))(tFit2[1])
    lEst=tFit2[2]*2.8
    ploc2 = ploc2[(tu2>TCA-lEst)&(tu2<TCA+lEst)]
    tu2 = tu2[(tu2>TCA-lEst)&(tu2<TCA+lEst)]

    residu2 = ploc2- Tanh(tu2, *tFit2)
    rest2, residu2 = removeOutliers(tu2, residu2, 0, 100*window)

   # rFit2, _ = resFit(rest2, residu2, dispFig)
    fit2 = Tanh(tu2, *tFit)# + fourier5(tu2, *rfit) 

    print("Estimated frequency: ", fit2[1])
    print("Estimated midpoint: ", fit2[3])

    t2= tu2[abs(ploc2-fit2)<35*window]
    pks2= ploc2[abs(ploc2-fit2)<35*window]
    Id=np.multiply((data-mData),mask)

    #python runs from 0_to_end so t2-1 is needed.
    #std over rows of Id
    stdpks = np.std(Id[(t2).astype(int),:],axis=1)
    amppks = np.zeros(len(t2))
    for i in range(len(t2)):
        amppks[i]=Id[int(t2[i]),int(pks2[i])]

    with np.errstate(divide='ignore', invalid='ignore'):    
        acc = np.divide(stdpks,amppks)  #normalized standard deviation
        acc[~np.isfinite(acc)] = 0      #catch divide by zeroes

    tf=t2
    pksf=freq[pks2.astype(int)]

    tresh = np.median(acc)+np.std(acc)*2
    tresha = np.median(amppks)-np.std(amppks)

    #filter out weak parts
    tf = tf[(acc<=tresh)&(amppks>=tresha)]
    pksf = pksf[(acc<=tresh)&(amppks>=tresha)]
    acc = acc[(acc<=tresh)&(amppks>=tresha)]

    if dispFig:
        #plt.imshow(freq,t,np.log10(data))
        plt.scatter(pksf,tf)
        plt.show()


    t=tf
    freq=pksf

    class im():
        def __init__(self, t, freq, acc, fc, TCA):
            self.t = t
            self.freq = freq
            self.acc = acc
            self.fc = fc
            self.TCA = TCA


    return im(t, freq, acc, fc, TCA)




    
def removeOutliers(t, y, mean, band):
    y_ = y[(y>mean-band) & (y<mean+band)]
    t_ = t[(y>mean-band) & (y<mean+band)]
    return t_, y_
    

def filterSatData(data,window, _filter):
    sz = data.shape
    noisefilt = sc.gaussian(int(14/window)+1, 2.5)
    print("filtersatdata")
    noisefilt = noisefilt/np.sum(noisefilt)

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
                plt.show()            
            return fitresult, covar, fn
        except:
            continue

    #This return statement only occurs when no match is found. 
    return [], 0, p0


    
def bisquare(rho):
    pass