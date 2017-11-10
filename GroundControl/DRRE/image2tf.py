import scipy.signal as sc
import numpy as np
from math import *
import matplotlib.pyplot as plt
from fitting import *


def image2tf(FourierData, SatData, mask, window, dispFig):
    """
    Function that analyses the spectogram of the satellite signal to find the 
    transmission frequency of the satellite. 
    """
    sideBand = SatData.sideBand
    data = FourierData.I
    t = FourierData.time
    freq = FourierData.freq
    
    mData = np.mean(data)
    If = np.multiply(data,mask) + np.multiply(1-mask, mData)

    # select usable time area for fitting 
    _, loc, us = filterSatData(If, window, satSignal(window, sideBand))
    print("i2tf")
    tu = t[us==1]
    ploc = loc[us==1]
    ploc = ploc[tu>100]
    tu = tu[tu>100]
    
    # fit tanh to retrieve carrier frequency, the first 100s aren't used for plotting    
    # The outliers are progressively removed to obtain a tighter fit
    tFit, _ = fitTanh(tu, ploc, window ,dispFig)
    tu, ploc = removeOutliers(tu, ploc, Tanh(tu, *tFit), 2.5*sideBand)
    tFit, _ = fitTanh(tu, ploc, window, dispFig)
    tu, ploc = removeOutliers(tu, ploc, Tanh(tu, *tFit), 1.5*sideBand)
    tFit, _ = fitTanh(tu, ploc, window, dispFig)
    tu, ploc = removeOutliers(tu, ploc, Tanh(tu, *tFit), .5*sideBand)
    tFit, _ = fitTanh(tu, ploc, window, dispFig)
    x = np.arange(freq.size)
    fc = (interp1d(x, freq))(tFit[1])
    TCA = tFit[3]

    #crop to first length estimate
    lEst = tFit[2]*2
    ploc = ploc[(tu>TCA-lEst) & (tu<TCA+lEst)]
    tu = tu[(tu>TCA-lEst) & (tu<TCA+lEst)]

    # Fit a disturbance model on remaining errors
    residu = ploc - Tanh(tu, *tFit)
    rest, residu = removeOutliers(tu, residu, 0, 200*window)
    #rfit, _, fn = resFit(rest, residu, dispFig)
    fit = Tanh(t, *tFit) #+ fn(t, *rfit)


    if dispFig:
        plt.plot(t, fit, color='r')
        plt.scatter(tu, ploc)
        plt.title("First fit")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.show()

    #Create a mask from the first fit
    mask = np.zeros((t.size, freq.size))
    for j in range(t.size):
        line = [0 if (i < fit[j] - 100*window) | (i > fit[j] + 100*window) else 1 for i in range(freq.size)]
        mask[j,:] = line

    #filter data and find peaks
    Ic = np.multiply(If, mask) + np.multiply(1-mask, mData)
    _, loc2, us2 = filterSatData(Ic, window, sc.gaussian(100*window,2.5))
    tu2 = t[us2==1]
    ploc2 = loc2[us2==1]
    ploc2 = ploc2[tu2>100]
    tu2 = tu2[tu2>100]

    #remove outliers and fit tanh to retrieve carrier frequency
    tFit2, _ = fitTanh(tu, ploc, window ,dispFig)
    TCA = tFit2[3]
    fc = (interp1d(x, freq))(tFit2[1])
    lEst=tFit2[2]*2.8
    ploc2 = ploc2[(tu2>TCA-lEst)&(tu2<TCA+lEst)]
    tu2 = tu2[(tu2>TCA-lEst)&(tu2<TCA+lEst)]
    residu2 = ploc2- Tanh(tu2, *tFit2)
    rest2, residu2 = removeOutliers(tu2, residu2, 0, 100*window)

    rFit2, _, fn = resFit(rest2, residu2, dispFig)
    fit2 = Tanh(tu2, *tFit) + fn(tu2, *rFit2) 
    print("Estimated frequency: ", tFit2[1])
    print("Estimated midpoint: ", tFit2[3])
    t2= tu2[abs(ploc2-fit2)<35*window]
    pks2= ploc2[abs(ploc2-fit2)<35*window]
    Id=np.multiply((data-mData),mask)

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
        FD = FourierData
        plt.imshow(10*np.log10(np.flipud(FD.I.T)),
            extent=(FD.time[0], FD.time[-1],FD.freq[0], FD.freq[-1]),
            aspect='auto', cmap='afmhot')
        plt.scatter(tf, pksf, color='b', s=9)
        plt.title("Final Peaks")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.show()
        """
        plt.imshow(10*np.log10(np.flipud(Ic.T)),
        extent=(FD.time[0], FD.time[-1],FD.freq[0], FD.freq[-1]),
        aspect='auto', cmap='afmhot')
        plt.show()        """
    t=tf
    freq=pksf

    class im():
        def __init__(self, t, freq, acc, fc, TCA, Ic):
            self.t = t
            self.freq = freq
            self.acc = acc
            self.fc = fc
            self.TCA = TCA
            self.Ic = Ic
    return im(t, freq, acc, fc, TCA, Ic)

 
def removeOutliers(t, y, mean, band):
    y_ = y[(y>mean-band) & (y<mean+band)]
    t_ = t[(y>mean-band) & (y<mean+band)]
    return t_, y_
    

def filterSatData(data,window, _filter):
    sz = data.shape
    noisefilt = sc.gaussian(int(14/window)+1, 2.5)
    print("filterSatData")
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
    dp = int(sideBand * freqStep)
    dz = int(dp/4)
    l = int(dp*1.2)
    h = np.zeros((1,l*2))
    h[0,l-1] = 1.2
    h[0,l-dp-1] = 1
    h[0,l+dp-1] = 1
    sig = moving_average(h, int(freqStep*21)*2+1)
    sig = moving_average(sig, int(freqStep*11)*2+1)
    sig = moving_average(sig, int(freqStep*9)*2+1)
    return sig
