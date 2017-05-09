import scipy.signal as sc
import numpy as np


def filterSatData(data,window):
    sz = data.shape
    noisefilt = sc.gaussian(int(14/window)+1, 2.5)
    print(noisefilt.shape)
    noisefilt = noisefilt/np.sum(noisefilt)
    mask = sc.gaussian(100*window, 2.5)
    avgline = np.mean(np.mean(data,axis=1))
    rdata=np.zeros(sz)
    peak=np.zeros((1,sz[0]))
    peaks=np.zeros((1,sz[0]))
    for i, row in enumerate(data):
        avgdata=row-avgline
        avgdata[avgdata<5*np.std(avgdata)]=0
        avgdata=np.convolve(avgdata,noisefilt,mode='same')
        p= np.convolve(avgdata,np.transpose(mask),mode='same')
        rdata[i][:]=avgdata
        pks = np.max(avgdata)
        loc = np.argmax(avgdata)
        #why???
        if not pks:
            peaks[0][i]=pks
            peak[0][i]=loc
#    %define usable data space
    peaks[peaks<(np.mean(peaks)-np.std(peaks)*0.5)]=0
    peaks[peaks>0]=1
    print(pks)
    peaks = moving_average(peaks, int(200/window))
    usable=np.array([peaks>0.2])
    #print(usable)
    # 0-usable?
    #usable=diff(usable)
    d1 = np.nonzero(usable>0.5)[0]
    d2 = np.nonzero(usable<-0.5)[-1]
    if not d1 or d1>(sz[0]/2):
        d1=1
    if not d2 or d2<(sz[0]/2):
        d2=sz[0]
    usable=np.arange(1,sz[0])
    usable[usable>d2]=0
    usable[usable<d1]=0
    usable[usable>0]=1

    rpeaks = peak
    #print(peaks)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n