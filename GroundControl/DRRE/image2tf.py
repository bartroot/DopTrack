import scipy.signal as sc
import numpy as np


def filterSatData(data,timeStep):
    sz = data.shape


    noisefilt = sc.gaussian(int(14/timeStep)+1, 2.5)
    noisefilt = noisefilt/np.sum(noisefilt)
    mask = sc.gaussian(100*timeStep, 2.5)
    avgline = np.mean(np.mean(data,axis=1))
    
    rdata=np.zeros(sz)
    peak=np.zeros((1,sz[0]))
    peaks=np.zeros((1,sz[0]))
    for i, row in enumerate(data):
        print(i)
        avgdata=row-avgline
        avgdata[avgdata<5*np.std(avgdata)]=0
        avgdata=np.convolve(avgdata,noisefilt,mode='same')
        p= np.convolve(avgdata,np.transpose(mask),mode='same')
        rdata[i][:]=avgdata
        [pks,loc] = np.max(avgdata)
        print(np.max(avgdata))
"""        
        if ~isempty(pks)
            peaks(i)=pks
            peak(i)=loc
        end
    end
    
    %define usable data space
    peaks(peaks<(mean(peaks)-std(peaks)*0.5))=0
    peaks(peaks>0)=1
    peaks=smooth(peaks,200/timeStep)
    usable=[0peaks>0.2]
    usable=diff(usable)
    d1=find(usable>0.5,1,'first')
    d2=find(usable<-0.5,1,'last')
    if isempty(d1)||d1>sz(1)/2
        d1=1
    end
    if isempty(d2)||d2<sz(1)/2
        d2=sz(1)
    end
    usable=(1:sz(1))
    usable(usable>d2)=0
    usable(usable<d1)=0
    usable(usable>0)=1

    rpeaks = peak
end
"""