import matplotlib.pyplot as plt
import numpy as np

from doptools.io import *
from doptools.model_doptrack import Spectrogram
from doptools.analysis import ResidualAnalysis, BulkAnalysis
from doptools.groundstation import *

#dataids = ['Delfi-C3_32789_201602121133',
#           'Delfi-C3_32789_201609151132']
##           'Delfi-C3_32789_201602021010',
##           'Delfi-C3_32789_201603020932']




dataids = ['Delfi-C3_32789_201602121133',
           'Delfi-C3_32789_201602210946',
           'Delfi-C3_32789_201602201122',
           'Delfi-C3_32789_201607132324']

#dataids = ['Delfi-C3_32789_201602121133_02s',
#           'Delfi-C3_32789_201602210946_02s',
#           'Delfi-C3_32789_201602201122_02s',
#           'Delfi-C3_32789_201607132324_02s']
#
#dataids = ['Delfi-C3_32789_201602210946_02s',
#           'Delfi-C3_32789_201602201122_02s']

#dataids = ['Delfi-C3_32789_201602121133',
#           'Delfi-C3_32789_201602210946',
#           'Delfi-C3_32789_201607132324']
#
dataids = ['Delfi-C3_32789_201602210946']

#dataids = ['Delfi-C3_32789_201607132324']
#dataids = ['Delfi-C3_32789_201602201122']

#dataids = ['Delfi-C3_32789_201602210946',
#           'Delfi-C3_32789_201602201122']

import doptools.drre.drre as drre

for dataid in dataids:
    drre.main(dataid)




#s1 = Spectrogram.load('Delfi-C3_32789_201602121133_50khz')
#s1.plot()
#
#s2 = Spectrogram.create('Delfi-C3_32789_201602121133')
#s2.plot()


#for dataid in dataids:
#    s = Spectrogram.create(dataid, nfft=250000, dt=0.2)
#    s.save(f'{dataid}_02s')
##    s.plot(clim=(0, 0.1), cmap='viridis')





#data = np.fromfile('../data/tracking/Delfi-C3_32789_201602121133.32fc')
#
#mask = np.zeros(20)
#mask[[0,1]] = 1
#mask = np.broadcast_to(mask, (int(len(data)/20), 20))
#mask = mask.flatten()
#mask = mask.astype(bool)
#
#data = data[mask]
#
#data.tofile('../data/tracking/Delfi-C3_32789_201602121133_25khz.32fc')






#dataid1 = 'Delfi-C3_32789_201602121133'
#dataid2 = 'Delfi-C3_32789_201602121133_50khz'
#dataid3 = 'Delfi-C3_32789_201602121133_25khz'

#s1 = Spectrogram.create(dataid1)
#s2 = Spectrogram(dataid2)
#s3 = Spectrogram(dataid3)

#s1.plot()
#s2.plot(bounds=None, clim=(-65, -45))
#s3.plot(bounds=None, clim=(-65, -45))

#for dataid in dataids:
#    a = ResidualAnalysis(dataid)
#
#    plt.figure()
#    plt.plot(a.tle.time, a.tle.rangerate, label='tle')
#    plt.plot(a.doptrack.time, a.doptrack.rangerate, label='doptrack')
#    plt.title(dataid)
#    plt.legend()
#    plt.show()


#a = BulkAnalysis()

#dataid = 'Delfi-C3_32789_201602121133'
#dataid = 'Delfi-C3_32789_201603020932'
#a = ResidualAnalysis(dataid)


