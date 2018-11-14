from doptools.io import Database
from doptools.model_doptrack import Spectrogram
from doptools.extraction import ExtractedData


db = Database()

L0_dataids = db.dataids['L0']
L1B_dataids = db.dataids['L1B']

dataids_to_process = L0_dataids - L1B_dataids

for dataid in dataids_to_process:
    spec = Spectrogram.create(dataid, nfft=250_000, dt=1)
    data = ExtractedData.create(spec)
    data.save()
