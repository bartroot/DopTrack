import logging

from doptools.io import Database
from doptools.model_doptrack import Spectrogram, Recording
from doptools.extraction import FrequencyData


logger = logging.getLogger(__name__)

fh1 = logging.FileHandler('process.log')
fh1.setLevel(logging.INFO)

fh2 = logging.FileHandler('process_debug.log')
fh2.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh1.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh1)
logger.addHandler(fh2)
logger.addHandler(ch)


# TODO later run 1 sec to find good passes and then only run good passes with 0.1
# TODO add quaility values (just tanh, and tanh+residual_func)  look at RMS or std

db = Database()
L0_dataids = db.dataids['L0']
L1B_dataids = db.dataids['L1B']

dataids_to_process = L0_dataids - L1B_dataids

for dataid in dataids_to_process:
    rec = Recording(dataid)
    # TODO read sample rate from meta file
    spec = Spectrogram.create(dataid, nfft=250_000, dt=0.2)
    data = FrequencyData.create(spec)
    data.save()
    path = db.paths['default'] / '../output' / f'{dataid}.png'
    data.plot(savepath=path)
