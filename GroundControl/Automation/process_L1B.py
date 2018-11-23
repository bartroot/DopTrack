import logging
import sys

from doptools.io import Database
from doptools.data import Spectrogram, FrequencyData


db = Database()


formatter = logging.Formatter('%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)-s')

logger = logging.getLogger('process_L1B')
logger.setLevel(logging.DEBUG)
doplogger = logging.getLogger('doptools')
doplogger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh1 = logging.FileHandler(db.paths['logs'] / 'process_L1B.log')
fh1.setLevel(logging.INFO)
fh1.setFormatter(formatter)

fh2 = logging.FileHandler(db.paths['logs'] / 'process_L1B_fail.log')
fh2.setLevel(logging.WARNING)
fh2.setFormatter(formatter)

logger.addHandler(sh)
doplogger.addHandler(sh)
logger.addHandler(fh1)
doplogger.addHandler(fh1)
logger.addHandler(fh2)


db = Database()

L0_dataids = db.dataids['L0']
L1B_dataids = db.dataids['L1B']

dataids_to_process = L0_dataids - L1B_dataids


logger.info(f'Processing {len(dataids_to_process)} data sets to level 1B')
for dataid in sorted(dataids_to_process):
    logger.info(f'Processing {dataid}')

    spec = Spectrogram.create(dataid, nfft=250_000, dt=1)
    path = db.paths['output'] / 'L1A' / f'{dataid}.png'
    spec.plot(savepath=path)

    try:
        data = FrequencyData.create(spec)
        data.save()
        path = db.paths['output'] / 'L1B' / f'{dataid}.png'
        data.plot(savepath=path, fit_func=False)
        if data.residual_func.__name__ == 'fourier6':
            logger.info(f'Extraction of {dataid} succeded.')
        else:
            logger.warning(f'Extraction of {dataid} succeded, but may be extracted incorrectly.')
    except Exception:
        logger.exception(f'Extraction of {dataid} failed.')
