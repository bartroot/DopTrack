""""Process DopTrack data to level 1B (time-frequency data)."""
import logging
import gc

from doptools.io import Database
from doptools.data import L1A, L1B, EmptyRecordingError
from doptools.extraction import PassNotFoundError
from doptools.utils import log_formatter


db = Database()


logger = logging.getLogger('process_L1B')
logger.setLevel(logging.DEBUG)
doplogger = logging.getLogger('doptools')
doplogger.setLevel(logging.DEBUG)

fh1 = logging.FileHandler(db.paths['logs'] / 'process_L1B.log')
fh1.setLevel(logging.INFO)
fh1.setFormatter(log_formatter)

fh2 = logging.FileHandler(db.paths['logs'] / 'process_L1B_error.log')
fh2.setLevel(logging.ERROR)
fh2.setFormatter(log_formatter)

logger.addHandler(fh1)
doplogger.addHandler(fh1)
logger.addHandler(fh2)


db = Database()
dataids_to_process = db.dataids['L0'] - db.dataids['L1B'] - db.dataids['L1B_failed']

logger.info(f'Processing {len(dataids_to_process)} data sets to level 1B')
for dataid in sorted(dataids_to_process):
    logger.info(f'Processing {dataid}')

    try:
        try:
            L1A_obj = L1A.load(dataid)
            assert L1A_obj.dt == 0.2
        except (FileNotFoundError, AssertionError):
            L1A_obj = L1A.create(dataid, nfft=250_000, dt=0.1)
    except EmptyRecordingError as e:
        logger.warning(e)
        db.update_status(dataid, status='empty_recording')
        continue

    try:
        L1B_obj = L1B.create(L1A_obj)
        L1B_obj.save()
        figpath = db.paths['output'] / 'L1B' / f'{dataid}.png'
        L1B_obj.plot(savepath=figpath, L1A=L1A_obj, fit_func=False)
        logger.info(f'Extraction of {dataid} succeded')
        db.update_status(dataid, status='success')
    except PassNotFoundError as e:
        figpath = db.paths['output'] / 'L1B_failed' / f'{dataid}.png'
        L1A_obj.plot(savepath=figpath)
        logger.info(f'No pass was found in {dataid}: {e}')
        db.update_status(dataid, status='pass_not_found')
    except Exception:
        figpath = db.paths['output'] / 'L1B_failed' / f'{dataid}.png'
        L1A_obj.plot(savepath=figpath)
        logger.exception(f'Extraction of {dataid} failed unexpectedly')
        db.update_status(dataid, status='unknown_error')

    gc.collect()  # Added to ensure that memory usage stays as low as possible
