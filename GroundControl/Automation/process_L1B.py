import logging

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

fh2 = logging.FileHandler(db.paths['logs'] / 'process_L1B_fail.log')
fh2.setLevel(logging.WARNING)
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
        spec = L1A.create(dataid, nfft=250_000, dt=1)
        path = db.paths['output'] / 'L1A' / f'{dataid}.png'
        spec.plot(savepath=path)

        data = L1B.create(spec)
        data.save()
        path = db.paths['output'] / 'L1B' / f'{dataid}.png'
        data.plot(savepath=path, fit_func=False)

        logger.info(f'Extraction of {dataid} succeded')
        db.update_status(dataid, status='success')

    except EmptyRecordingError as e:
        logger.warning(e)
        db.update_status(dataid, status='empty_recording')
    except PassNotFoundError as e:
        logger.info(f'No pass was found in {dataid}: {e}')
        db.update_status(dataid, status='pass_not_found')
    except Exception:
        logger.exception(f'Extraction of {dataid} failed unexpectedly')
        db.update_status(dataid, status='unknown_error')
