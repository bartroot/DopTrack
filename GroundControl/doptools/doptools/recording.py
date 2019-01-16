from datetime import timedelta

from .io import read_meta


class Recording:

    def __init__(self, dataid):
        meta = read_meta(dataid)
        self.duration = int(meta['Sat']['Predict']['Length of pass'])
        self.start_time = meta['Sat']['Record']['time1 UTC']
        self.stop_time = self.start_time + timedelta(seconds=self.duration)
        self.sample_freq = int(meta['Sat']['Record']['sample_rate'])
        self.tuning_freq = int(meta['Sat']['State']['Tuning Frequency'])
        self.tle = [meta['Sat']['Predict']['used TLE line1'],
                    meta['Sat']['Predict']['used TLE line2']]
