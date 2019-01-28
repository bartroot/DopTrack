from datetime import timedelta

from doptools.io import read_meta


class Recording:

    def __init__(self, dataid):
        meta = read_meta(dataid)
        self.dataid = dataid
        self.duration = int(meta['Sat']['Predict']['Length of pass'])
        self.n_samples = int(meta['Sat']['Record']['num_sample'])
        self.start_time = meta['Sat']['Record']['time1 UTC']
        self.start_time_local = meta['Sat']['Record']['time3 LT']
        self.stop_time = self.start_time + timedelta(seconds=self.duration)
        self.sample_freq = int(meta['Sat']['Record']['sample_rate'])
        self.tuning_freq = int(meta['Sat']['State']['Tuning Frequency'])

        self.prediction = {
                'tle': (meta['Sat']['Predict']['used TLE line1'],
                        meta['Sat']['Predict']['used TLE line2']),
                'max_elevation': meta['Sat']['Predict']['Elevation'],
                'start_azimuth': meta['Sat']['Predict']['SAzimuth']}

        self.station = {
                'name': meta['Sat']['Station']['Name'],
                'latitude': meta['Sat']['Station']['Lat'],
                'longitude': meta['Sat']['Station']['Lon'],
                'altitude': meta['Sat']['Station']['Height']}
