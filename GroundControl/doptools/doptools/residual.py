import numpy as np
import pandas as pd

from doptools.io import read_meta, read_rre
from doptools.estimate import propogate_tle
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84




def analyse_residual(data_id):
    meta = read_meta(data_id)
    rre = read_rre(data_id)
    tle = [meta['Sat']['Predict']['used TLE line1'],
           meta['Sat']['Predict']['used TLE line2']]
    propogate_tle(tle, times)
    satellite = twoline2rv(tle[0], tle[1], wgs84)



    return pd.DataFrame(data)