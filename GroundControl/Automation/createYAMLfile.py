#/usr/bin/python
#

import yaml

def make():
   data = {}
   data = {'Sat':{'State':{'Name': {}}, 'Predict':{}, 'Record':{}, 'Station':{}}}

   # input the State variables in the meta-structure
   data['Sat']['State']['Name'] = ''
   data['Sat']['State']['NORADID'] = ''
   data['Sat']['State']['Tuning Frequency'] = ''
   data['Sat']['State']['Antenna'] = ''
   data['Sat']['State']['Priority'] = ''

   # input the prediction variables into the meta-structure
   data['Sat']['Predict']['used TLE line1'] = ''
   data['Sat']['Predict']['used TLE line2'] = ''
   data['Sat']['Predict']['time used UTC'] = ''
   data['Sat']['Predict']['timezone used'] = ''
   data['Sat']['Predict']['Elevation'] = ''
   data['Sat']['Predict']['SAzimuth'] = ''
   data['Sat']['Predict']['EAzimuth'] = ''
   data['Sat']['Predict']['Length of pass'] = ''

   # input the recording variables in the meta-structure
   data['Sat']['Record']['sample_rate'] = ''
   data['Sat']['Record']['num_sample'] = ''
   data['Sat']['Record']['time1 UTC'] = ''
   data['Sat']['Record']['time2 UTC'] = ''
   data['Sat']['Record']['time3 LT'] = ''
   data['Sat']['Record']['Start of recording'] = ''

   # input for recording station
   data['Sat']['Station']['Name'] = ''
   data['Sat']['Station']['Lat'] = ''
   data['Sat']['Station']['Lon'] = ''
   data['Sat']['Station']['Height'] = ''

   with open('empty.yml', 'w') as outfile:
       outfile.write( yaml.dump(data, default_flow_style=False) )

if __name__ == "__main__":
   make()
