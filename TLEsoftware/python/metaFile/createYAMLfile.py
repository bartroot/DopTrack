#/usr/bin/python
#

import yaml

#f = open('metaYAML.yaml')

data = {}
data = {'Sat':{'State':{'Name': {}}, 'Predict':{}, 'Record':{}}}

# input the State variables in the meta-structure
data['Sat']['State']['Name'] = ''
data['Sat']['State']['NORADID'] = ''
data['Sat']['State']['Tuning Frequency'] = ''
data['Sat']['State']['Antenna'] = ''
data['Sat']['State']['Priority'] = ''

# input the prediction variables into the meta-structure
data['Sat']['Predict']['used TLE'] = ''
data['Sat']['Predict']['time used'] = ''
data['Sat']['Predict']['Elevation'] = ''
data['Sat']['Predict']['SAzimuth'] = ''
data['Sat']['Predict']['EAzimuth'] = ''
data['Sat']['Predict']['Length of pass'] = ''

# input the recording variables in the meta-structure
data['Sat']['Record']['sample_rate'] = ''
data['Sat']['Record']['num_sample'] = ''
data['Sat']['Record']['time1'] = ''
data['Sat']['Record']['time2'] = ''
data['Sat']['Record']['time3'] = ''
data['Sat']['Record']['Start of recording'] = ''

with open('empty.yml', 'w') as outfile:
    outfile.write( yaml.dump(data, default_flow_style=False) )


