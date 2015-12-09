#!/usr/bin/python
#
# This function extracts data from the meta-file and starts a recording using the USRP
#
# Development log:
#
#	- Bart Root, 09-12-2015: Initial development
#
########################################################################

# import libraries

import sys, getopt

# initialisation of global variables

LOC_REC = '/media/data/'
LOC_ARM = '/REC_ARMED/'

def main(argv):
   inputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg

   # load meta-file
   with open(inputfile, 'r') as metaf:
        meta = yaml.load(fmeta)  

   # set the parameters for the recording


if __name__ == "__main__":
    main(sys.argv[1:])
