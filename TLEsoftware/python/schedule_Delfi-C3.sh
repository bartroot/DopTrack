#!/bin/bash
#

# first retrieve the TLE file from spcaetrack.org website and construct new TLE.txt file

./getTLE.sh

# run the prediction software, which constucts the prediction_Delfi-C3.txt file

python predictDelfi-C3.py

# make a new atq run file

python set_QLine_at.py

# now remove all pending at jobs, such that the new jobs are refreshed

for i in `atq | awk '{printf "$1"}'`; do atrm $1;done

# run Qline file

chmod a+x atq_Delfi-C3.sh
./atq_Delfi-C3.sh

