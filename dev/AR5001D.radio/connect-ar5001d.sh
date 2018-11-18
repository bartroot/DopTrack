#!/bin/bash -u

TTY=/dev/ttyUSB0

#echo "Connecting to $TTY..."
#echo "Use Ctr-A K to kill screen (as usual)."
#sudo screen $TTY 115200

minicom -D $TTY
