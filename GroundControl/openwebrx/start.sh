#!/bin/bash

DIR=$( cd $( dirname $0);pwd)
LOCK=$DIR/lock

if [ -e $LOCK ]
then
  echo "ERROR: cannot start openwebrx because it is aldreay running. Run $DIR/stop.sh before starting openwebrx with this script."
  exit 3
fi

#get radio status
FREQ=$( ../../dev/AR5001D.radio/command-ar5001d.rb CF)
 ANT=$( ../../dev/AR5001D.radio/command-ar5001d.rb AN)
#clean up radio output
FREQ=${FREQ::-2}
FREQ=${FREQ/CF0/}
ANT=${ANT::-2}
echo "The radio is tunned at $FREQ Hz and using antenna $ANT"

cat $DIR/config_webrx.py.template | sed ''s/ANTENNA_ANCHOR/''$ANT''/'' | sed ''s/CENTER_FREQUENCY_ANCHOR/''$FREQ''/'' > $DIR/github/config_webrx.py
cd $DIR/github
./openwebrx.py > $DIR/log 2>&1 &
touch $LOCK
cd - > /dev/null

