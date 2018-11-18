#!/bin/bash

DIR=$(cd $(dirname $0);pwd)

PORTNAME_LIST=$DIR/portnames.txt
DELAY=10

if [ $# -ne 1 ]
then
  echo "ERROR: need one input: port name (one of the values in $PORTNAME_LIST)"
  exit 3
fi

PORTNR=$(grep $1 $PORTNAME_LIST | awk -F':' '{print $1}' | sed 's/Port//' | sed 's/Name//' | sed 's/ //g')

if [ -z "$PORTNR" ]
then
  echo "ERROR: Cannot find port with name $1 in $PORTNAME_LIST"
  exit 3
fi

#set main command
COM="setpower"
#turn this port off
COM+="+p6${PORTNR}=0"
#then turn this port on
COM+="+p6${PORTNR}n=1"
#with a certain delay
COM+="+t6${PORTNR}=$DELAY"

$DIR/pwsw.sh $COM

