#!/bin/bash

DIR=$( cd $( dirname $0);pwd)

LOCK=$DIR/lock

if [ -e $DIR/lock ]
then
  kill `lsof -i :8080 | tail -n +2 | awk '{print $2}' | uniq`
  killall uhd_rx_cfile
  rm -f $LOCK
else
  echo "ERROR: cannot find lock file $DIR/lock"
  exit 3
fi

