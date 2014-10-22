#!/bin/bash -u

TTY=/dev/ttyS0

if [[ ! "${@//-serial/}" == "$@" ]]
then
  echo "Connecting to $TTY..."
  echo "Use Ctr-A K to kill screen (as usual)."
  sudo screen /dev/ttyS0 9600
else
  telnet 192.168.1.2
fi

