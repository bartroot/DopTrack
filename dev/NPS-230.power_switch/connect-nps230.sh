#!/bin/bash -u


if [[ ! "${@//-serial/}" == "$@" ]]
then
  TTY=/dev/ttyS0
  echo "Use Ctr-A K to kill screen (as usual)."
  echo "Connecting to $TTY..."
  sudo screen $TTY 9600
else
  IP=192.168.1.2
  echo "Connecting to $IP..."
  telnet 192.168.1.2
fi

