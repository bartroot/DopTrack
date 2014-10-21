#!/bin/bash -u

TTY=/dev/ttyS0

echo "Connecting to $TTY..."
echo "Use Ctr-A K to kill screen (as usual)."

sudo screen /dev/ttyS0 9600
