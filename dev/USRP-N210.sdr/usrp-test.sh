#!/bin/bash

uhd_rx_cfile -v -f 45.07M -samp-rate=250k -N 500k test.out

if [ $? -ne 0 ]
then
	echo "ERROR: non-zero status returned from uhd_rx_cfile"
	exit 3
fi

if [ ! -e test.out ]
then
	echo "ERROR: cannot find recording file"
	exit 2
else
	rm -fv test.out
fi
echo "uhd_rx_cfile working as expected"
