#!/bin/bash

SERVER=checkip.dyndns.org

ping -c 1 -i 1 $SERVER >& /dev/null || exit $?

wget  -q -O - $SERVER | sed -e 's/.*Current IP Address: //' -e 's/<.*$//'