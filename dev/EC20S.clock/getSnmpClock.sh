#!/bin/bash -u

snmpgetnext -v 2c -c public 192.168.1.5 EC20S-MIB::ec20s-$1
