#!/bin/bash
#
snmpgetnext -v 2c -c public 192.168.1.5 EC20S-MIB::ec20s-STA-DATETIME
#snmpgetnext -v 2c -c public 192.168.1.5 EC20S-MIB::ec20s-STA-TIMEREF
#snmpgetnext -v 2c -c public 192.168.1.5 EC20S-MIB::ec20s-STA-1PPSOUT
