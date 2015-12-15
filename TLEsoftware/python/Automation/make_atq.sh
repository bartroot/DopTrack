#!/bin/bash
#
# This script will check the pending meta files and arm them according to a priority setting
#
# Development log:
#			- 15-11-2015, Bart Root: Initial development
#
#--------------------------------------------------------------------------

LOC_PEN='REC_PENDING/'
LOC_ARM='REC_ARMED/'
LOC_REC='/media/data/'

ls $LOC_PEN | grep ".yml" > pending.list
ls $LOC_ARM | grep ".yml" > armed.list
grep -vf pending.list armed.list > old.list
grep -vf armed.list pending.list > new.list
grep -f pending.list armed.list > double.list

numl=$(cat pending.list | wc -l)

for i in `seq 1 $numl`;
do
        line=$(tail -n+$i pending.list | head -n1)
        echo $line
done



rm pending.list
rm armed.list
rm old.list
rm new.list
rm double.list
