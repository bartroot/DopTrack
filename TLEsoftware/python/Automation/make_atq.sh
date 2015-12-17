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
max_prio=$(cat /home/doptrack/rec.list | wc -l)

# get all the meta files
ls $LOC_PEN | grep ".yml" > pending.list
ls $LOC_ARM | grep ".yml" > armed.list
grep -vf pending.list armed.list > old.list
grep -vf armed.list pending.list > new.list
grep -f pending.list armed.list > double.list

# add all the new metafiles to the armed list
numl=$(cat new.list | wc -l)
echo "Amount of new recordings: $numl"

for i in `seq 1 $numl`;
do
        line=$(tail -n+$i new.list | head -n1)
        mv $LOC_PEN$line $LOC_ARM$line
done

# with doubles check the time of TLE update and remove oldest
numl=$(cat double.list | wc -l)
echo "Amount of double recordings: $numl"

for i in `seq 1 $numl`;
do
        line=$(tail -n+$i double.list | head -n1)
        ntime=$(cat $LOC_PEN$line | grep "time used UTC" | awk '{print $4}')
        otime=$(cat $LOC_ARM$line | grep "time used UTC" | awk '{print $4}')
        # if new file has improved TLE time
        if [ "$ntime" -gt "$otime" ]; then
	    mv $LOC_PEN$line $LOC_ARM$line

	elif [ "$ntime" -eq "$otime" ]; then
	    mv $LOC_PEN$line $LOC_ARM$line
        else
            rm $LOC_PEN$line
        fi 
done

for p in `seq 1 $max_prio`
do      
        # update armed.list
	rm armed.list
	ls $LOC_ARM | grep ".yml" > armed.list
	echo "Priority check: $p"

	# check for priority
	numl=$(cat armed.list | wc -l)

	for i in `seq 1 $numl`;
	do
        	record=$(tail -n+$i armed.list | head -n1)
		if [ -f $LOC_ARM$record ];then
        		prio=$(cat $LOC_ARM$record | grep "Priority" | awk '{print $2}')
                	if [ "$prio" -eq "$p" ]; then
        			start_rec=$(cat $LOC_ARM$record | grep "Start of recording" | awk '{print $4}')
        			year=$(echo $start_rec | cut -c1-4)
        			month=$(echo $start_rec | cut -c5-6)
        			day=$(echo $start_rec | cut -c7-8)
        			hour=$(echo $start_rec | cut -c9-10) 
        			minute=$(echo $start_rec | cut -c11-12)
       				lofp=$(cat $LOC_ARM$record | grep "Length of pass" | awk '{print $4}')
        			end_rec=$(date -d "${year}-${month}-${day} ${hour}:${minute} $lofp seconds" +%Y%m%d%H%M)
        			# make selection that overlaps the selected recording
        			for a in `seq 1 $numl`
        			do
              				rec_test=$(tail -n+$a armed.list | head -n1)
              				if ! [ $rec_test == $record ]; then
                   				# not the same record, so check if overlapping
  						if [ -f $LOC_ARM$rec_test ];then
                   					start_test=$(cat $LOC_ARM$rec_test | grep "Start of recording" | awk '{print $4}') 
							prio_test=$(cat $LOC_ARM$rec_test | grep "Priority" | awk '{print $2}')
                  					if  [ "$start_rec" -lt "$start_test" -a "$start_test" -lt "$end_rec" ]; then
                     						# recordings are overlapping
								if [ "$prio_test" -lt "$prio" ]; then
									echo "Overlapping file is removed: $LOC_ARM$rec_test"
                      							rm $LOC_ARM$rec_test
                                                                elif [ "$prio_test" -eq "$prio" ]; then
									# similar recordings but off with one or two minutes
									# check the newest TLE propagation
									ntime=$(cat $LOC_ARM$rec_test | grep "time used UTC" | awk '{print $4}')
								        otime=$(cat $LOC_ARM$record | grep "time used UTC" | awk '{print $4}')
								        # if new file has improved TLE time
								        if [ "$ntime" -gt "$otime" ]; then
									    # don't do anything. In the following i loop the record will be removed
									    otime=$otime	
								        elif [ "$ntime" -eq "$otime" ]; then
									    echo "Old file is removed: $LOC_ARM$rec_test"
								            rm $LOC_ARM$rec_test
      								        else
								            echo "Old file is removed: $LOC_ARM$rec_test"
           								    rm $LOC_ARM$rec_test
       									fi
								fi
                  					fi
						fi
              				fi

        			done
			fi
		fi
	done
done

# update armed.list
rm armed.list
ls $LOC_ARM | grep ".yml" > armed.list

# remove old atq list
numj=$(cat armed.list | wc -l)
if [ $numj > 0 ]; then
	# now remove all pending at jobs, such that the new jobs are refreshed
	for j in `atq | awk '{print $1}'`; do atrm $j;done
fi

# update the atq list
for t in `seq 1 $numj`
do
	# set the new Recording of Doptrack
        line=$(tail -n+$t armed.list | head -n1) 
        Stime=$(cat $LOC_ARM$line | grep "Start of recording" | awk '{print $4}') 
        #echo "python Record.py -i $line at time: $Stime"  
        echo "python Record.py -i $line" | at -t $Stime
done

rm pending.list
rm armed.list
rm old.list
rm new.list
rm double.list
