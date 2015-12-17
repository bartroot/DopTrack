#!/bin/bash
#

# create data log file
#ls /media/data/ | grep Delfi | cut -c 1-32 > tmpDelfi.txt
#ls /media/data/ | grep NOAA | cut -c 1-33 > tmpDelfi.txt

ls /media/data/ | grep Delfi | grep ".32fc" | cut -c 1-27 | sort > test1
ls /home/doptrack/www/archive_all/ | grep Delfi | cut -c 1-27 | sort > test2

grep -vf test2 test1 > tmpDelfi.txt
rm test1 test2

cat tmpDelfi.txt

numl=$(cat tmpDelfi.txt | wc -l)

for i in `seq 1 $numl`;
do
        line=$(tail -n+$i tmpDelfi.txt | head -n1)
	dir=$(echo "/media/data/")
	line_in=$(echo $dir$line".32fc")
        line_m=$(echo $dir$line".yml")
        line_out=$(echo $line"_zoomedin.png")

        python spectrogram.py -i $line_in -o $line_out -m $line_m
        #python spectrogram_NOAA.py -i $line_in -o $line_out
done


echo "Resampling set completed!"

