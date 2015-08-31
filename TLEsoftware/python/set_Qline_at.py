# This function sets the at log que-line from the prediction logfile
#
# input is the prediction file made by predictDelfi-C3.py
#
#--------------- Start function -------------------------------------
import datetime

tnow = datetime.datetime.now()
year = tnow.year
# input file
fname = 'prediction_Delfi-C3.txt'

fin = open(fname,'r')
fout = open('atq_Delfi-C3.sh','w')

# write heading of shell script
fout.write("#!/bin/bash\n")
fout.write("#\n")

# Start reading the text file
for fline in fin.readlines():
	# get the acquired variables from the line
	day = int(fline[3:5])
	month = int(fline[6:8])
	hour = int(fline[9:11])
	minute = int(fline[12:14])
	elevation = int(fline[25:27])

	# check if elevation is above a certain treshold
	if elevation > 10 :
		# do the calculations
		if minute == 0 :
			minute = 59
			hour = hour - 1
		else :
			minute = minute - 1

		# print 
		fout.write('echo "uhd_rx_cfile -a "addr=192.168.10.1" -f 45.07M --samp-rate=250k -N 225000000 DelfiC3_145.870M_%02i%02i%04i_%02i%02iLT.32fc" | at -t %04i%02i%02i%02i%02i\n' % (day,month,year,hour,minute,year,month,day,hour,minute) )
	else : 
		# do nothing
		elevation = elevation


print "End of for-loop!"
fin.close()
fout.close()





