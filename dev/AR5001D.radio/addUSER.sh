#!/bin/bash

if [ $# -ge 2 ]
then
	TARGET_GROUP=$2
else
	TARGET_GROUP=dialout
fi

if [ $# -ge 1 ]
then
	USER_NOW=$1
else
	USER_NOW=$USER
fi

echo Trying to add user $USER_NOW to group $TARGET_GROUP...

#check if user exists
awk -F':' '{print $1}' /etc/passwd | grep $USER_NOW > /dev/null
if [ $? -ne 0 ]
then
	echo ERROR: unknown user $USER_NOW
	exit 3
fi

#check if target group exists
awk -F':' '{print $1}' /etc/group | grep $TARGET_GROUP > /dev/null
if [ $? -ne 0 ]
then
	echo ERROR: unknown group $TARGET_GROUP
	exit 3
fi

#check if user is already part of group
GROUP_LIST=("$(groups $USER_NOW)")
for i in $GROUP_LIST
do
	if [ "$i" == "$TARGET_GROUP" ]
	then
		echo User $USER_NOW already part of group $TARGET_GROUP.
		exit 0
	fi
done

#add user to target group
sudo useradd -G $TARGET_GROUP $USER_NOW

