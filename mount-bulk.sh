#!/bin/bash -u

MOUNT_DIR=/home/doptrack/bulk

if [ $# -lt 2 ]
then
  NETID=$LOGNAME
else
  NETID=$2
fi

if [ $# -lt 1 ]
then
  MODE=mount
else
  MODE=$1
fi


#checking if bart messed up things again
ls $MOUNT_DIR > /dev/null || sudo umount -l $MOUNT_DIR

[ ! -d $MOUNT_DIR ] && mkdir -pv $MOUNT_DIR

case $MODE in
  mount|m)
    if [ -z "$(ls $MOUNT_DIR)" ]
    then
      sshfs $NETID@sftp.tudelft.nl:/staff-bulk/lr/spe/as $MOUNT_DIR -o cache=no -o allow_other
    else
      echo "WARNING: $MOUNT_DIR is not empty, possible bulk is already mounted"
    fi
  ;;
  umount|u)
    if [ ! -z "$(ls $MOUNT_DIR)" ]
    then
      fusermount -u $MOUNT_DIR
    else
      echo "WARNING: $MOUNT_DIR is already empty, possible bulk is already unmounted"
    fi
  ;;
  *)
    echo "ERROR: unknown mode $MODE"
  ;;
esac


