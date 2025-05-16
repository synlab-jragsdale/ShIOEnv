bash
#!/bin/bash

# Script to monitor disk usage

# Send alert if disk usage exceeds 90%
ALERT_LEVEL=90
OUTPUT_FILE="/home/ubuntu/logs/disk_usage.log"

df -H | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{ print $5 " " $1 }' | while read output;
do
  usage=$(echo $output | awk '{ print $1}' | cut -d'%' -f1 )
  partition=$(echo $output | awk '{ print $2 }' )
  if [ $usage -ge $ALERT_LEVEL ]; then
    echo "Running out of space \"$partition ($usage%)\" on $(hostname) as on $(date)" >> $OUTPUT_FILE
    # Uncomment to enable email alert
    # mail -s "Alert: Almost out of disk space $usage%" user@example.com
  fi
done
