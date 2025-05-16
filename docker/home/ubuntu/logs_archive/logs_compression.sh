bash
#!/bin/bash

# Logs Compression Script
# Compresses all log files older than a week in the /var/log directory

LOG_DIR="/var/log"
ARCHIVE_DIR="/home/ubuntu/logs_archive"

# Finding all log files older than 7 days
echo "Finding all log files from $LOG_DIR older than 7 days..."
old_logs=$(find $LOG_DIR -name "*.log" -type f -mtime +7)

# Check if there are any files to compress
if [ -z "$old_logs" ]; then
    echo "No old log files to compress."
    exit 0
fi

# Compressing and moving the old logs
echo "Compressing and moving the old log files..."
for log in $old_logs; do
    tar -zcf $ARCHIVE_DIR/$(basename "$log").tar.gz -P $log && rm -f $log
    echo "$log compressed and moved to $ARCHIVE_DIR"
done

echo "Logs compression completed."
