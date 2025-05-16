bash
#!/bin/bash

# Daily Cleanup Script
# This script removes temporary files and cleans up log files daily.

# Define the path for logs and temp files
LOG_DIR="/var/log"
TEMP_DIR="/tmp"

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf ${TEMP_DIR}/*

# Rotate logs
echo "Rotating logs..."
logrotate /etc/logrotate.conf

# Clear old logs
echo "Removing old logs..."
find ${LOG_DIR} -type f -name '*.log' -mtime +10 -exec rm {} \;

# Echo completion message
echo "Daily cleanup completed successfully!"

# Record the action
echo "`date`: Daily cleanup executed" >> /home/ubuntu/scripts/cleanup_log.log
