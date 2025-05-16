#!/bin/bash

# Automated Cleanup Script
# This script will free up memory and remove temporary files
echo "Starting system cleanup..."

# Clearing cache
echo "Clearing cache files..."
sudo rm -rf /var/cache/*

# Deleting temporary files older than 7 days
echo "Deleting temporary files that are older than 7 days..."
find /tmp -type f -mtime +7 -exec rm {} \;

# Empty the trash
echo "Emptying the trash directories..."
rm -rf /home/*/Trash/*

# Remove old logs
echo "Removing old log files..."
sudo find /var/log -type f -name "*.log" -mtime +14 -exec rm {} \;

echo "System cleanup completed successfully."

exit 0