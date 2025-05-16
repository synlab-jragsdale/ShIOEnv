bash
#!/bin/bash
# Monthly Backup Script
# This script backs up user data and system configurations.

# Define the backup directories
USER_DATA="/home/ubuntu/data_snapshots/"
CONFIG_DIR="/home/ubuntu/"
BACKUP_DEST="/home/ubuntu/archives/"

# Get current date in yyyy-mm-dd format
CURRENT_DATE=$(date +"%Y-%m-%d")

# Create a filename with datestamp for our backup.
BACKUP_NAME="backup_$CURRENT_DATE.tar.gz"

# Print start status message.
echo "Starting backup at $(date)"
echo "Backing up to $BACKUP_DEST$BACKUP_NAME"

# Backup the directories into a tar file.
tar czf $BACKUP_DEST$BACKUP_NAME $USER_DATA $CONFIG_DIR

# Print end status message.
echo "Backup completed at $(date)"
echo "Backup file created: $BACKUP_DEST$BACKUP_NAME"

# Long listing of files in $dest to check file sizes and file names.
ls -lh $BACKUP_DEST
