#!/bin/bash

# Auto Backup Script
# This script automatically backs up specified directories to a backup location.

# Define source and destination directories
SOURCE_DIR="/home/ubuntu/data"
BACKUP_DIR="/home/ubuntu/backups"

# Create backup directory if it does not exist
if [ ! -d "$BACKUP_DIR" ]; then
  mkdir -p "$BACKUP_DIR"
fi

# Function to perform backup
perform_backup() {
  echo "Starting backup of $SOURCE_DIR to $BACKUP_DIR"
  rsync -av --delete "$SOURCE_DIR/" "$BACKUP_DIR"
  echo "Backup completed successfully."
}

# Logging function
backup_log() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Backup performed" >> /home/ubuntu/logs/backup.log
}

# Perform backup and log the activity
perform_backup
backup_log

echo "Backup script execution finished."