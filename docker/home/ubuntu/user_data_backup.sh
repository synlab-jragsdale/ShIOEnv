bash
#!/bin/bash

# Script to backup user data from specified directories

# Define backup directory
backup_dir="/home/ubuntu/backups/user_data"

# Log start
echo "Starting user data backup at $(date)" >> /home/ubuntu/logs_archive/user_backup.log

# Create backup directory if it doesn't exist
mkdir -p $backup_dir

# Copying files
cp -a /home/ubuntu/data/*.json $backup_dir
cp -a /home/ubuntu/data/*.xml $backup_dir

# Compress backups
tar -czf $backup_dir/backup_$(date +%Y%m%d_%H%M%S).tar.gz $backup_dir

# Cleanup uncompressed backup files
rm -rf $backup_dir/*.json
rm -rf $backup_dir/*.xml

# Log completion
echo "User data backup completed at $(date)" >> /home/ubuntu/logs_archive/user_backup.log

