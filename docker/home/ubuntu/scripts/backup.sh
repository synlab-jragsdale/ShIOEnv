bash
#!/bin/bash
# Script to backup important user data and configuration files

# Define backup directory
backup_dir="/home/ubuntu/backups"

# Create backup directory if it does not exist
mkdir -p "$backup_dir"

# Define the files and directories to backup
items_to_backup=(
    "/home/ubuntu/config_settings.json"
    "/home/ubuntu/user_data"
    "/home/ubuntu/system_info.txt"
    "/var/log/system_logs"
)

# Timestamp for backup naming
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
backup_file="$backup_dir/backup_$timestamp.tar.gz"

# Creating the backup
tar -czf "$backup_file" "${items_to_backup[@]}"

# Report completion
echo "Backup completed successfully. File saved as $backup_file"
