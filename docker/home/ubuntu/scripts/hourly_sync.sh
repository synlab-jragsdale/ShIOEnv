bash
#!/bin/bash
# Script to synchronize local files with remote server on an hourly basis

# Logging
log_file="/home/ubuntu/logs_archive/hourly_sync.log"
echo "$(date +%Y-%m-%d_%H:%M:%S) - Starting hourly sync." >> "$log_file"

# Source and Destination directories
src_directory="/home/ubuntu/data_snapshots"
dest_directory="ubuntu@remote_server:/backup/data_snapshots"

# Rsync for file synchronization
rsync -avz --delete --exclude 'temp_files/' "$src_directory" "$dest_directory" >> "$log_file" 2>&1

if [[ $? -eq 0 ]]; then
  echo "$(date +%Y-%m-%