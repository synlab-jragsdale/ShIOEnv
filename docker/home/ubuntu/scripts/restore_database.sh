bash
#!/bin/bash

# Script to restore database from the latest backup
BACKUP_PATH="/home/ubuntu/data_snapshots"
DATABASE_NAME="MyDatabase"

echo "Starting database restoration for $DATABASE_NAME..."

# Locate latest backup file
LATEST_BACKUP=$(ls -Art $BACKUP_PATH | tail -n 1)
if [ -z "$LATEST_BACKUP" ]; then
    echo "Error: No backup file found in $BACKUP_PATH"
    exit 1
fi

echo "Restoring from $LATEST_BACKUP..."
restore_command="mysql -u root -pYourPassword $DATABASE_NAME < $BACKUP_PATH/$LATEST_BACKUP"

# Execute restore command
eval $restore_command

if [ $? -eq 0 ]; then
    echo "Database restoration successful."
else
    echo "Error in database restoration."
fi

echo "Database restore process completed."
