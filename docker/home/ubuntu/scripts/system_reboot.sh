bash
#!/bin/bash

# Script to safely reboot the system

echo "Initiating system reboot process..."
sync    # Synchronize cached writes to persistent storage

# Check if there are any critical processes that should not be interrupted
if pgrep -x "critical_process" > /dev/null
then
    echo "Critical process is running, aborting reboot."
    exit 1
fi

# Log the reboot initiation
echo "$(date) - System reboot initiated by user" | tee -a /home/ubuntu/logs_archive/reboot.log

# Notify users about the reboot
wall "System will reboot in 1 minute. Please save your work."

# Wait for a minute to give users time to save their work
sleep 60

# Reboot the system
echo "Rebooting now..."
/sbin/shutdown -r now "System reboot initiated by maintenance script."

exit 0
