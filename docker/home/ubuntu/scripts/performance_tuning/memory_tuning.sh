bash
#!/bin/bash

# Script to tune memory usage on the server

echo "Starting memory tuning process."

# Free up page cache, dentries and inodes
sync; echo 1 > /proc/sys/vm/drop_caches

# Adjust the swappiness value to improve memory management:
# Swappiness parameter controls the relative weight given to swapping out runtime memory,
# as opposed to dropping pages from the system page cache.
# A lower value will avoid swapping processes out of physical memory for as long as possible
echo "Setting swappiness to 10."
sysctl vm.swappiness=10

# Display current memory usage
echo "Current memory usage:"
free -h

echo "Memory tuning completed successfully."
