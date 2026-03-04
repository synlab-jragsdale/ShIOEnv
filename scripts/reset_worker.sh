#!/bin/sh
pkill -f run_collection
killall firecracker
rm -r /tmp/fc-*
rm nohup.out
