python
import os
import shutil
import datetime

def backup_directory(source, destination):
    """
    This function creates a backup of a directory by copying it into a specified destination.
    :param source: str - Path to the source directory
    :param destination: str - Path to the backup directory
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    backup_path = os.path.join(destination, f"backup_{formatted_time}")

    try:
        shutil.copytree(source, backup_path)
        print(f"Backup successful! Directory {source} has been backed up to {backup_path}")
    except Exception as e:
        print(f"Error during backup: {e}")

if __name__ == "__main__":
    source_path = "/home/ubuntu/data/"
    dest_path = "/home/ubuntu/backups/"
    backup_directory(source_path, dest_path)
