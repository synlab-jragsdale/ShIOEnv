#!/bin/bash
#
# install-agent.sh - Install Firecracker VSOCK agent executable and calling script to .ext4 filesystem
#
# Usage:  ./install-agent.sh path/to/filesystem.ext4

set -euo pipefail

[[ $(id -u) -eq 0 ]] || { echo "[!] Error: must run as root." >&2; exit 1; }
[[ $# -eq 1 ]]       || { echo "[!] Usage: $0 <rootfs.ext4>" >&2; exit 1; }
FS_IMAGE="$1"
[[ -f $FS_IMAGE ]]    || { echo "[!] File not found: $FS_IMAGE" >&2; exit 1; }

# sanity check: require .ext4 extension TODO: CHANGE TO FILE SIG
if [[ "${FS_IMAGE##*.}" != "ext4" ]]; then
  echo "[!] Error: filesystem image must have .ext4 extension"
  exit 1
fi

# check files exist
for SRC in scripts/fc-agent/fc-agent scripts/fc-agent/init-agent; do
  if [[ ! -f "$SRC" ]]; then
    echo "[!] Error: source file '$SRC' not found."
    exit 1
  fi
done

# temporary mount point
MNT=$(mktemp -d)
cleanup() {
  echo "[*] Unmounting and cleaning up"
  # ignore errors in cleanup
  umount "$MNT" 2>/dev/null || true
  rm -rf "$MNT"
}
trap cleanup EXIT

echo "[*] Mounting '$FS_IMAGE' at '$MNT'"
sudo mount -o loop,rw "$FS_IMAGE" "$MNT"

sudo mkdir -p "$MNT/usr/sbin"
echo "[*] Copying fc-agent -> /usr/sbin/fc-agent"
sudo cp scripts/fc-agent/fc-agent "$MNT/usr/sbin/fc-agent"
echo "[*] Copying init-agent -> /init-agent"
sudo cp scripts/fc-agent/init-agent "$MNT/init-agent"
echo "[*] Setting execute permissions"
sudo chmod +x "$MNT/usr/sbin/fc-agent" "$MNT/init-agent"

echo "[*] Syncing"
sync

echo "[+] agent and initialization script successfully installed in $FS_IMAGE"
