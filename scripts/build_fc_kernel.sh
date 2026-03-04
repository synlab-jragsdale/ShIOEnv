#!/bin/bash
# build_fc_kernel.sh - Build a Firecracker-compatible x86_64 kernel
set -Eeuo pipefail

print_help() {
  cat <<'EOF'
build_fc_kernel.sh - Build a Firecracker-compatible x86_64 kernel

Usage:
  ./build_fc_kernel.sh [--help|-h]

Examples:
  ./build_fc_kernel.sh                 # build latest stable (default linux-X.Y.y)
  KVER=v5.10.224 ./build_fc_kernel.sh  # build a specific tag/branch
  SRC_DIR=$HOME/linux KVER=v6.1 ./build_fc_kernel.sh

Environment variables you can override:
  KVER       - tag/branch/commit to fetch or use
               (default: latest stable branch, e.g. linux-X.Y.y)
  SRC_DIR    - where the kernel source lives
               (default: $PWD/linux-$KVER)
  BUILD_DIR  - out-of-tree build dir
               (default: $SRC_DIR/build)
  OUTPUT_DIR - where vmlinux is copied
               (default: $PWD/fc_src/kernel)
  THREADS    - make -j value
               (default: nproc)
  KEEP_MODS  - if set (non-empty), keep modules instead of forcing built-ins

Notes:
  - Requires standard kernel build deps (git, make, gcc, etc.)
  - Uses a KVM guest defconfig baseline plus Firecracker-friendly tweaks.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

KVER="${KVER:-}"  # empty means: build the tip of "stable" branch
SRC_DIR="${SRC_DIR:-}"  # if empty, will be derived below
BUILD_DIR="${BUILD_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/fc_src/kernel}"
THREADS="${THREADS:-$(nproc)}"

GIT_URL="https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git"

# Pick a default version (HEAD of stable) if caller did not specify
if [[ -z "$KVER" ]]; then
  KVER="$(git ls-remote --heads "$GIT_URL" | grep -Eo "refs/heads/linux-[0-9]+\.[0-9]+\.y" | sort -V | tail -1 | sed 's#refs/heads/##')"
  echo "[*] No KVER specified - using latest stable: $KVER"
fi

# Derive directories when not provided
[[ -z "$SRC_DIR" ]]   && SRC_DIR="$PWD/linux-${KVER//\//_}"
[[ -z "$BUILD_DIR" ]] && BUILD_DIR="$SRC_DIR/build"

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "[*] Cloning kernel $KVER -> $SRC_DIR"
  git clone --depth 1 --branch "$KVER" "$GIT_URL" "$SRC_DIR"
fi

pushd "$SRC_DIR" >/dev/null

# Older LTS branches (e.g. 5.10) keep the file under a different name. fall back gracefully
if [[ -f "arch/x86/configs/kvm_guest_defconfig" ]]; then
  BASE_DEFCONFIG="kvm_guest_defconfig"
elif [[ -f "arch/x86/configs/x86_64_kvm_guest_defconfig" ]]; then
  BASE_DEFCONFIG="x86_64_kvm_guest_defconfig"
else
  BASE_DEFCONFIG="x86_64_defconfig"   # always present
fi

echo "[*] Using base defconfig: $BASE_DEFCONFIG"
make O="$BUILD_DIR" ARCH=x86_64 "$BASE_DEFCONFIG"

# Append/override Firecracker-specific options
cat >"$BUILD_DIR/firecracker.cfg" <<'EOF'
# +----------------------------------------------------------------------------+
# | Firecracker microVM guest additions - x86_64                               |
# +----------------------------------------------------------------------------+
# Serial console for easy debugging via ttyS0
CONFIG_SERIAL_8250=y
CONFIG_SERIAL_8250_CONSOLE=y
CONFIG_PRINTK=y

# Fast / trusted entropy
CONFIG_HW_RANDOM_VIRTIO=y
CONFIG_RANDOM_TRUST_CPU=y

# VirtIO MMIO (Firecracker exposes devices that way)
CONFIG_VIRTIO=y
CONFIG_VIRTIO_MMIO=y
CONFIG_VIRTIO_MMIO_CMDLINE_DEVICES=n  # legacy path not needed
CONFIG_VIRTIO_NET=y
CONFIG_VIRTIO_BLK=y
CONFIG_VIRTIO_BALLOON=y
CONFIG_VSOCKETS=y
CONFIG_VIRTIO_VSOCKETS=y

# Block-device helpers (boot from /dev/vda)
CONFIG_BLK_DEV_INITRD=y
CONFIG_MSDOS_PARTITION=y

# OverlayFS & tmpfs/devtmpfs (for early userspace)
CONFIG_OVERLAY_FS=y
CONFIG_TMPFS=y
CONFIG_DEVTMPFS=y
CONFIG_DEVTMPFS_MOUNT=y

# Ensure ext4 root works without an initramfs
CONFIG_EXT4_FS=y
CONFIG_EXT4_FS_POSIX_ACL=y

# High-precision guest time & graceful shutdown helpers
CONFIG_KVM_GUEST=y
CONFIG_PTP_1588_CLOCK=y
CONFIG_PTP_1588_CLOCK_KVM=y
CONFIG_SERIO_I8042=y
CONFIG_KEYBOARD_ATKBD=y

# ACPI path (Firecracker >= 1.5)
CONFIG_ACPI=y
CONFIG_PCI=y  # required by ACPI initialisation, FC still has no PCI devices

# +----------------------------------------------------------------------------+
# | Netfilter / nftables so iptables 1.8+ (nft backend) works inside the guest |
# +----------------------------------------------------------------------------+

CONFIG_NF_TABLES_IPV4=y
CONFIG_NF_TABLES_IPV6=y
CONFIG_NF_TABLES_INET=y
CONFIG_NF_CONNTRACK=y
CONFIG_NETFILTER_XT_MATCH_TCP=y
CONFIG_NETFILTER_XT_MATCH_UDP=y
CONFIG_NETFILTER_XT_TARGET_LOG=y

CONFIG_NETFILTER=y
CONFIG_NETFILTER_ADVANCED=y
CONFIG_NF_TABLES=y
CONFIG_NFT_CT=y
CONFIG_NFT_COUNTER=y
CONFIG_NFT_LOG=y
CONFIG_NFT_NAT=y
CONFIG_NFT_MASQ=y
CONFIG_NFT_REDIR=y
CONFIG_NFT_CHAIN_NAT=y
CONFIG_NETFILTER_XTABLES=y  # legacy iptables backend
CONFIG_IP_NF_IPTABLES=y
CONFIG_IP_NF_FILTER=y
CONFIG_IP_NF_NAT=y
CONFIG_IP_NF_TARGET_MASQUERADE=y
EOF

# If don't keep modules, force everything above to built-in (=y)
if [[ -z "${KEEP_MODS:-}" ]]; then
  sed -i 's/=m$/=y/' "$BUILD_DIR/firecracker.cfg"
fi

# Merge fragment and accept defaults
scripts/kconfig/merge_config.sh -m -O "$BUILD_DIR" "$BUILD_DIR/.config" "$BUILD_DIR/firecracker.cfg"
make O="$BUILD_DIR" ARCH=x86_64 olddefconfig
make -C "$SRC_DIR" O="$BUILD_DIR" ARCH=x86_64 -j"$THREADS" vmlinux
cp "$BUILD_DIR/vmlinux" "$OUTPUT_DIR/vmlinux-${KVER//\//_}"
echo "[*] Firecracker-ready kernel saved to $OUTPUT_DIR/vmlinux-${KVER//\//_}"

popd >/dev/null
