#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: sudo $0 -o <uplink-interface> [-n <count>] [-u <owner>] [-b <base-ip>]"
  echo "  -o  Uplink interface to NAT out of (e.g., eno1, eth0)      [required]"
  echo "  -n  Number of taps to create                               [default: 32]"
  echo "  -u  Owner username for the TAP devices                     [default: \$SUDO_USER or \$USER]"
  echo "  -b  Base host IP to start from (first usable of a /30)     [default: 172.16.0.1]"
  exit 1
}

# Defaults
COUNT=32
UPLINK=""
OWNER="${SUDO_USER:-${USER}}"
BASE_IP="172.16.0.1"

while getopts ":n:o:u:b:h" opt; do
  case "$opt" in
    n) COUNT="$OPTARG" ;;
    o) UPLINK="$OPTARG" ;;
    u) OWNER="$OPTARG" ;;
    b) BASE_IP="$OPTARG" ;;
    h|*) usage ;;
  esac
done

[[ -z "$UPLINK" ]] && usage

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (needed for ip/iptables)" >&2
  exit 1
fi

IPT="$(command -v iptables-nft || true)"
[[ -z "$IPT" ]] && IPT="$(command -v iptables || true)"
if [[ -z "$IPT" ]]; then
  echo "iptables or iptables-nft not found in PATH" >&2
  exit 1
fi

# IP helpers
ip_to_int() {
  local a b c d
  IFS=. read -r a b c d <<< "$1"
  echo $(( (a<<24) + (b<<16) + (c<<8) + d ))
}
int_to_ip() {
  local ip=$1
  printf "%d.%d.%d.%d" \
    $(( (ip>>24)&255 )) $(( (ip>>16)&255 )) $(( (ip>>8)&255 )) $(( ip&255 ))
}

# Align base to first usable of its /30
BASE_INT=$(ip_to_int "$BASE_IP")
BASE_NET_INT=$(( BASE_INT & 0xFFFFFFFC ))   # mask /30 -> clear last 2 bits
HOST0_INT=$(( BASE_NET_INT + 1 ))
if (( HOST0_INT != BASE_INT )); then
  echo "[*] Aligning base to first usable of its /30: $(int_to_ip "$HOST0_INT") (from $BASE_IP)"
fi

# Enable IPv4 forwarding
sysctl -w net.ipv4.ip_forward=1 >/dev/null

# Sanity: ensure we won't overflow IPv4 space for requested count
LAST_HOST_INT=$(( HOST0_INT + (COUNT-1)*4 + 1 )) # guest ip check
if (( LAST_HOST_INT > 0xFFFFFFFF )); then
  echo "Requested count overruns IPv4 space from base $BASE_IP" >&2
  exit 1
fi

for i in $(seq 0 $((COUNT-1))); do
  TAP="fctap$i"
  HOST_INT=$(( HOST0_INT + i*4 ))
  GUEST_INT=$(( HOST_INT + 1 ))
  HOST_IP="$(int_to_ip "$HOST_INT")"
  GUEST_IP="$(int_to_ip "$GUEST_INT")"

  echo "[*] $TAP  host=${HOST_IP}/30  guest=${GUEST_IP}  uplink=${UPLINK}  owner=${OWNER}"

  # Create TAP if missing (owned by non-root so Firecracker can open it)
  if ! ip link show "$TAP" &>/dev/null; then
    ip tuntap add dev "$TAP" mode tap user "$OWNER"
  fi

  # Assign /30 to host side if missing; bring link up
  if ! ip -4 addr show dev "$TAP" | grep -q "$HOST_IP/30"; then
    # Remove any stale /30 if present on this tap
    while read -r line; do
      [[ -z "$line" ]] && continue
      stale="$(awk '{print $2}' <<<"$line")"
      ip addr del "$stale" dev "$TAP" || true
    done < <(ip -4 -o addr show dev "$TAP" | awk '/inet /{print}')
    ip addr add "$HOST_IP/30" dev "$TAP"
  fi
  ip link set "$TAP" up

  # NAT guest -> uplink (idempotent add)
  if ! $IPT -t nat -C POSTROUTING -o "$UPLINK" -s "$GUEST_IP" -j MASQUERADE &>/dev/null; then
    $IPT -t nat -A POSTROUTING -o "$UPLINK" -s "$GUEST_IP" -j MASQUERADE
  fi

  # Allow established flows back from uplink to this tap
  if ! $IPT -C FORWARD -i "$UPLINK" -o "$TAP" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT &>/dev/null; then
    $IPT -A FORWARD -i "$UPLINK" -o "$TAP" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
  fi

  # Allow new flows from this tap to uplink (scoped to its guest IP)
  if ! $IPT -C FORWARD -i "$TAP" -o "$UPLINK" -s "$GUEST_IP" -j ACCEPT &>/dev/null; then
    $IPT -A FORWARD -i "$TAP" -o "$UPLINK" -s "$GUEST_IP" -j ACCEPT
  fi
done

echo "[*] Done. Guest per tap i: ip addr add \$(base + i*4 + 1)/30 dev eth0 && ip route add default via \$(base + i*4)"
