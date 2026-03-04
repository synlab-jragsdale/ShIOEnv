#!/bin/bash
# clean_fctap.sh - remove all fctap* devices and their iptables rules
# Usage: sudo ./clean_fctap.sh [--dry-run] [--disable-ip-forward-once-clean]

set -euo pipefail

DRY_RUN=false
DISABLE_IPFWD=false
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    case "$arg" in
      --dry-run) DRY_RUN=true ;;
      --disable-ip-forward-once-clean) DISABLE_IPFWD=true ;;
      *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
  done
fi

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (needed for ip/iptables/sysctl)" >&2
  exit 1
fi

# Choose iptables wrapper (nft or legacy)
IPT="$(command -v iptables-nft || true)"
if [[ -z "${IPT}" ]]; then
  IPT="$(command -v iptables || true)"
fi
if [[ -z "${IPT}" ]]; then
  echo "No iptables or iptables-nft found on PATH." >&2
  exit 1
fi

run() {
  if $DRY_RUN; then
    echo "    (dry-run) $*"
  else
    "$@"
  fi
}

# Convert dotted IPv4 to int and back (no external deps, handles leading zeros safely)
ip2int() {
  local IFS=. a b c d
  read -r a b c d <<<"$1"
  printf '%u\n' $(( (10#$a<<24) + (10#$b<<16) + (10#$c<<8) + 10#$d ))
}
int2ip() {
  local ip=$1
  printf '%d.%d.%d.%d\n' $(( (ip>>24)&255 )) $(( (ip>>16)&255 )) $(( (ip>>8)&255 )) $(( ip&255 ))
}
inc_ip() {
  local ip="$1" inc="${2:-1}"
  local n; n="$(ip2int "$ip")"
  n=$(( n + inc ))
  int2ip "$n"
}

# All fctap* interfaces
mapfile -t TAP_DEVS < <(ip -o link show | awk -F': ' '$2 ~ /^fctap[0-9]+$/ {print $2}')

if [[ ${#TAP_DEVS[@]} -eq 0 ]]; then
  echo "[*] No fctap* interfaces found. Nothing to do."
  exit 0
fi

# Consider all non-fctap, non-lo links as possible uplinks (rules are scoped by -o <uplink>)
mapfile -t UPLINKS < <(ip -o link show | awk -F': ' '$2 !~ /^fctap[0-9]+$/ && $2 != "lo" {print $2}')

ipt_check_delete() {
  # Usage: ipt_check_delete <table or ''> <CHAIN> <spec...>
  local table="$1"; shift
  local chain="$1"; shift
  local args=()
  [[ -n "$table" ]] && args+=(-t "$table")
  args+=(-C "$chain" "$@")
  if $IPT "${args[@]}" >/dev/null 2>&1; then
    args=()
    [[ -n "$table" ]] && args+=(-t "$table")
    args+=(-D "$chain" "$@")
    run "$IPT" "${args[@]}"
  fi
}

for TAP in "${TAP_DEVS[@]}"; do
  echo "[*] Processing $TAP"

  # Determine host /30 address assigned to the tap
  HOST_CIDR="$(ip -4 -o addr show dev "$TAP" | awk '{print $4}' | head -n1 || true)"
  if [[ -z "$HOST_CIDR" ]]; then
    echo "[*]   No IPv4 address on $TAP; will remove FORWARD rules by interface and delete device."
    HOST_IP=""
    GUEST_IP=""
  else
    HOST_IP="${HOST_CIDR%/*}"
    GUEST_IP="$(inc_ip "$HOST_IP" 1)"
    echo "[*]   Host IP: $HOST_IP (/30); Guest IP (assumed): $GUEST_IP"
  fi

  # Delete iptables rules:
  #   nat POSTROUTING: -o <uplink> -s <guest_ip> -j MASQUERADE
  #   FORWARD (return path): -i <uplink> -o <tap> -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
  #   FORWARD (outbound): -i <tap> -o <uplink> -s <guest_ip> -j ACCEPT
  for UPL in "${UPLINKS[@]}"; do
    if [[ -n "$GUEST_IP" ]]; then
      ipt_check_delete nat POSTROUTING -o "$UPL" -s "$GUEST_IP" -j MASQUERADE
      ipt_check_delete ''  FORWARD      -i "$TAP" -o "$UPL" -s "$GUEST_IP" -j ACCEPT
    fi
    ipt_check_delete ''  FORWARD      -i "$UPL" -o "$TAP" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
  done

  # Bring link down and delete
  if ip link show "$TAP" >/dev/null 2>&1; then
    run ip link set "$TAP" down
    # Prefer 'ip link del', fall back to 'ip tuntap del'
    if ! run ip link del "$TAP"; then
      run ip tuntap del dev "$TAP" mode tap || true
    fi
    echo "[*]   Deleted $TAP"
  fi
done

# Optionally disable ip_forward if no fctap remain
if $DISABLE_IPFWD; then
  mapfile -t REMAIN < <(ip -o link show | awk -F': ' '$2 ~ /^fctap[0-9]+$/ {print $2}')
  if [[ ${#REMAIN[@]} -eq 0 ]]; then
    if $DRY_RUN; then
      echo "    (dry-run) sysctl -w net.ipv4.ip_forward=0"
    else
      sysctl -w net.ipv4.ip_forward=0 >/dev/null
    fi
    echo "[*] Disabled net.ipv4.ip_forward (no fctap devices remain)."
  fi
fi

echo "[*] Done."

