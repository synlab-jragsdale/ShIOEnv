import base64
import copy
import ipaddress
import json
import os.path
import random
import shlex
import re
import socket
import string
import struct
from datetime import datetime
from pprint import pprint
from typing import List, Tuple, Dict, Any


class BaseExecutor:

    base_id: str  # base image/rootfs desc
    id: str  # worker-specific id (base_id + config params)

    def __init__(self, pre_exec_context: Dict[str, Any] = None, timeout: int = 30, quiet: bool=True, verbose: bool=False, max_output_len: int = 0, **kwargs):
        self.quiet = quiet
        self.verbose = verbose
        self.system_contents = pre_exec_context if pre_exec_context else {}
        self.system_users = kwargs.get('active_user', ['root'])
        self.timeout = timeout
        self.max_output_len = max_output_len
        self._booted = False

        self.session_user = None  # e.g., "deptit"  (None or "root" = run as root)
        self.__dict__.update(kwargs)

        self.probes = [
            ("user", "whoami", False),
            ("pwd", "pwd", False),
            ("ls", "ls -la | head -n 1000", False),
            ("groups", "groups", False),
            ("env", "env", False),
            ("set", "set -o posix; set", False),
            ("shopt", "shopt", False),
            ("ulimit", "ulimit -a", False),
            ("ip_addr", "ip -j addr 2>/dev/null || ip -o addr show 2>/dev/null || ifconfig -a", True),
            ("ip_route", "ip -j route 2>/dev/null || ip route show 2>/dev/null || route -n", True),
            ("ip_link", "ip -j link 2>/dev/null || ip -o link show", True),
            ("iptables", "sudo iptables-legacy -S 2>/dev/null || sudo iptables -S || true", True)
        ]

    def setup(self):
        raise NotImplementedError

    def close(self):
        pass

    def shutdown(self):
        pass

    def teardown(self):
        pass

    def restart(self):
        pass

    def reset(self):
        pass

    def get_start_context(self) -> List[List[str]]:
        """
        Create random start environment commands to set user, environment variables, and other defined local session values

        :return: List of commands to be executed before any other upon session start.
        """
        ret_user_cmds = []
        ret_root_cmds = []
        self.session_user = self.system_users[random.randint(0, len(self.system_users) - 1)]
        for cmd_key, cmd_dict in self.system_contents.items():
            n_times = random.randint(0, 5) if cmd_dict['repeatable'] else random.randint(0, 1)
            t_keys = random.sample(sorted(cmd_dict['command']), min(len(cmd_dict['command'].keys()), n_times))  # number of instances
            for t_key in t_keys:
                try:
                    chosen_cmd = t_key + cmd_dict['command'][t_key][random.randint(0, len(cmd_dict['command'][t_key]) - 1)]
                except ValueError:  # empty options, only use key
                    chosen_cmd = t_key
                if cmd_dict.get('root', False):
                    ret_root_cmds.append(chosen_cmd)
                else:
                    ret_user_cmds.append(chosen_cmd)

        return [ret_user_cmds, ret_root_cmds]

    def _prep_cmd(self, cmds: List[str], **kwargs) -> List[str]:
        # Silence all but the last command (session sim)
        if self.quiet:
            for i, cmd in enumerate(cmds[:-1]):
                if '>' not in cmd:
                    cmds[i] = f"{cmd} > /dev/null 2>&1"

        cmd_send = "; ".join(cmds)

        # Allow per-call override; else use the sticky session user
        effective_user = kwargs.get("as_user", self.session_user)
        if effective_user and effective_user != "root":
            # su works from root without password/TTY; login shell gives HOME, PATH, env, etc.
            cmd_send = f"su {shlex.quote(effective_user)} -c {shlex.quote(cmd_send)}"

        final_command = ["/usr/bin/timeout", f"{kwargs.get('timeout', 10)}s",
                         "/bin/bash", "-c", cmd_send]
        return final_command

    def run_cmd(self, cmds: List[str], pre_exec_context: List[List[str]] = None, **kwargs) -> Tuple[str, int]:
        """ pre_exec_context[user-space context, root-level context].
        Passed after get_start_context() randomly chooses. Different each time. """
        context = pre_exec_context if pre_exec_context is not None else [[], []]
        _cmds = self._prep_cmd(context[0] + cmds, **kwargs)

        for _c in context[1]:
            self._send_cmd(_c)  # send pre-exec context without killing session
        ret = self._send_cmd_exec(_cmds, **kwargs)
        return ret

    def _send_cmd_exec(self, cmds: List[str], **kwargs) -> Tuple[str, int]:
        raise NotImplementedError

    def _send_cmd(self, cmd: str, **kwargs) -> Tuple[str, int]:
        raise NotImplementedError

    def run_block_repeat(self, cmds: list[str], repeat: int, pre_exec_context=None, timeout: int | None = None) -> list[str]:
        """
        Executes the last command of cmds repeat times inside a single shell.
        Returns list of captured outputs (length == repeat).
        Resets VM once (via FirecrackerExecutor._send_cmd_exec).
        """
        if not cmds:
            return ["" for _ in range(repeat)]
        prefix = ";".join(["true"] + [f"{c} > /dev/null 2>&1" for c in cmds[:-1]])
        last = cmds[-1]
        S = "__RB_OUT__"

        # Print sentinelled output blocks, each run bounded with `timeout`
        body = []
        if prefix.strip() != "true":
            body.append(f"{prefix}; ")
        body.append("i=0; ")
        body.append(f"while [ $i -lt {int(max(1, repeat))} ]; do ")
        # keep the command’s visible output
        body.append(f'''  printf "{S}\n"; timeout \'{self.timeout-1}s\' bash -lc "{last}{' | head -c ' + str(self.max_output_len) if self.max_output_len > 0 else ''}" ; printf "{S}\n"; ''')
        body.append("  i=$((i+1)); ")
        body.append("done")

        out, _ec = self.run_cmd(["".join(body)], pre_exec_context=pre_exec_context, timeout=int((timeout or self.timeout) * repeat), max_output_len=0)
        outs = []
        # Parse between sentinels
        while S in out:
            try:
                start_idx = out.index(S)
                end_idx = out.index(S, len(S))
                outs.append(out[start_idx + len(S):end_idx])
                out = out[end_idx + len(S):]
            except ValueError:
                break

        return outs

    def get_env_context(self, cmds: List[str], pre_exec_context: List[List[str]] | None = None):
        """
        Returns (exit_code, observable_output, raw_context_dict)
        Stdout layout:
            [command stdout/stderr...]
            CTX_START__<nonce>
            key=BASE64_VALUE
            ...
            CTX_END__<same nonce>
        """
        # Only risky probes are bounded
        T_RISKY = 2

        # Tolerate empty cmds; silence all but last for visibility parity
        cmd_block = ";".join(["true"] + [f"{c} > /dev/null 2>&1" for c in cmds[:-1]] + ([cmds[-1]] if cmds else []))

        # Printable, unique sentinels so we can split reliably
        nonce = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        S = f"CTX_START__{nonce}"
        E = f"CTX_END__{nonce}"

        # executing body
        body = rf'''
_emit() {{
  k="$1"; c="$2"; risky="$3"
  if [ "$risky" = "1" ]; then
    out="$(timeout {T_RISKY}s bash -c "$c" 2>&1 || true)"
  else
    out="$(bash -c "$c" 2>&1 || true)"
  fi
  printf "%s=%s\n" "$k" "$(printf "%s" "$out" | base64 | tr -d '\n')"
}}

# Flag to avoid duplicate ctx printing (normal path vs trap)
__ctx_printed=0\

_dump_ctx() {{
  [ "$__ctx_printed" -eq 1 ] && return 0
  printf "{S}\n"
'''
        for key, cmd, risky in self.probes:
            body += f'  _emit {shlex.quote(key)} {shlex.quote(cmd)} {"1" if risky else "0"}\n'

        body += rf'''  printf "{E}\n"
  __ctx_printed=1
}}

# On timeout (TERM) or manual kill (INT), emit ctx then exit with 124
trap '_dump_ctx; exit 124' TERM INT
# As a last resort, also dump on EXIT if not already printed (covers odd paths)
trap '_dump_ctx' EXIT

# Background watchdog: sends TERM after the deadline so the trap fires
( sleep {self.timeout-1} && kill -TERM $$ 2>/dev/null ) & __wd=$!

# Run user commands directly in this shell so side-effects (cd, export, etc.) persist
# for the probes that _dump_ctx will execute afterwards.
{cmd_block}
ec=$?

# Cancel watchdog
kill $__wd 2>/dev/null; wait $__wd 2>/dev/null || true

# Normal-path context (if no timeout happened)
_dump_ctx

# Exit with the command's exit code
exit "$ec"
'''
        shell = "".join(body)
        script_b64 = base64.b64encode(shell.encode()).decode()
        exec_cmd = (
            f"printf '%s' '{script_b64}' | base64 -d > /tmp/_probe_$$.sh && "
            f"bash /tmp/_probe_$$.sh; _rc=$?; rm -f /tmp/_probe_$$.sh; exit $_rc"
        )

        raw_out, code = self.run_cmd(
            [exec_cmd],
            pre_exec_context=pre_exec_context or [[], []],
            max_output_len=0,
            timeout=self.timeout,
            as_user=self.session_user,
        )

        # Split observable output from the context block using printable sentinels
        start_idx = raw_out.find(S)
        end_idx = raw_out.find(E, start_idx + len(S)) if start_idx != -1 else -1

        if start_idx == -1 or end_idx == -1:
            # No context captured; return what we have so your caller can log it
            print("NOT CTX CAUGHT")
            print(cmds)
            print(f"START IDX: {start_idx}, END IDX: {end_idx}")
            print(f"OUT: {repr(raw_out[:100])}")
            print(f"RAW LEN: {len(raw_out)}")
            return -1, raw_out, {}

        cmd_out = raw_out[:start_idx]
        if self.max_output_len > 0:
            cmd_out = cmd_out[:self.max_output_len]
        kv_blob = raw_out[start_idx + len(S) + 1: end_idx].strip()  # lines between S and E

        # Parse key=BASE64 lines
        context_dict = {}
        for line in kv_blob.splitlines():
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                context_dict[k] = base64.b64decode(v.encode("ascii"), validate=True).decode("utf-8", "replace")
            except Exception:
                # If somehow not base64, keep raw; but with our emit() it should be b64
                context_dict[k] = v

        return code, cmd_out, self.parse_context_env_features(context_dict)

    # gets context for second to last and last command for diff comparison
    def get_env_context_pair(
            self,
            full_cmds: List[str],
            *,
            pre_exec_context: List[List[str]] | None = None,
            timeout: int | None = None,
    ) -> Tuple[str, int, dict, dict]:
        """
        Execute the prefix (all-but-last) and the full command sequence in *one* shell,
        capturing two context blocks:
          - ctx_pre: after prefix (for baseline)
          - ctx_post: after full command (for deltas + output)
        Returns: (final_output, exit_code, ctx_pre, ctx_post)
        """
        if not full_cmds:
            return "", 0, {}, {}

        prefix_cmds = full_cmds[:-1]
        last_cmd = full_cmds[-1]

        # Build the same probe list used by get_env_context
        T_RISKY = 2

        nonce = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        S1 = f"CTX1_START__{nonce}"
        E1 = f"CTX1_END__{nonce}"
        S2 = f"CTX2_START__{nonce}"
        E2 = f"CTX2_END__{nonce}"

        # Silence all but last sub-command in each segment
        def _silence(cmds):
            if not cmds:
                return "true"
            return ";".join([*(f"{c} > /dev/null 2>&1" for c in cmds[:-1]), cmds[-1]])

        body = []
        body.append(rf'''
_emit() {{
  k="$1"; c="$2"; risky="$3"
  if [ "$risky" = "1" ]; then
    out="$(timeout {T_RISKY}s bash -c "$c" 2>&1 || true)"
  else
    out="$(bash -c "$c" 2>&1 || true)"
  fi
  printf "%s=%s\n" "$k" "$(printf "%s" "$out" | base64 | tr -d '\n')"
}}

__ctx_printed_1=0
__ctx_printed_2=0

_dump_ctx() {{
  tag="$1"
  case "$tag" in
    1)
      [ "$__ctx_printed_1" -eq 1 ] && return 0
      printf "{S1}\n"
''')
        for key, pcmd, risky in self.probes:
            body.append(f'      _emit {shlex.quote(key)} {shlex.quote(pcmd)} {"1" if risky else "0"}\n')
        body.append(rf'''      printf "{E1}\n"
      __ctx_printed_1=1
      ;;
    2)
      [ "$__ctx_printed_2" -eq 1 ] && return 0
      printf "{S2}\n"
''')
        for key, pcmd, risky in self.probes:
            body.append(f'      _emit {shlex.quote(key)} {shlex.quote(pcmd)} {"1" if risky else "0"}\n')
        body.append(rf'''      printf "{E2}\n"
      __ctx_printed_2=1
      ;;
  esac
}}

# If the last command is killed by timeout (TERM/INT), still emit the post context
trap '_dump_ctx 2; exit 124' TERM INT
# Safety net: ensure post context exists on any exit path
trap '_dump_ctx 2' EXIT

# Run prefix silently
{{ {_silence(prefix_cmds)} ; }} >/dev/null 2>&1 || true

_dump_ctx 1

# Background watchdog: sends TERM after the deadline so the trap fires
( sleep {self.timeout-1} && kill -TERM $$ 2>/dev/null ) & __wd=$!

# Run last command directly in this shell so side-effects (cd, export, etc.) persist
# for the probes that _dump_ctx 2 will execute afterwards.
{last_cmd}
ec=$?

# Cancel watchdog
kill $__wd 2>/dev/null; wait $__wd 2>/dev/null || true

# Normal post context
_dump_ctx 2

# Exit with the last command's exit code
exit "$ec"
''')
        shell = "".join(body)
        script_b64 = base64.b64encode(shell.encode()).decode()
        exec_cmd = (
            f"printf '%s' '{script_b64}' | base64 -d > /tmp/_probe_$$.sh && "
            f"bash /tmp/_probe_$$.sh; _rc=$?; rm -f /tmp/_probe_$$.sh; exit $_rc"
        )

        out, code = self.run_cmd(
            [exec_cmd],
            pre_exec_context=pre_exec_context or [[], []],
            max_output_len=0,
            timeout=timeout or self.timeout,
            as_user=self.session_user,
        )

        # Parse 2 contexts out of stdout
        def _extract(S, E, src):
            s = src.find(S)
            e = src.find(E, s + len(S))
            if s == -1 or e == -1:
                return "", {}
            head = src[:s]
            block = src[s + len(S) + 1:e].strip()
            tail = src[e + len(E) + 1:]
            ctx = {}
            for line in block.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    try:
                        ctx[k] = base64.b64decode(v.encode("ascii"), validate=True).decode("utf-8", "replace")
                    except Exception:
                        ctx[k] = v
            return head, self.parse_context_env_features(ctx), tail

        head1, ctx1, rest = _extract(S1, E1, out)
        head2, ctx2, tail = _extract(S2, E2, rest)

        # The command's user-visible output is the concat of what printed between blocks
        final_output = (head1 + head2).rstrip("\n")
        if self.max_output_len > 0:
            final_output = final_output[:self.max_output_len]

        return final_output, code, ctx1, ctx2

    def parse_context_env_features(self, context: dict):
        """
        Parses context strings into objects (lists/dicts of literals)
        """
        # helpers
        def process_fs(fs: list):
            fs_data = {}
            for filedata in fs:
                if not filedata.strip():
                    continue
                data_split = filedata.split()
                if len(data_split) < 9:
                    # skip weird lines like "total 12" or device lines without fields
                    continue
                fs_data[" ".join(data_split[8:])] = {
                    "perms": data_split[0],
                    "links": int(data_split[1]),
                    "owner": data_split[2],
                    "group": data_split[3],
                    "size": int(data_split[4]),
                    "month": data_split[5],
                    "day": int(data_split[6]) if data_split[6].isdigit() else "",
                    "year": int(data_split[7]) if ":" not in data_split[7] and data_split[7].isdigit() else datetime.now().year,
                    "time": data_split[7] if ":" in data_split[7] else "00:00",
                }
            return fs_data

        def process_env(env: list):
            env_data = {}
            for _var in env:
                if "=" not in _var:
                    continue
                var, value = _var.split("=", 1)
                env_data[var] = value
            return env_data

        _limit_re = re.compile(r"(.+?)\s*\([^)]*\)\s*(\S+)$")

        def process_limits(lines: list):
            limits = {}
            for ln in lines:
                m = _limit_re.match(ln.strip())
                if not m:
                    continue  # skip malformed rows
                name, raw_val = (s.strip() for s in m.groups())
                if raw_val == "unlimited":
                    val = -1  # or float("inf"), if you prefer
                else:
                    try:
                        val = int(raw_val)
                    except ValueError:
                        val = raw_val  # fallback for odd strings
                limits[name] = val
            return limits

        onoff = {"on": 1, "off": 0}

        # helpers: IP parsing
        def _maybe_json(raw: str):
            raw = raw.strip()
            if not raw:
                return None
            if raw[0] in "[{":
                try:
                    return json.loads(raw)
                except Exception:
                    return None
            return None

        # ip -j addr  |  ip -o addr show  |  ifconfig -a
        def process_ip_addr(raw: str):
            parsed = _maybe_json(raw)
            result = {}
            if isinstance(parsed, list):
                # ip -j addr
                for dev in parsed:
                    ifname = dev.get("ifname")
                    if not ifname:
                        continue
                    addrs = []
                    for a in dev.get("addr_info", []):
                        addrs.append({
                            "family": a.get("family"),  # "inet"/"inet6"
                            "local": a.get("local"),
                            "prefixlen": a.get("prefixlen"),
                            "scope": a.get("scope"),
                            "label": a.get("label"),
                            "valid_lft": a.get("valid_life_time"),  # seconds or "forever"
                            "preferred_lft": a.get("preferred_life_time"),
                        })
                    if addrs:
                        result[ifname] = {"addresses": addrs}
                return result

            # ip -o addr show fallback
            # ex: "2: eth0    inet 10.0.2.15/24 brd 10.0.2.255 scope global eth0 ..."
            ip_o_re = re.compile(
                r"^\s*\d+:\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+brd\s+\S+)?(?:\s+scope\s+(\S+))?.*$"
            )
            fam_map = {"inet": "inet", "inet6": "inet6"}
            if "inet " in raw or "inet6 " in raw:
                for line in raw.splitlines():
                    m = ip_o_re.match(line)
                    if not m:
                        continue
                    ifname, family, cidr, scope = m.groups()
                    family = fam_map.get(family, family)
                    local, _, prefixlen = cidr.partition("/")
                    result.setdefault(ifname, {"addresses": []})["addresses"].append({
                        "family": family,
                        "local": local,
                        "prefixlen": int(prefixlen) if prefixlen.isdigit() else None,
                        "scope": scope or None,
                        "label": None,
                        "valid_lft": None,
                        "preferred_lft": None,
                    })
                if result:
                    return result

            # ifconfig -a fallback (very loose)
            # iface line starts with "eth0:" or "eth0    Link encap:Ethernet", etc.
            cur = None
            for line in raw.splitlines():
                if not line.strip():
                    continue
                # interface header
                if not line.startswith(" "):
                    cur = line.split()[0].rstrip(":")
                    result.setdefault(cur, {"addresses": []})
                    continue
                line = line.strip()
                if line.startswith("inet6 "):
                    parts = line.split()
                    addr = parts[1] if len(parts) > 1 else None
                    local, _, prefix = (addr or "").partition("/")
                    result[cur]["addresses"].append({
                        "family": "inet6", "local": local, "prefixlen": int(prefix) if prefix.isdigit() else None,
                        "scope": None, "label": None, "valid_lft": None, "preferred_lft": None
                    })
                elif line.startswith("inet "):
                    # "inet 192.168.1.10  netmask 255.255.255.0  broadcast ..."
                    parts = line.split()
                    ip = parts[1] if len(parts) > 1 else None
                    result[cur]["addresses"].append({
                        "family": "inet", "local": ip, "prefixlen": None,
                        "scope": None, "label": None, "valid_lft": None, "preferred_lft": None
                    })
            return result

        # ip -j route  |  ip route show  |  route -n
        def process_ip_route(raw: str):
            parsed = _maybe_json(raw)
            routes = []
            if isinstance(parsed, list):
                for r in parsed:
                    routes.append({
                        "dst": r.get("dst"),  # e.g., "default" or "10.0.0.0/24"
                        "gateway": r.get("gateway"),
                        "dev": r.get("dev"),
                        "oif": r.get("oif"),
                        "metric": r.get("metric"),
                        "table": r.get("table"),
                        "protocol": r.get("protocol"),
                        "prefsrc": r.get("prefsrc"),
                        "scope": r.get("scope"),
                    })
                return routes

            # ip route show fallback
            # ex: "default via 10.0.2.2 dev eth0 proto dhcp metric 100"
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                route = {"dst": None, "gateway": None, "dev": None, "metric": None, "protocol": None, "table": None,
                         "scope": None}
                # destination
                route["dst"] = toks[0]
                # scan key-value-ish tokens
                for i, t in enumerate(toks):
                    if t == "via" and i + 1 < len(toks):
                        route["gateway"] = toks[i + 1]
                    elif t == "dev" and i + 1 < len(toks):
                        route["dev"] = toks[i + 1]
                    elif t == "metric" and i + 1 < len(toks):
                        try:
                            route["metric"] = int(toks[i + 1])
                        except ValueError:
                            route["metric"] = toks[i + 1]
                    elif t == "proto" and i + 1 < len(toks):
                        route["protocol"] = toks[i + 1]
                    elif t == "scope" and i + 1 < len(toks):
                        route["scope"] = toks[i + 1]
                    elif t == "table" and i + 1 < len(toks):
                        route["table"] = toks[i + 1]
                routes.append(route)
            if routes:
                return routes

            # route -n fallback (columns)
            # Kernel IP routing table
            # Destination Gateway Genmask Flags Metric Ref Use Iface
            header_seen = False
            for line in raw.splitlines():
                if not line.strip():
                    continue
                if line.startswith("Destination"):
                    header_seen = True
                    continue
                if not header_seen:
                    continue
                cols = line.split()
                if len(cols) < 8:
                    continue
                dst, gateway, genmask, flags, metric, refcnt, use, iface = cols[:8]
                routes.append({
                    "dst": f"{dst}/{genmask}", "gateway": gateway, "dev": iface,
                    "metric": int(metric) if metric.isdigit() else metric,
                    "protocol": None, "table": None, "scope": None
                })
            return routes

        # ip -j link  |  ip -o link show
        def process_ip_link(raw: str):
            parsed = _maybe_json(raw)
            links = {}
            if isinstance(parsed, list):
                for l in parsed:
                    ifname = l.get("ifname")
                    if not ifname:
                        continue
                    links[ifname] = {
                        "state": l.get("operstate"),
                        "mtu": l.get("mtu"),
                        "mac": l.get("address"),
                        "kind": l.get("link_type"),
                        "flags": l.get("flags"),
                    }
                return links

            # ip -o link show fallback
            # ex: '2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ... state UP ... link/ether 12:34:56:...'
            line_re = re.compile(
                r"^\s*\d+:\s+([^:]+):\s+<([^>]+)>.*?\smtu\s(\d+).*?\sstate\s(\S+).*?\slink/\S+\s([0-9a-f:]{17})",
                re.IGNORECASE,
            )
            for ln in raw.splitlines():
                m = line_re.search(ln)
                if not m:
                    # try a looser match without state or mac
                    m2 = re.match(r"^\s*\d+:\s+([^:]+):\s+<([^>]+)>.*?\smtu\s(\d+)", ln, re.IGNORECASE)
                    if not m2:
                        continue
                    ifname, flags, mtu = m2.groups()
                    links[ifname] = {"state": None, "mtu": int(mtu), "mac": None, "kind": None,
                                     "flags": flags.split(",")}
                    continue
                ifname, flags, mtu, state, mac = m.groups()
                links[ifname] = {
                    "state": state,
                    "mtu": int(mtu),
                    "mac": mac,
                    "kind": None,
                    "flags": flags.split(","),
                }
            return links

        def _kv_onoff(block: str) -> dict:
            d = {}
            for ln in block.splitlines():
                parts = ln.split()
                if len(parts) >= 2:
                    d[parts[0]] = 1 if parts[1] == "on" else 0 if parts[1] == "off" else parts[1]
            return d

        # build return dict (robust to missing keys)
        ls_block = (context.get("ls") or "").strip()
        env_block = (context.get("env") or "").strip()
        set_block = (context.get("set") or "").strip()
        shopt_block = (context.get("shopt") or "").strip()
        ulimit_block = (context.get("ulimit") or "").strip()
        iptables_block = (context.get("iptables") or "").strip()

        ip_addr_block = (context.get("ip_addr") or "").strip()
        ip_route_block = (context.get("ip_route") or "").strip()
        ip_link_block = (context.get("ip_link") or "").strip()

        try:
            ret_dict = {
                "user": (context.get("user") or "").strip(),
                "cwd": (context.get("pwd") or "").strip(),
                "fs": process_fs(ls_block.splitlines()[1:]) if ls_block else [],
                "env": process_env(env_block.splitlines()) if env_block else {},
                "groups": (context.get("groups") or "").strip().split(),
                "settings": process_env(set_block.splitlines()) if set_block else {},
                # "settings": _kv_onoff(set_block),
                "shell_options": _kv_onoff(shopt_block),
                "system_limits": process_limits(ulimit_block.splitlines()) if ulimit_block else {},
                "iptable_rules": iptables_block.splitlines(),  # if (iptables_block and user == "root") else [],
                "ip_addrs": process_ip_addr(ip_addr_block) if ip_addr_block else {},
                "ip_routes": process_ip_route(ip_route_block) if ip_route_block else [],
                "ip_links": process_ip_link(ip_link_block) if ip_link_block else {}
            }
        except ValueError as e:
            # print(context)
            raise e
        return ret_dict
