import os
import pathlib
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback
import uuid
from enum import auto, Enum
from typing import List, Tuple

import psutil
import requests
import requests_unixsocket as ru

from ShIOEnv.base_executor import BaseExecutor
from ShIOEnv.utils import ipv4_to_int, int_to_ipv4

class _VMState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    SNAPSHOT_READY = auto()


class FirecrackerExecutor(BaseExecutor):
    """
    USAGE

    fc = FirecrackerExecutor(config)
    fc.setup()  # starts VM and takes snapshot of initial state.
    fc.run_cmd(['ls'])  # runs command/sequence of commands and resets VM to snapshot.
    ...
    fc.close()  # when experiments are done

    -----
    OR
    -----
    with FirecrackerExecutor(config) as fc:
        fc.run_cmd(['ls'])
    """

    def __init__(self, base_id: str, rootfs_path: str, kernel_path: str, worker_id: str = None,
                 vcpus: int = 1, mem_mib: int = 256,
                 firecracker_path: str = '/usr/local/bin/firecracker',
                 vsock_port: int = 52, tap_host: bool = False, fc_uid: int = 3,
                 verbose: bool = False, snapshot: bool = True, use_shell: bool = True,
                 timeout: int = 10, max_output_len: int = 0,
                 pre_exec_context: str = None,
                 base_ip: str="172.16.0.1", dns: str="10.100.1.1",  # 172.16.0.1 = 2886729729
                 **kwargs):
        """
        :param base_id:
        :param rootfs_path:
        :param kernel_path:
        :param worker_id:
        :param vcpus:
        :param mem_mib:
        :param disk_size:
        :param firecracker_path:
        :param vsock_port:
        :param tap_host:
        :param fc_uid:
        :param verbose:
        :param snapshot:
        :param use_shell:
        :param timeout:
        :param max_output_len:
        """
        super(FirecrackerExecutor, self).__init__(pre_exec_context=pre_exec_context, timeout=timeout, verbose=verbose, max_output_len=max_output_len, **kwargs)
        self._state = _VMState.STOPPED

        self.base_id = base_id
        self.worker_id = worker_id if worker_id else f"{base_id}-{uuid.uuid4().hex[:8]}"
        self.uid = fc_uid  # CRITICAL: Must be unique per concurrent VM
        if self.uid < 3:
            raise ValueError(f"fc_uid must be >= 3, got {self.uid}")
        if verbose:
            print(f"[*] Initializing {worker_id} with fc_uid={fc_uid}")

        # Ensure snapshot directory exists
        os.makedirs("/tmp/fc-snapshots", exist_ok=True)

        # Firecracker run variables
        self.firecracker_path = firecracker_path
        self.workdir = pathlib.Path(f"/tmp/{self.worker_id}")
        self.api_sock = self.workdir / "api.sock"
        self.vsock_uds = pathlib.Path(f"/tmp/{self.worker_id}_agent.vsock")
        self.proc = None
        self.session = ru.Session()

        self.tap_host = tap_host
        self.vsock_port = vsock_port

        # Exec variables
        self.use_shell = use_shell
        self.timeout = timeout
        self.max_output_len = max_output_len

        # VM variables
        self.vcpu = vcpus
        self.mem = mem_mib

        # Snapshot variables
        self.use_snapshot = snapshot
        self.snapshot_dir = pathlib.Path(f"/tmp/fc-snapshots/{self.worker_id}")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_mem = self.snapshot_dir / "memfile"
        self.snapshot_state = self.snapshot_dir / "statefile"

        # Resource variables
        self.kernel_path = kernel_path
        self.kernel = pathlib.Path(self.kernel_path)
        self.base_rootfs_path = rootfs_path
        self.rootfs = self._get_rootfs()
        # self.rootfs = pathlib.Path(self.base_rootfs_path)

        # TAP reservation/locking (so multiple executors don't collide)
        self.tap_lock_dir = pathlib.Path(os.environ.get("FC_TAP_LOCK_DIR", "/run/fc-taps"))
        try:
            self.tap_lock_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # fallback when /run not writable
            self.tap_lock_dir = pathlib.Path("/tmp/fc-taps")
            self.tap_lock_dir.mkdir(parents=True, exist_ok=True)
        self._tap_lock_file = None
        self.dns = dns
        # base_ip start of tap range (only used as start point for host/guest ip assignment based on n taps)
        self.host_ip = base_ip  # ip of host tap (needed for tunnel + guest route + iptable rules)
        self.guest_ip = base_ip  # ip of guest  (needed for interface def on guest)
        self.host_dev_name = 'fctap'
        if self.tap_host:
            self.get_tap()

        # socket retry configuration
        self._socket_retry_count = 3
        self._socket_retry_delay = 0.5
        self._api_timeout = 30.0  # seconds

    def __enter__(self):
        """
        for non-persistent command testing. Call as:

        with FirecrackerExecutor(config) as fe:
            vm.run_command(cmd)
        """
        self.setup()  # default: cold boot
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.teardown()

    def teardown(self) -> None:
        try:
            self.shutdown()
        finally:
            self.close()

    def get_tap(self):
        """
        Attach to a pre-created TAP (fctapX), reserve it via a lock-file,
        and derive host/guest IPs from its configured /30.

        Expects: TAPs already exist and have an IPv4 /30 assigned.
        Does NOT create devices or add iptables rules.
        """
        def _reserve_tap(dev: str) -> bool:
            """Atomically reserve dev by creating a lock file."""
            lock = self.tap_lock_dir / f"{dev}.lock"
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w") as f:
                    f.write(f"{self.worker_id}:{os.getpid()}\n")
                self._tap_lock_file = lock
                return True
            except FileExistsError:
                return False

        def _release_tap():
            if self._tap_lock_file and self._tap_lock_file.exists():
                try:
                    self._tap_lock_file.unlink()
                except Exception:
                    pass
                finally:
                    self._tap_lock_file = None

        def _get_dev_ipv4(dev: str):
            """Return (addr, prefixlen) for IPv4 on dev, or (None, None)."""
            addrs = psutil.net_if_addrs().get(dev, [])
            for a in addrs:
                if a.family == socket.AF_INET:
                    addr = a.address
                    # derive prefix from netmask
                    try:
                        import ipaddress
                        prefix = ipaddress.IPv4Network(f"0.0.0.0/{a.netmask}").prefixlen
                    except Exception:
                        # fallback: count mask bits
                        mask_int = ipv4_to_int(a.netmask)
                        prefix = bin(mask_int).count("1")
                    return addr, prefix
            return None, None

        # Scan existing fctap* and pick an unlocked one that has a /30
        candidates = sorted([n for n in psutil.net_if_stats().keys() if n.startswith("fctap")])

        for dev in candidates:
            # skip locked
            if (self.tap_lock_dir / f"{dev}.lock").exists():
                continue

            addr, pfx = _get_dev_ipv4(dev)
            if not addr or pfx != 30:
                continue  # not provisioned as expected

            # Try to reserve atomically
            if not _reserve_tap(dev):
                continue

            # We have a device; compute host/guest for its /30:
            host_int = ipv4_to_int(addr)
            base_net = host_int & ~0x3  # clear last 2 bits (/30 network)
            usable1 = base_net + 1
            usable2 = base_net + 2
            guest_int = usable2 if host_int == usable1 else usable1

            self.host_dev_name = dev
            self.host_ip = addr
            self.guest_ip = int_to_ipv4(guest_int)

            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] attached {dev}: host_ip={self.host_ip}/30 guest_ip={self.guest_ip}")

            # Ensure link is up (precreate script should have done this already)
            try:
                subprocess.run(["ip", "link", "set", dev, "up"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            return dev

        # No free TAP found
        raise RuntimeError(self.prep_exception_msg(
            RuntimeError.__name__,
            f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] No free pre-created fctap devices with /30 found (or all are locked)."
        ))

    def _get_rootfs(self):
        """
        Return path object of rootfs path.
        If snapshotting (possible multiple workers using the same base_rootfs), copy rootfs to tmp directory.
        """
        if not self.use_snapshot:
            return pathlib.Path(self.base_rootfs_path)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)  # create snapshot directory for storing ext4 copy and snapshot diff

        dst = self.snapshot_dir / "rootfs.ext4"
        if not dst.exists():
            src = pathlib.Path(self.base_rootfs_path)
            try:
                # Try CoW first
                subprocess.run(
                    ["cp", "-a", "--reflink=auto", str(src), str(dst)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                try:
                    # Try hardlink (fast if same fs)
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [_get_rootfs] prepared worker rootfs at {dst}")
        return dst

    def destroy_tap(self):
        """
        Release the reservation on the TAP we attached.
        DOES NOT delete the device or change iptables (pre-provisioned).
        Safe to call multiple times.
        """
        # unlock if we own it
        if self._tap_lock_file and self._tap_lock_file.exists():
            try:
                self._tap_lock_file.unlink()
                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] released {self.host_dev_name}")
            except Exception:
                pass
            finally:
                self._tap_lock_file = None

    def setup(self):
        """Boot once and create a baseline snapshot (if enabled). Idempotent."""
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] setup()")
        self._restore_or_boot(prefer_snapshot=True)
        if self.use_snapshot and not self._have_valid_snapshot():
            self.create_snapshot()
            self._state = _VMState.SNAPSHOT_READY
        elif self.use_snapshot:
            self._state = _VMState.SNAPSHOT_READY
        else:
            self._state = _VMState.RUNNING

    def reset(self):
        """Rewind to baseline snapshot. If that fails, fall back to destructive recreate."""
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Beginning soft restart")
        self._rewind_to_baseline()
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] soft restart complete")

    def restart(self, preserve_snapshot: bool = True):
        """Compatibility wrapper around destructive path when requested explicitly."""
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Beginning hard restart(preserve_snapshot={preserve_snapshot})")
        self._ensure_stopped()
        if preserve_snapshot and self._have_valid_snapshot():
            self._restore_or_boot(prefer_snapshot=True)
        else:
            self._restore_or_boot(prefer_snapshot=False)
            if self.use_snapshot:
                self.create_snapshot()
        self._state = _VMState.SNAPSHOT_READY if self.use_snapshot else _VMState.RUNNING
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] hard restart complete")

    def close(self):
        self._ensure_stopped()
        self._booted = False
        self._state = _VMState.STOPPED

    def shutdown(self, restart: bool = False):
        """Tear down artifacts. Idempotent."""
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] shutdown(restart={restart})")
        self._ensure_stopped()
        # delete snapshot + vsock files
        if self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir, ignore_errors=True)
        if self.vsock_uds.exists():
            try:
                self.vsock_uds.unlink()
            except:
                pass
        if self.tap_host and not restart:
            self.destroy_tap()
        self._state = _VMState.STOPPED

    def _restore_or_boot(self, prefer_snapshot: bool):
        """
        The ONLY place that decides between restoring snapshot vs cold boot.
        Assumes process is not running.
        """
        # Always start from a clean process:
        self._ensure_stopped()
        self._spawn()

        # Try snapshot first without configuring anything
        if prefer_snapshot and self._have_valid_snapshot():
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] restoring snapshot")

            # Try restoration with timeout
            restore_attempts = 0
            max_restore_attempts = 5

            while restore_attempts < max_restore_attempts:
                try:
                    self.restore_snapshot()
                    self._booted = True
                    self._state = _VMState.SNAPSHOT_READY if self.use_snapshot else _VMState.RUNNING
                    return
                except Exception as e:
                    restore_attempts += 1
                    if restore_attempts < max_restore_attempts:
                        if self.verbose:
                            print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Restore attempt {restore_attempts} failed, retrying...")
                        self._ensure_stopped()
                        time.sleep(0.5 * restore_attempts)  # Back off
                        self._spawn()
                    else:
                        if self.verbose:
                            print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] All restore attempts failed, falling back to cold boot")
                        self._ensure_stopped()
                        self._spawn()
                        break

        # Cold boot path
        self._configure()
        self._start()
        # self._wait_for_agent_ready()
        self._verify_agent_connection(retries=40)
        if self.tap_host:
            self._setup_tap_in_guest()
        self._booted = True
        self._state = _VMState.RUNNING

    def _rewind_to_baseline(self):
        """
        Return the VM to baseline after a command.
        Prefer fast snapshot restore; if anything fails, do one destructive rebuild.
        """
        if not self.use_snapshot:
            return  # nothing to do


        if self._have_valid_snapshot():
            try:
                self._restore_or_boot(prefer_snapshot=True)
                return
            except Exception as e:
                if self.verbose:
                    print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] fast rewind failed: {e}")

        # destructive rebuild once
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] destructive rewind")
        self._restore_or_boot(prefer_snapshot=False)
        try:
            self.create_snapshot()
            self._state = _VMState.SNAPSHOT_READY
        except Exception as e:
            # still usable without snapshot
            if self.verbose:
                print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] snapshot re-create failed: {e}")
            self._state = _VMState.RUNNING

    def _ensure_stopped(self):
        """Idempotent: stop process and remove API workdir/vsock."""
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            finally:
                self.proc = None

        # More aggressive socket cleanup
        if self.vsock_uds.exists():
            try:
                # Force close any open connections
                import glob
                for sock in glob.glob(f"/tmp/*{self.worker_id}*"):
                    try:
                        os.unlink(sock)
                    except:
                        pass
                self.vsock_uds.unlink()
            except:
                pass

        # Clean up workdir
        if self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)

        # Force close session
        if hasattr(self, "session") and self.session:
            try:
                self.session.close()
            except:
                pass
            self.session = None
        self._booted = False
        self._state = _VMState.STOPPED

    def _ensure_running(self):
        if self._state == _VMState.STOPPED:
            self._restore_or_boot(prefer_snapshot=True)

    def _setup_tap_in_guest(self):
        cmds = [
            f"ip addr add {self.guest_ip}/30 dev eth0 || true",
            "ip link set eth0 up || true",
            f"ip route add default via {self.host_ip} dev eth0 || true",
            f"echo \"nameserver {self.dns}\" > /etc/resolv.conf",
            "ping -c1 -W1 1.1.1.1 >/dev/null 2>&1 || true"
        ]
        _o, ec = self._send_cmd(" && ".join(cmds))
        # time.sleep(0.25)
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] tap {'ok' if ec == 0 else 'failed'}")

    def create_snapshot(self):
        """Create a Firecracker snapshot in the given directory."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        memfile = os.path.abspath(str(self.snapshot_dir / "memfile"))
        statefile = os.path.abspath(str(self.snapshot_dir / "statefile"))
        # Create snapshot via FC API

        if self._booted:
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Pausing Running VM to snapshot")
            self._patch("/vm", {"state": "Paused"}).raise_for_status()
            self._booted = False
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Creating snapshot")
        self._put("/snapshot/create", {"snapshot_type": "Full", "mem_file_path": memfile, "snapshot_path": statefile})
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Snapshot created")
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Resuming VM from paused state")
        self._patch("/vm", {"state": "Resumed"}).raise_for_status()
        self._booted = True

        return memfile, statefile

    def restore_snapshot(self):
        """Restore this VM from a snapshot in the given directory.
        You must call this on a FRESH Firecracker process with matching config."""
        memfile = os.path.abspath(str(self.snapshot_dir / "memfile"))
        statefile = os.path.abspath(str(self.snapshot_dir / "statefile"))
        # The microVM must be created but not started yet (i.e. after _configure, before _start)

        # Verify snapshot files exist
        if not os.path.exists(memfile) or not os.path.exists(statefile):
            raise RuntimeError(
                self.prep_exception_msg(
                    RuntimeError.__name__,
                    f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Snapshot files missing: memfile={os.path.exists(memfile)}, "
                    f"statefile={os.path.exists(statefile)}"
                )
            )

        try:
            self._put("/snapshot/load", { "mem_backend": {"backend_path": memfile, "backend_type": "File"}, "snapshot_path": statefile, "resume_vm": True})

            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Snapshot restored")
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Resuming VM from snapshot")

            # self._patch("/vm", {"state": "Resumed"}).raise_for_status()

            # Give agent time to initialize after restore
            time.sleep(0.25)
            # Verify with more retries
            self._verify_agent_connection(retries=40)
            self._booted = True

        except requests.exceptions.HTTPError as e:
            # If snapshot restore fails, try to recover
            if self.verbose:
                print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Snapshot restore failed: {e}")
            raise
        except Exception as e:
            if self.verbose:
                print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Unexpected error during snapshot restore: {e}")
            raise
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Snapshot restore complete")

    def _verify_agent_connection(self, retries=10):
        """Enhanced agent verification with more retries"""
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Verifying VSOCK connection.")
        time.sleep(0.25)
        for i in range(retries):
            try:
                output, code = self._send_cmd("echo test", timeout=60)
                if code == 0 and "test" in output:
                    if self.verbose:
                        print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Agent verified responsive (attempt {i + 1})")
                    return True
            except (RuntimeError, socket.timeout, Exception) as e:
                if self.verbose and i == 0:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Agent not ready yet, waiting... ({e})")
                time.sleep(0.25 * (i + 1))  # Progressive backoff

        raise RuntimeError(
            self.prep_exception_msg(
                RuntimeError.__name__,
                f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Agent not responsive after {retries} attempts"
            )
        )

    def _spawn(self):
        # Ensure vsock UDS is removed before starting new VM
        self.session = ru.Session()  # recreate HTTP handler
        os.makedirs(self.workdir, exist_ok=True)
        if self.vsock_uds.exists():
            try:
                self.vsock_uds.unlink()
                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Unlinked stale vsock uds")
            except Exception as e:
                print(e.__name__, f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Warning: Could not remove leftover vsock uds: {e}")
        cmd = [self.firecracker_path, "--api-sock", str(self.api_sock)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self.workdir)

        # micro-poll: 150 * 20ms ≈ 3s cap
        for i in range(150):
            if self.api_sock.exists():
                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] FC API socket ready (#{i})")
                break
            time.sleep(0.02)
        else:
            raise RuntimeError(self.prep_exception_msg(RuntimeError.__name__, f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] API socket was not created"))

    def _configure(self):
        """
        Configures VM using provided rootfs, kernel, and hardware definitions.

        Do not call if restoring to snapshot.
        """
        if self._booted or self._state == _VMState.RUNNING:
            raise RuntimeError("configure() called after VM start/resume")
        self._put("/machine-config", {"vcpu_count": self.vcpu, "mem_size_mib": self.mem, "smt": False})
        self._put("/boot-source", {"kernel_image_path": os.path.abspath(self.kernel), "boot_args": "console=ttyS0 reboot=k pci=off init=/init-agent root=/dev/vda ro"})
        # "boot_args": "console=ttyS0 reboot=k panic=1 pci=off init=/init-agent root=/dev/vda ro"
        self._put("/drives/rootfs", {"drive_id": "rootfs", "path_on_host": os.path.abspath(self.rootfs), "is_root_device": True, "is_read_only": True})
        if self.tap_host:
            self._put(f"/network-interfaces/eth0", {"iface_id": "eth0", "host_dev_name": self.host_dev_name})
        self._put("/vsock", {"guest_cid": self.uid, "uds_path": str(self.vsock_uds)})
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Firecracker API/machine configuration complete")

    def _start(self):
        if not self._booted:
            self.is_running = self._put("/actions", {"action_type": "InstanceStart"}).status_code == 204
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] instance started")
        else:
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] already running. Ignoring InstanceStart action.")

    def _put(self, endpoint, obj, timeout=None):
        url = f"http+unix://{self.api_sock.as_posix().replace('/', '%2F')}{endpoint}"
        try:
            r = self.session.put(url, json=obj, timeout=timeout or self._api_timeout)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("[!] Status code:", r.status_code)
            print("[!] Response body:", r.text)
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(self.prep_exception_msg(RuntimeError.__name__, f"[!] API PUT {endpoint} failed: {e}"))
        return r

    def _patch(self, endpoint, obj, timeout=None):
        url = f"http+unix://{self.api_sock.as_posix().replace('/', '%2F')}{endpoint}"
        try:
            r = self.session.patch(url, json=obj, timeout=timeout or self._api_timeout)
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            raise requests.exceptions.HTTPError(
                self.prep_exception_msg(requests.exceptions.HTTPError.__name__, f"[!] Status code: {r.status_code}\n[!] Response body: {r.text}"))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(self.prep_exception_msg(RuntimeError.__name__, f"[!] API PATCH {endpoint} failed: {e}"))
        return r

    def _send_cmd_exec(self, cmds: List[str], **kwargs) -> Tuple[str, int]:  # TODO: ALL EXECUTING IN ROOT DIRECTORY
        cmd = " ".join(shlex.quote(part) for part in cmds)

        output, exit_code = self._send_cmd(cmd, **kwargs)

        if exit_code == 137 or exit_code == 124:  # container kill from max_output reached or timeout reached
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] TIMEOUT OR MAX OUT")
                print(f"    [*] cmd: {' '.join(shlex.quote(part) for part in cmds)}")
                print(f"    [*] OUT: {output[:20]}")
            exit_code = 0
        try:
            self._rewind_to_baseline()
        except Exception as e:
            if self.verbose:
                print(f"[!] rewind failed after command: {e}")
            # last-chance recreate to keep session usable
            self._ensure_stopped()
            self._restore_or_boot(prefer_snapshot=False)
            if self.use_snapshot:
                try:
                    self.create_snapshot()
                except:
                    pass
        return output, exit_code

    def _send_cmd(self, cmd: str, **kwargs) -> Tuple[str, int]:
        """
        Send one command string to the guest agent through VSOCK and return (output, exit_code).
        Agent will remain active in current context.

        payload header
        ----------------
            [int32 timeout]         - seconds
            [int32 use_shell]       - 0/1
            [int32 max_len]         - max length of output (<0 shortens to no limit)
            [int32 cmd_len]         - bytes that follow
            [cmd_len bytes]         - UTF-8 command
        ----------------
        response format
        ----------------
            [int32 exit_code]       - 0+
            [int32 out_len]         - bytes that follow
            [out_len bytes]         - UTF-8 output
        ----------------
        """

        def _vsock_connect(port, timeout=30, retry_count=3):
            """VSOCK connection with retry logic"""
            for attempt in range(retry_count):
                deadline = time.time() + timeout
                last_resp = b""

                while time.time() < deadline:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    resp = b""
                    try:
                        # Check if socket file exists
                        if not self.vsock_uds.exists():
                            s.close()
                            time.sleep(0.5)
                            continue

                        s.connect(str(self.vsock_uds))
                        s.sendall(f"CONNECT {port}\n".encode())
                        s.settimeout(10)  # Short timeout for response

                        try:
                            resp = s.recv(128)
                            last_resp = resp
                            if resp.startswith(b"OK"):
                                return s
                        except socket.timeout:
                            pass

                    except (ConnectionRefusedError, FileNotFoundError) as e:
                        if self.verbose:
                            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Connection attempt {attempt + 1} failed: {e}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Unexpected error in vsock_connect: {e}")
                    finally:
                        if not (s and s.fileno() != -1 and resp and resp.startswith(b"OK")):
                            s.close()

                    time.sleep(min(0.5 * (attempt + 1), 1.0))  # back off slow

                # If we get here, this attempt timed out
                if attempt < retry_count - 1:
                    if self.verbose:
                        print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] VSOCK connection attempt {attempt + 1} timed out, retrying...")
                    time.sleep(self._socket_retry_delay)

                    # Try to ensure VM is responsive
                    if self._booted:
                        try:
                            # Check if VM is paused and resume if needed
                            self._patch("/vm", {"state": "Resumed"})
                        except:
                            pass

            raise RuntimeError(
                self.prep_exception_msg(
                    RuntimeError.__name__,
                    f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] vsock handshake failed after {retry_count} attempts: {last_resp!r}"
                )
            )

        def _recv_exact(sock, n):
            """Enhanced receive with timeout handling"""
            buf = b''

            while len(buf) < n:
                try:
                    chunk = sock.recv(n - len(buf))
                    if not chunk:
                        raise RuntimeError(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Socket closed prematurely")
                    buf += chunk
                except socket.timeout:
                    raise RuntimeError(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Socket timeout while waiting to receive data (hang)")
                except Exception as e:
                    raise RuntimeError(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Socket error: {e}")
            return buf

        max_output_len = kwargs.get("max_output_len", self.max_output_len)
        cmd_bytes = cmd.encode()
        if len(cmd_bytes) >= (1 << 18):  # 256Kb max len
            raise RuntimeError(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Command length {len(cmd_bytes)} bytes is near/over agent cap (256Kb).")

        payload = struct.pack(
            "iiii",
            kwargs.get('timeout', self.timeout),
            int(self.use_shell),
            max(0, max_output_len),
            len(cmd_bytes)
        ) + cmd_bytes

        # Try to send command with retry logic
        last_error = None
        for attempt in range(self._socket_retry_count):
            try:
                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Connecting to VSOCK socket (attempt {attempt + 1})")

                s = _vsock_connect(self.vsock_port)

                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Connected to VSOCK socket")
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Sending input: {cmd[:40]}...")  # Truncate for display

                with s:
                    s.sendall(payload)
                    s.settimeout(kwargs.get('timeout', self.timeout + 5))

                    exit_code = struct.unpack("i", _recv_exact(s, 4))[0]
                    output_len = struct.unpack("i", _recv_exact(s, 4))[0]
                    output = _recv_exact(s, output_len).decode(errors="replace")

                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Output received successfully")

                return output, exit_code

            except RuntimeError as e:
                last_error = e
                print(traceback.format_exc())
                if "Socket closed prematurely" in str(e) and attempt < self._socket_retry_count - 1:
                    if self.verbose:
                        print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Socket error on attempt {attempt + 1}, soft-retrying...")
                    time.sleep(self._socket_retry_delay)
                    # Soft nudge: try to ensure the VM isn't paused and agent is alive.
                    try:
                        self._patch("/vm", {"state": "Resumed"})
                    except Exception:
                        pass
                    try:
                        self._verify_agent_connection(retries=3)
                    except Exception:
                        # Keep retrying connect first; don't reset yet.
                        pass
                    continue
                else:
                    break  # exit retry loop and handle below
            except Exception as e:
                print(traceback.format_exc())
                last_error = e
                if attempt < self._socket_retry_count - 1:
                    if self.verbose:
                        print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Unexpected error on attempt {attempt + 1}: {e}")
                    time.sleep(self._socket_retry_delay)
                else:
                    # Last attempt: let loop end and raise after loop.
                    break

        # After the retry loop: if we’re here, all attempts failed.
        if self.use_snapshot and self.snapshot_state.exists():
            if self.verbose:
                print(
                    f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Command send failed after retries; performing one reset...")
            try:
                self.reset()
            except Exception:
                pass
        raise last_error or RuntimeError(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Failed to send command after all retries")

    def _have_valid_snapshot(self) -> bool:
        try:
            return (
                    self.snapshot_state.exists() and self.snapshot_mem.exists() and
                    os.path.getsize(self.snapshot_state) > 0 and
                    os.path.getsize(self.snapshot_mem) > 0
            )
        except Exception:
            return False

    def prep_exception_msg(self, exception: str, message: str):
        return f"{exception}: {message}"
