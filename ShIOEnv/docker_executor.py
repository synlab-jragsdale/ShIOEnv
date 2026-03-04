import sys
import threading
import uuid
from typing import List, Tuple
import docker
from docker.types import IPAMConfig, IPAMPool


from ShIOEnv.base_executor import BaseExecutor


class DockerExecutor(BaseExecutor):
    def __init__(self, base_id: str, worker_id: str = None,
                 timeout: int = 10, max_output_len: int = 0,
                 verbose: bool = False, pre_exec_context: str = None,
                 networking: bool = False, network_name: str = "docknet0",
                 ipv4: str = "172.28.0.2", mac_address: str = "00:00:00:00:00:01",
                 **kwargs):
        """

        :param base_id: name of the docker image (required)
        :param worker_id: id of worker (default: image_name + uuid.uuid4().hex)
        :param timeout: timeout in seconds (default: 10)
        :param max_output_len: max output length to read (default: -1 (no limit))

        """

        super(DockerExecutor, self).__init__(pre_exec_context=pre_exec_context, verbose=verbose, **kwargs)
        self._local = threading.local()  # one client per (OS) process

        self.image_name = base_id
        self.base_id = self.image_name
        self.worker_id = worker_id if worker_id else f"dock-{self.base_id}-{uuid.uuid4().hex[:8]}"
        self.timeout = timeout
        self.networking = networking
        self.max_output_len = max_output_len

        self.network_name = network_name
        self.ipv4 = ipv4
        self.mac_address = mac_address
        self.setup()

    def _ensure_network(self):
        if not self.network_name:
            return None
        cli = self._get_client()
        try:
            return cli.networks.get(self.network_name)
        except docker.errors.NotFound:
            # create with a default pool if you want this class to be self-contained
            ipam_pool = IPAMPool(subnet="172.28.0.0/16", gateway="172.28.0.1")
            ipam_cfg = IPAMConfig(pool_configs=[ipam_pool])
            return cli.networks.create(self.network_name, driver="bridge", ipam=ipam_cfg)


    def __enter__(self):
        """
        for non-persistent command testing. Call as:

        with FirecrackerExecutor(config) as fe:
            vm.run_command(cmd)
        """
        self.setup()  # default: cold boot
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def setup(self):
        """
        Setup/verification of existence of worker's executor
        """
        try:
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Setting up worker")
            self.verify()
            if self.verbose:
                print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Worker verified")
        except docker.errors.APIError:
            raise RuntimeError(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Image does not exist.")
        self._booted = True  # set for consistency with firecracker error handling

    def _send_cmd_exec(self, cmds: List[str], **kwargs) -> Tuple[str, int]:
        """
        Executes a list of commands inside a Docker container, returns (logs, exit_code).
        If max_output is not -1, stops the container as soon as max_output characters have been observed.

        :return: (output: str, exit_code: int) tuple
        """
        max_output_len = kwargs.get("max_output_len", self.max_output_len)
        if self.verbose:
            print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Sending input {'; '.join(cmds)}")
        try:
            net = self._ensure_network()
            run_kwargs = dict(
                image=self.image_name,
                command=cmds,
                hostname="svr01",
                mem_limit="256m",
                memswap_limit="512m",
                nano_cpus=int(1e9),
                privileged=True,
                auto_remove=False,
                detach=True,
                tty=False,
            )

            if self.networking:
                if net:
                    if self.verbose:
                        print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Net found. Updating client with address {self.ipv4}")
                    # attach to user-defined network with static IP
                    ep = self._get_client().api.create_endpoint_config(ipv4_address=self.ipv4)
                    net_cfg = self._get_client().api.create_networking_config({net.name: ep})
                    run_kwargs.update(network=net.name, networking_config=net_cfg)
                    if self.mac_address:
                        run_kwargs["mac_address"] = self.mac_address
                else:
                    if self.verbose:
                        print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Net not found. Using default bridge")
                    # fall back to default bridge
                    run_kwargs.update(network_mode="bridge")
            else:
                if self.verbose:
                    print(f"[*] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Net off. Running without networking")
                run_kwargs.update(network_mode="none")

            container = self._get_client().containers.run(**run_kwargs)

            logs_output = ""
            if max_output_len > 0:
                # Stream logs and stop once we've collected max_output characters.
                for log in container.logs(stdout=True, stderr=True, stream=True):
                    # Decode the log chunk (if needed) and accumulate it.
                    chunk = log.decode("utf-8", errors="replace") if isinstance(log, bytes) else log
                    logs_output += chunk

                    # Check if we reached (or exceeded) the limit.
                    if len(logs_output) >= max_output_len:
                        # Optionally, truncate to exactly max_output characters.
                        logs_output = logs_output[:max_output_len]
                        # Stop the container as the output limit is reached.
                        try:
                            container.reload()  # Update container status
                            if container.status == "running":
                                container.kill()
                        except docker.errors.APIError as e:
                            if "not running" not in str(e):
                                pass
                        break

                # Wait for the container to exit after killing it.
                result = container.wait()
                exit_code = result.get("StatusCode", -1)
            else:
                # No output limit: wait for container to finish normally.
                result = container.wait()
                exit_code = result.get("StatusCode", -1)
                logs_output = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

            container.remove()

        except docker.errors.ContainerError as e:
            # Handles cases where Docker raises an error due to a non-zero exit code.
            exit_code = e.exit_status
            logs_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        except docker.errors.APIError as e:
            if self.verbose:
                print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] Error creating or running container: {e}")
            exit_code = -1
            logs_output = f"APIError: {e}"

        if exit_code == 137 or exit_code == 124:  # container kill from max_output reached or timeout reached
            if self.verbose:
                print(f"[!] [ {self.worker_id} ] [{sys._getframe(0).f_code.co_name}] TIMEOUT OR MAX OUT\ncmd: {';'.join(cmds)}")
                print(f"    [!] OUT: {logs_output[:20]}")
            exit_code = 0

        return logs_output, exit_code

    def _send_cmd(self, cmd: str, **kwargs) -> Tuple[str, int]:
        # Cannont send root context commands outside of session. Ignore.
        pass

    def verify(self):
        image = self._get_client().images.get(self.image_name)

    def _get_client(self):
        if getattr(self._local, "cli", None) is None:
            self._local.cli = docker.from_env()
        return self._local.cli

