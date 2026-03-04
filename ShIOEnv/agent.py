import copy
import itertools
import json
import multiprocessing.pool
import os
import time
import threading
import traceback
import uuid
from datetime import datetime
from threading import Lock
from typing import List, Tuple, Dict, Iterable, Iterator

import gymnasium as gym

from ShIOEnv.docker_executor import DockerExecutor
from ShIOEnv.firecracker_executor import FirecrackerExecutor
from ShIOEnv.shioenv import ShIOEnv
from ShIOEnv.utils import is_placeholder, split_placeholders, init_local_placeholders, is_image, get_working_dirs, \
                           get_local_ph, append_to_ndjson, make_exec_cfg, build_local_action_spaces_for_cmd, stable_unique
from ShIOEnv.placeholder_types import *

BASE_CONFIG = dict.fromkeys(["id", "base_id", ], None) # Assigns None as the default value
nest_starters = ["[START]", "[InputNest]", "[PipeInputNest]", "[SemiInputNest]", "[FileRedirInputNest]", "[LOpInputNest]", "[Input]"]  # new command starters, set to pop if trunc



from collections import ChainMap, defaultdict

class OverlayDirs:
    """
    Read-through overlay for dir_map[env_name]: {cwd: {PH: [values...]}}
    Writes go to the overlay dict only.
    """
    def __init__(self, base_env_dir_map: dict[str, dict]):
        self.base = base_env_dir_map
        self.overlay = defaultdict(dict)

    def get(self, cwd: str):
        return ChainMap(self.overlay.setdefault(cwd, {}), self.base.get(cwd, {}))

    def add_ph_values(self, cwd: str, ph: str, values: list[str]):
        o = self.overlay.setdefault(cwd, {})
        o.setdefault(ph, [])
        o[ph].extend(values)

    def has_cwd(self, cwd: str) -> bool:
        return cwd in self.overlay or cwd in self.base

class ShIOAgent:
    def __init__(self, config: dict) -> None:

        if not all([_ in config.keys() for _ in ["executor", "env", "runner", "dataset"]]):
            raise KeyError("Missing at least 1 inner config dicts: ", '["executor", "model", "env", "runner", "dataset"]')

        self.config = config
        self.verbose = self.config['runner'].get('verbose', True)
        self.debug = self.config['runner'].get('debug', False)

        gym.register(id='vm-env', entry_point=ShIOEnv, nondeterministic=True)

        self.rand_map = {}  # global placeholders to be selected randomly  {id: {rand_map from before}
        self.dir_map = {}  # directory specific placeholders to be selected randomly
        self.base_executor_configs = []  # list of base dicts to modify pre-env setup (used as keys for working_dirs)
        self.working_dirs = {}
        base_working_dirs = self.config['executor']['working_dirs']  # where to start from to expand

        self.pre_exec_context = {}  # pre-exec commands to modify user, env, and other local features
        with open(self.config["executor"].get("pre_exec_context_path", "maps/sys_context_maps.json"), "r") as f:
            self.pre_exec_context_base = json.load(f)
        self.exec_method = config["env"]["exec_method"]  # switch to build sample sets

        """ CONCURRENCY VARIABLES """
        # each having their own environments, choose some execs randomly (kernel-rootfs pair, vcpus, mem_mib, etc.
        self.n_workers = config["runner"].get("n_workers", 1) if not self.debug else 1

        self.all_append_lock = Lock()  # lock for post-episode logging+caching
        self._global_fc_uid_counter = 3  # Start from 3 (0-2 are reserved)
        self._fc_uid_lock = Lock()

        self.barrier_lock = Lock()
        self.active_workers = set()
        self.barrier = threading.Barrier(self.n_workers)

        def _prepare_executor(executor):
            """
            Use a pre instantiated executor to harvest working directories and
            placeholder maps, then tear it down again.

            This helper is intended to run in a multiprocessing.pool.ThreadPool.
            It returns (base_id, working_dirs, rand_map, dir_map, pre_exec_context).
            """
            local_rand_map, local_dir_map, pre_exec_context = {}, {}, {}

            base_id = getattr(executor, "base_id", None)
            if base_id is None:
                raise ValueError("Executor instance must expose a base_id attribute")

            # The executor context manager is responsible for setup and teardown
            wdirs = get_working_dirs(executor=executor, starting_dirs=base_working_dirs)
            init_local_placeholders(
                rand_map=local_rand_map,
                dir_map=local_dir_map,
                pre_exec_context=pre_exec_context,
                pre_exec_context_base=self.pre_exec_context_base,
                executor=executor,
                working_dirs=wdirs,
            )
            executor.teardown()

            return base_id, wdirs, local_rand_map, local_dir_map, pre_exec_context

        def _check_executor(cfg: dict) -> Tuple[dict, bool]:
            try:
                cfg['fc_uid'] = self._get_next_fc_uid()
                with self.Executor_cls(**cfg) as executor:
                    out, _ec = executor.run_cmd(["pwd"])
                    if _ec != 0:
                        raise RuntimeError
            except (RuntimeError, TimeoutError) as e:
                print(f"[!] {cfg.get('base_id', 'unk_id')} {cfg.get('kernel_path', 'unk_kern')} failed to execute or agent timed out")
                return cfg, False
            return cfg, True

        # Pre-check for existence of environments
        if self.exec_method == 'firecracker':
            self.Executor_cls = FirecrackerExecutor
            self.valid_executor_ids = os.listdir(config['executor']['fc_rootfs_store'])
            self.valid_kernel_ids = [os.path.join(config['executor']['fc_kernel_store'], _) for _ in os.listdir(config['executor']['fc_kernel_store'])]
            # build the cartesian product once
            check_cfgs = [  # all kernel-rootfs configs
                dict(base_id=eid,
                     worker_id=f"fc-{eid}-{uuid.uuid4().hex[:8]}",
                     rootfs_path=os.path.join(config['executor']['fc_rootfs_store'], eid),
                     kernel_path=kid)
                for kid in self.valid_kernel_ids
                for eid in self.valid_executor_ids
            ]

            results_rootfs = defaultdict(list)  # rootfs-id: [True/False, ...]
            results_kernel = defaultdict(list)  # kernel-path: [True/False, ...]
            print(f"[*] Checking compatibility for {len(check_cfgs)} (r={len(self.valid_executor_ids)},k={len(self.valid_kernel_ids)}) configs.")

            for cfg in check_cfgs:
                cfg, ok = _check_executor(cfg)
                results_rootfs[cfg['base_id']].append(ok)
                results_kernel[cfg['kernel_path']].append(ok)
                print(f"[*] {'GOOD' if ok else 'BAD '} --- kernel: {cfg['kernel_path']} | fs: {cfg['base_id']} ")

            # Keep a rootfs if any kernel works with it
            self.valid_executor_ids = [rid for rid, attempts in results_rootfs.items() if any(attempts)] # at least one success
            # Keep a kernel only if it worked with every tested rootfs
            self.valid_kernel_ids = [kid for kid, attempts in results_kernel.items() if all(attempts)]  # no failures at all

            ph_cfgs = [  # configs for placeholder population
                dict(
                    base_id=eid,
                    worker_id=f"fc-{eid}-{uuid.uuid4().hex[:8]}",
                    rootfs_path=os.path.join(config['executor']['fc_rootfs_store'], eid),
                    kernel_path=self.valid_kernel_ids[0]
                )
                for eid in self.valid_executor_ids
            ]
        else:
            self.Executor_cls = DockerExecutor
            self.valid_executor_ids = [img for img in config['executor']['docker_images'] if is_image(img)]
            self.valid_kernel_ids = []
            ph_cfgs = [dict(base_id=_) for _ in self.valid_executor_ids]

        if not self.valid_executor_ids:
            raise FileNotFoundError("[!] No valid images or .ext4 filesystems provided.")

        f_r = '\n'.join(self.valid_executor_ids)
        print(f"[*] Compatibility check complete. exec method: {self.exec_method}\n-----\nrootfs\n-----\n{f_r}\n-----")

        n_ph_get_check_workers = self.config['runner'].get('n_workers', min(32, len(self.valid_executor_ids)))
        print(f"[*] Populating {len(self.valid_executor_ids)} placeholder pops using {n_ph_get_check_workers} threads.")
        # Populate the dictionaries concurrently
        ph_executors = []
        for cfg in ph_cfgs:
            cfg['tap_host'] = False  # no TAPs/snapshots for placeholder harvesting
            cfg['snapshot'] = False
            cfg['fc_uid'] = self._get_next_fc_uid()
            executor = self.Executor_cls(**cfg)
            executor.setup()
            ph_executors.append(executor)
            time.sleep(0.1)

        if self.debug:
            # Sequential execution for easier debugging
            for executor in ph_executors:
                base_id, wdirs, rmap, dmap, pexec_map = _prepare_executor(executor)
                self.working_dirs[base_id] = wdirs
                self.rand_map.update(rmap)
                self.dir_map.update(dmap)
                self.pre_exec_context.update(pexec_map)
        else:
            # Concurrent placeholder population using threads
            with multiprocessing.pool.ThreadPool(n_ph_get_check_workers) as pool:
                results = pool.map(_prepare_executor, ph_executors)

            for base_id, wdirs, rmap, dmap, pexec_map in results:
                self.working_dirs[base_id] = wdirs
                self.rand_map.update(rmap)
                self.dir_map.update(dmap)
                self.pre_exec_context.update(pexec_map)

        # config to be modified per-worker with random combinations
        self.base_exec_template = {  # firecracker
                    "base_id": None,
                    "rootfs_path": None,
                    "kernel_path": None,
                    "vcpus": None,
                    "mem_mib": None,
                    "snapshot": True,
                    "use_shell": True,
                    "timeout": self.config["env"].get("timeout", 10),
                    "pre_exec_context": None,
                    "max_output_len": 8192,
                    "verbose": False,
                    "tap_host": self.config['executor'].get('tap_host', False),
                    "dns": self.config['executor'].get('fc_tap_dns', "10.100.1.1"),
                } if self.exec_method == "firecracker" else {  # docker
                    "base_id": None,  # used as image name
                    "timeout": self.config["env"].get("timeout", 10),
                    "pre_exec_context": None,
                    "max_output_len": 8192,
                    "networking": self.config['executor'].get('tap_host', False)
                }

        self.config_iterables = {  # loop over to get all valid configs
            "vcpus": self.config['executor'].get('fc_vcpus', [1]),
            "mem_mib": self.config['executor'].get('fc_mem_mib', [256]),
            "kernel_path": self.valid_kernel_ids,
            "base_id": self.valid_executor_ids,  # passed last so is inner most iterator
            } if self.exec_method == 'firecracker' else {
            "base_id": self.valid_executor_ids,
        }

        self.step_buffer = []  # holds (obs, len, act, return, adv, logp, pid)
        self.step_counter = 0
        self.global_step_counter = 0

        self.ep_counter = 0  # per-worker update step counter/collector
        self.tr_counter = 0  # global counter
        self.n_datasets = 1  # global dataset loop counter

        self.n_episodes = self.config['runner'].get('n_episodes', 10000)  # total number of episodes to rollout
        self.n_steps = self.config['runner'].get('n_steps', 1000000)  # total number of steps to rollout
        self.early_stop_repeated = self.config["runner"].get("early_stop_repeated", False)  # whether to early stop random policy's repeated actions
        self.log_cache = []

        """ GRAMMAR POP """
        with open(config["runner"]["utility_map"], "r") as f:
            self.util_map = json.load(f)

        """ UTIL/ACTION MAP POPULATE + MASKING """
        self.util_map["[Input]"] = self.config["runner"]["test_cmd"][:]  # str: list assignment
        self.util_map["[Command]"] = self.config["runner"]["test_cmd"][:]  # same as input, but inner nested (send as local placeholder)

        # which strs to send to new_global_action (may break randomly with some terminals)
        self.global_cmds = [_.lstrip("[").rstrip("]") for _ in self.config["runner"]["test_cmd"]]

        # list of nonterminals that can repeat and need a termininating action. Include all cmdOptions by default
        repeating_options = [f"[{_}Options]" for _ in self.global_cmds if f"[{_}Options]" in self.util_map.keys()]
        with open(config["runner"]["repeat_productions"], "r") as f:  # add known repeating placeholders that need terminating action
            self.repeat_productions = list(json.load(f))
        self.repeat_productions = list(set(self.repeat_productions + repeating_options))

        self.grammar_mask = self.config['runner'].get('grammar_mask', False)  # whether to mask action space according to NT production rule
        self.rand_arg_weight = self.config["runner"].get("random_arg_weight", 3)

        # get action spaces for each production
        self.action_heads = build_local_action_spaces_for_cmd(
            cmd_root="[START]",
            util_map=self.util_map
        )
        n_rand_grounded = len([item for sublist in RANDOM_STR_PLACEHOLDERS.values() for item in sublist])
        n_rand_global = len([item for sublist in self.rand_map[list(self.rand_map.keys())[0]].values() for item in sublist])
        n_rand_local = 0
        for cwds in self.dir_map[list(self.rand_map.keys())[0]].values():
            for types in cwds.values():
                n_rand_local += len(types)
        print(f"{len(self.action_heads.keys()) + len(UNLEARNED_PLACEHOLDERS)} grammar productions")
        print(f"{len([item for sublist in self.action_heads.values() for item in sublist])} defined nonterminals")
        print(f"{n_rand_grounded + n_rand_local + n_rand_global} system-grounded productions")

        if not self.grammar_mask:  # combine all actions heads/trunks into one
            all_actions = []
            for prod_key, prod_actions in self.action_heads.items():
                # if prod_key not in nest_starters:
                if prod_key in self.repeat_productions:
                    all_actions.extend(prod_actions)
                    self.action_heads[prod_key] = ["[all]"]
            all_actions = stable_unique(all_actions)
            self.action_heads["[all]"] = all_actions  # single head
            self.util_map['[all]'] = all_actions

        self.runprefix = config['runner'].get('runprefix', datetime.now().strftime("%Y%m%d_%H%M"))
        self.modeldir = os.path.join(config["runner"].get("modeldir", "policymodel/"), self.runprefix)
        self.datadir = os.path.join(config["runner"].get("datadir", "data/"), self.runprefix)

        """ DATASET VARIABLES """
        self.session_id = 0  # iterate with full sequence write
        self.dataset_write_cache = []
        self.dataset_write_seqs = set()
        self.dataset_file = self.config["dataset"].get("datafile", "data.ndjson")  # dataset file
        self.dataset_seq_file = self.config["dataset"].get("seqsfile", "seqs.json")  # used to ensure uniqueness of commands
        self.dataset_size = self.config["dataset"].get("dataset_size", 5000)
        self.dataset_store_every = self.config["dataset"].get("dataset_store_every", 500)
        self.dataset_every = self.config['dataset'].get('dataset_every', 100000)
        self.dataset_ctr = 0  # number of samples per datafile

        self.dataset_tries = 0  # number of actual generations
        self.max_dataset_tries = self.config['dataset'].get('max_trials', self.dataset_size * 5)

        print(f"Grammar Masked: {self.grammar_mask}")
        print("Handler initialized")

    def _get_next_fc_uid(self) -> int:
        """Thread-safe method to get the next available fc_uid"""
        with self._fc_uid_lock:
            uid = self._global_fc_uid_counter
            self._global_fc_uid_counter += 1
            return uid

    def _get_exec_config(
            self, iterables: Dict[str, Iterable],
    ) -> Iterator[dict]:
        """
        Generate all executor-config permutations with unique fc_uid values.
        """
        if not iterables:
            yield make_exec_cfg(self.base_exec_template)
            return

        keys, pools = zip(*iterables.items())
        for combo in itertools.product(*pools):
            overrides = dict(zip(keys, combo))
            if self.exec_method == 'firecracker':
                overrides['rootfs_path'] = os.path.join(self.config['executor']['fc_rootfs_store'], overrides['base_id'])
                chosen_kernel = os.path.basename(overrides.get('kernel_path', ''))  # get actual name for id
                overrides['pre_exec_context'] = self.pre_exec_context[overrides['base_id']]

                # Use thread-safe fc_uid assignment
                overrides['fc_uid'] = self._get_next_fc_uid()
                # Include uid in worker_id so we can re-mint later
                overrides['worker_id'] = (
                    f"fc-{overrides.get('base_id')}-{chosen_kernel}-"
                    f"{overrides.get('vcpus', 0)}c-{overrides.get('mem_mib', 0)}m-"
                    f"uid{overrides['fc_uid']}-{uuid.uuid4().hex[:8]}"
                )
            else:
                # docker: we’ll set worker_id per worker later
                pass

            yield make_exec_cfg(self.base_exec_template, **overrides)

    def exec_configs_for_workers(self, iterables, n_workers):
        """
        Return exactly n_workers configs, ensuring unique fc_uid for each.
        Generate all configs at once to ensure unique IDs.
        """
        unique_cfgs = list(self._get_exec_config(iterables))
        if not unique_cfgs:
            raise RuntimeError("No executor configurations could be generated.")

        # Pre-generate all configs with unique fc_uids
        worker_configs = []
        cfg_cycle = itertools.cycle(unique_cfgs)

        for i in range(n_workers):
            cfg = copy.deepcopy(next(cfg_cycle))
            # For Firecracker, ensure each worker gets a truly unique fc_uid
            if self.exec_method == 'firecracker':
                # Mint a fresh fc_uid for EVERY worker instance
                cfg['fc_uid'] = self._get_next_fc_uid()
                cfg['active_user'] = self.rand_map[cfg.get('base_id')]['[Username]']

                # Recompute worker_id to include new uid and a fresh suffix
                chosen_kernel = os.path.basename(cfg.get('kernel_path', ''))
                cfg['worker_id'] = (
                    f"fc-{cfg.get('base_id')}-{chosen_kernel}-"
                    f"{cfg.get('vcpus', 0)}c-{cfg.get('mem_mib', 0)}m-"
                    f"uid{cfg['fc_uid']}-{uuid.uuid4().hex[:8]}"
                )
            else:
                # Docker: ensure unique worker_id per worker
                cfg['active_user'] = self.rand_map[cfg.get('base_id')]['[Username]']
                cfg['worker_id'] = f"docker-{cfg.get('base_id')}-{uuid.uuid4().hex[:8]}"

            worker_configs.append(cfg)

        return worker_configs

    def _sample_action_index(self, head_key: str) -> str:
        """
        Returns:
          idx    : int   sampled action index in the head action list
          Ah     : int   action-space size for this head
        """
        if head_key in self.action_heads:
            actions = self.action_heads[head_key]
        else:
            raise KeyError(f"No valid head action list for head_key={head_key}")
        Ah = len(actions)
        if Ah <= 0:
            raise RuntimeError(f"Empty action list for head_key={head_key}")
        idx = random.randrange(Ah)  # uniform
        return actions[idx]

    def run_episode_multi_syntax(self, env, dataset_check: bool = True):
        """
        Main sampler for episode. Policy builds arguments from grammar expansions and sends to environment
        """
        obs, info = env.reset()
        done, trunc = False, False
        in_cmd = False  # check for if a new command production is nested or global in session
        ARG_START = "<ARG>"
        ARG_END = "</ARG>"
        arg_depth = 0  # counter for when when argument is built
        n_seqs = 0  # counter for trunc reset for next seq if multicommand session

        env_name = env.unwrapped.get_executor_id()
        rand_map = {env_name: self.rand_map[env_name]}  # use direct reference for rand_map (read-only in an episode)
        dir_overlay = OverlayDirs(self.dir_map[env_name])  # overlay for dir_map per-episode

        policy_control_id = "[START]"  # starting nonterminal
        policy_output_stack = ["[START]"]
        policy_output_buffer = ""
        n_id_err = 0

        all_infos = []

        action_log = []

        while not done:
            try:
                if trunc:  # option adding hit soft limit, pop items before [cmdOptions] and option PH, continue as normal, set done to true after
                    if arg_depth <= 3:  # no more items, stack is empty popped/back to start level, can end
                        _ = ""
                        new_global = 0
                        if len(policy_output_buffer) > 0:
                            if not in_cmd:
                                in_cmd = True
                                n_seqs += 1
                                new_global = 1  # in case nothing sent yet (no args)
                            action = {"exec_action": 0,
                                      "new_global": new_global,
                                      "input_addition": policy_output_buffer,
                                      }
                            obs, reward, done, trunc, info = env.step(action)
                            policy_output_buffer = ""
                            # rewards[-1] += reward  # add back in if want to reward for incomplete args still in stack when truncated
                        action = {  # last exec for final reward
                            "exec_action": 1,
                            "new_global": 0,
                            "input_addition": "",
                        }
                        try:
                            candidate_input = "; ".join([env.unwrapped.join_cmd_args(_) for _ in env.unwrapped.clean_input_seq()][1:])
                            env_name = env.unwrapped.get_executor_id()
                            if dataset_check and f"{env_name}-{candidate_input}" in self.dataset_write_seqs:
                                # mark skip and finish episode without executing
                                info = {
                                    "image": env_name, "input": candidate_input, "skip_sample": True,
                                    "redundancy_score": 0.0, "exit_code": -1, "redund_error": 0.0, "input_args": [""]
                                }
                                done = True
                                trunc = True
                                all_infos.append(info)
                                break
                            else:
                                obs, reward, done, trunc, info = env.step(action)
                        except Exception as e:
                            print(f"ACTION LOG: {action_log}")
                            raise e
                        in_cmd = False
                        all_infos.append(info)

                        cwd = env.unwrapped.get_curr_cwd()  # updating cwd after exec_action (full cmd in sequence)
                        if not dir_overlay.has_cwd(cwd):
                            get_local_ph(cwd, rand_map=rand_map, dir_map={env_name: dir_overlay.overlay}, executor=env.unwrapped.vm_executor)

                        if n_seqs >= env.unwrapped.get_max_global():  # final command is done, exit loop
                            done = True
                        else:
                            trunc = False  # reset loop for next iteration
                            policy_control_id = "[START]"
                            policy_output_stack = ["[START]"]
                            policy_output_buffer = ""
                            arg_depth = 0
                    else:  # some arguments (e.g. file that need to be added
                        if arg_depth <= 4:  # in global view, pop out any repeating productions  # changed from 1 with no uniform stack init
                            new_stack = []
                            for a in policy_output_stack:  # remove future inputs and repeating args
                                if a not in nest_starters and a not in self.repeat_productions:
                                    new_stack.append(a)
                            policy_output_stack = new_stack
                        _ = policy_output_stack.pop(0)
                else:
                    _ = policy_output_stack.pop(0)  # pop off item from stack
            except IndexError:  # last item is in buffer, stack is empty
                trunc = True  # trigger buffer clear (finish last arg) and exec in next check
                n_id_err += 1
                if n_id_err > 1000:
                    print(f"[!] BIG ERR {n_id_err} > 1000 cmd: {obs} | DEPTH: {arg_depth} | BUFFER: {policy_output_buffer}")
                    print("-----")
                    print(f"CONTROL ID: {policy_control_id}")
                    print(f"BUFFER: {policy_output_buffer}")
                    print(f"DEPTH: {arg_depth}")
                    print(f"OUT STACK: {policy_output_stack}")
                    print(f"LAST OBS: {obs}")
                    print(f"--------------------")
                    raise ValueError
                continue

            if _ == ARG_START:
                arg_depth += 1
                continue

            elif _ == ARG_END:
                arg_depth -= 1
                continue
            # Only flush when closed the outermost ARG

            if arg_depth <= 4 and policy_output_buffer.strip():
                # take global action
                new_global = 0
                if not in_cmd:
                    in_cmd = True
                    n_seqs += 1
                    new_global = 1
                action = {
                    "exec_action": 0,
                    "new_global": new_global,
                    "input_addition": policy_output_buffer.strip(),
                }
                obs, reward, done, trunc, info = env.step(action)
                policy_output_buffer = ""

            # control handoff (if not already given, generate option -> if not term option, add to stack (split)
            if _ in self.repeat_productions:  # special handling to check for terminating token and add nonterminal
                policy_control_id = _
                head_key = policy_control_id
                chosen = self._sample_action_index(head_key=head_key)
                early_stop = False

                # Early-stop repeats -> select terminal (last index in local list)
                if self.early_stop_repeated:
                    try:
                        if random.randint(0, self.rand_arg_weight - 1) == 0:
                            early_stop = True
                    except ValueError:
                        pass

                seq_split = split_placeholders(chosen)
                if early_stop:
                    action_log.append(f"{policy_control_id} -> TERM REPEAT")
                    policy_output_stack.insert(0, _)  # repeat head again
                else:
                    action_log.append(f"{policy_control_id} -> {chosen}")

                policy_output_stack.insert(0, ARG_END)
                for __ in reversed(seq_split):
                    policy_output_stack.insert(0, __)
                policy_output_stack.insert(0, ARG_START)

            elif is_placeholder(_):
                if _ in RANDOM_PLACEHOLDERS:
                    try:
                        new_seq = rand_map[env_name][_][random.randint(0, len(rand_map[env_name][_]) - 1)]
                    except ValueError:  # no valid placeholder in current directory, create random string
                        new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                elif _ in LOCAL_RANDOM_PLACEHOLDERS:
                    # get second to most previous cwd in case current command is cd (will be an invalid key since cd current cmd may not be fully built)
                    cwd = env.unwrapped.get_prev_cwd()
                    local_map = dir_overlay.get(cwd)
                    try:
                        new_seq = local_map[_][random.randint(0, len(local_map[_]) - 1)]
                    except ValueError:  # no valid placeholder in current directory, create random string
                        new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                elif _ in RANDOM_STR_PLACEHOLDERS.keys():
                    new_seq = RANDOM_STR_PLACEHOLDERS[_][random.randint(0, len(RANDOM_STR_PLACEHOLDERS[_]) - 1)]
                elif _ in RANDOM_NUM_PLACEHOLDERS.keys():
                    new_seq = str(random.randint(RANDOM_NUM_PLACEHOLDERS[_][0], RANDOM_NUM_PLACEHOLDERS[_][1]))
                elif _ not in self.util_map.keys():  # unknown placeholder, random str + debug
                    new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                else:  # non-defined utils captured above
                    policy_control_id = _
                    if policy_control_id in nest_starters:  # enforce random command selection to prevent picking "easy" commands
                        new_seq = self.util_map[policy_control_id][random.randint(0, len(self.util_map[policy_control_id]) - 1)]
                    # force start point if no grammar constraint (enforces starting command + per-sys nonterminals, nothing else)
                    else:
                        head_key = policy_control_id
                        new_seq = self._sample_action_index(head_key=head_key)

                action_log.append(f"{_} -> {new_seq}")

                seq_split = split_placeholders(new_seq)  # split to seperate terminals/nonterminals
                policy_output_stack.insert(0, ARG_END)  # depth handlers
                for __ in reversed(seq_split):  # reverse for FIFO
                    policy_output_stack.insert(0, __)
                policy_output_stack.insert(0, ARG_START)
            else:
                if len(policy_output_buffer) > 0 or _ != " ":  # don't append space to buffer if buffer is empty
                    policy_output_buffer += _

        return obs, done, trunc, all_infos

    def simulate_rollout(self, sim_cmd: List[str], id=0):
        """ Pass constructed sequence to test reward signals """
        env_cfg = self.config['env']
        exec_cfg = self.exec_configs_for_workers(self.config_iterables, self.n_workers)[0]
        env = gym.make('vm-env', env_config=env_cfg, exec_config=exec_cfg, working_dirs=self.working_dirs[exec_cfg['base_id']], verbose=self.verbose)
        obs, info = env.reset()
        states, actions, rewards, old_log_probs, value_estimates, controlling_policies, all_infos = ([] for _ in range(7))

        for _, arg in enumerate(sim_cmd):
            global_arg = 1 if _ == 0 else 0
            action = {"exec_action": 0,
                      "new_global": global_arg,
                      "input_addition": arg,
                      }
            obs, reward, done, trunc, info = env.step(action)
            rewards.append(reward)
            if id == 0:  # first worker
                print(f"Added {arg} to {'new seq' if global_arg else sim_cmd[0] + ' args'}.")
                print(f"Observed reward: {reward}")

        action = {"exec_action": 1,
                  "new_global": 0,
                  "input_addition": "",
                  }
        obs, reward, done, trunc, info = env.step(action)
        if id == 0:  # first worker
            print(f"Final sequence termination reward.")
            print(f"Observed reward: {reward}")
        rewards.append(reward)
        env.close()
        return obs, rewards, info

    def _select_random_placeholder(self, nt_key: str, env, rand_map, dir_overlay: OverlayDirs) -> List[str]:
        if nt_key in RANDOM_PLACEHOLDERS:
            try:
                new_seq = rand_map[nt_key][random.randint(0, len(rand_map[nt_key]) - 1)]
            except ValueError:  # no valid placeholder in current directory, create random string
                new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
        elif nt_key in LOCAL_RANDOM_PLACEHOLDERS:
            # get second to most previous cwd in case current command is cd (will be an invalid key since cd current cmd may not be fully built)
            cwd = env.unwrapped.get_prev_cwd()
            try:
                local_map = dir_overlay.get(cwd)
                new_seq = local_map[nt_key][random.randint(0, len(local_map[nt_key]) - 1)]
            except ValueError:  # no valid placeholder in current directory, create random string
                new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
        elif nt_key in RANDOM_STR_PLACEHOLDERS.keys():
            new_seq = RANDOM_STR_PLACEHOLDERS[nt_key][random.randint(0, len(RANDOM_STR_PLACEHOLDERS[nt_key]) - 1)]
        elif nt_key == "[Character]":
            new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=1))
        elif nt_key in RANDOM_NUM_PLACEHOLDERS.keys():
            new_seq = str(random.randint(RANDOM_NUM_PLACEHOLDERS[nt_key][0], RANDOM_NUM_PLACEHOLDERS[nt_key][1]))
        elif nt_key not in self.util_map.keys():  # unknown placeholder, random str + debug
            print(f"[*] Unknown nonterminal detected: {nt_key}")
            new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
        else:  # non-defined utils captured above
            new_seq = self.util_map[nt_key][random.randint(0, len(self.util_map[nt_key]) - 1)]
            if nt_key in self.repeat_productions and not random.randint(0, 1):  # random stop
                new_seq += f" {nt_key}"  # repeat production
        return split_placeholders(new_seq)

    def _expand_run_cmd(self, cmd: str, env:ShIOEnv) -> str:
        """
        Ground placeholder cmd str to environment
        """
        env_name = env.unwrapped.get_executor_id()
        rand_map = self.rand_map[env_name]  # use direct reference for rand_map (read-only in an episode)
        dir_overlay = OverlayDirs(self.dir_map[env_name])  # overlay for dir_map per-episode

        ret_cmd = []
        arg_stack = split_placeholders(cmd)
        while arg_stack:
            _arg = arg_stack.pop(0)
            if is_placeholder(_arg):
                new_args = self._select_random_placeholder(_arg, env, rand_map, dir_overlay)
                for __ in reversed(new_args):  # reverse for FIFO
                    arg_stack.insert(0, __)
            else:
                ret_cmd.append(_arg)
        return "".join(ret_cmd)

    def run_inputonly_creation(self, cmds: List[str]) -> None:
        """
        Take list of commands to run in ShIOEnv. Do not calculate redundancies.
        """
        os.makedirs(self.datadir, exist_ok=True)
        self.dataset_tries = 0  # reset in case of intermed_dataset
        self.session_id = 0

        env_cfg = copy.deepcopy(self.config['env'])  # turn off intermediate execution, only need redundancy score
        env_cfg['get_final_score'] = False
        env_cfg['sample_all_combs'] = False
        env_cfg['calc_redundant'] = False
        try:
            with open(os.path.join(str(self.datadir), self.dataset_seq_file), 'r', encoding='utf-8') as f:
                self.dataset_write_seqs = set(json.load(f))
                self.session_id = len(self.dataset_write_seqs)
                self.dataset_tries = len(self.dataset_write_seqs)
        except FileNotFoundError:
            self.dataset_write_seqs = set()
        if self.debug:
            exec_cfg = self.exec_configs_for_workers(self.config_iterables, self.n_workers)[0]
            env = gym.make('vm-env', env_config=env_cfg, exec_config=exec_cfg, working_dirs=self.working_dirs[exec_cfg['base_id']], verbose=self.verbose)
            self._run_inputonly_creation_tread(cmds, env, 0)
        else:
            with multiprocessing.pool.ThreadPool(self.n_workers) as pool:  # new pool since NUM_VMS was possibly modified
                envs = []
                for _, exec_cfg in enumerate(self.exec_configs_for_workers(self.config_iterables, self.n_workers)):
                    envs.append(gym.make('vm-env', env_config=env_cfg, exec_config=exec_cfg, working_dirs=self.working_dirs[exec_cfg['base_id']], verbose=self.verbose))
                    print(f"[*] Env {_} started")
                    time.sleep(0.2)
                [result.wait() for result in [pool.apply_async(self._run_inputonly_creation_tread, [cmds, env, _]) for _, env in enumerate(envs)]]
            self._save_clear_dataset_cache()  # last save

    def _run_inputonly_creation_tread(self, cmds: List[str], env:ShIOEnv, t_id: int = -1, dataset_check:bool = True) -> None:
        seed = self.config['runner'].get('seed', random.randint(1, 1000)) + t_id
        sampler = random.Random(seed)
        random.seed(seed)
        final_obs = (None)

        while len(self.dataset_write_seqs) < self.dataset_size:
            if self.dataset_tries > self.max_dataset_tries:
                break
            try:
                base_cmd = cmds[sampler.randint(0,len(cmds)-1)]
                adapted_cmd = self._expand_run_cmd(base_cmd, env)
                env.reset()
                action = {
                    "exec_action": 0,
                    "new_global": 1,
                    "input_addition": adapted_cmd,
                }
                obs, reward, done, trunc, info = env.step(action)
                candidate_input = "; ".join([env.unwrapped.join_cmd_args(_) for _ in env.unwrapped.clean_input_seq()][1:])
                env_name = env.unwrapped.get_executor_id()
                if dataset_check and f"{env_name}-{candidate_input}" in self.dataset_write_seqs:
                    # mark skip and finish episode without executing
                    info = {
                        "image": env_name, "input": candidate_input, "skip_sample": True,
                        "redundancy_score": 0.0, "redund_error": 0.0, "input_args": [""]
                    }
                    done = True
                    trunc = True
                else:
                    action = {
                        "exec_action": 1,
                        "new_global": 0,
                        "input_addition": ""
                    }
                    final_obs, reward, done, trunc, info = env.step(action)
                    info['input_args'] = info['input_args'][0].split()  # split to get cmd in first pos for per-util
                    if (t_id == 0 or self.dataset_tries % 100 == 0):
                        print(f"[W{t_id:02d}] [ {len(self.dataset_write_seqs)} / {self.dataset_tries} ] EC: {info['exit_code']} | {info['input'][:30]}{'...' if len(info['input']) > 30 else ''}")

                all_infos = [info]
            except Exception as e:
                print(traceback.format_exc())
                print(f"last env state: {env.unwrapped.last_state}")
                print(f"last env action: {env.unwrapped.last_action}")
                print("Throwing out episode and retrying.")
                try:
                    env.unwrapped.vm_executor.reset()
                    print(f"[*] [W{t_id}] Reset complete.")
                except:
                    print(f"[!] [W{t_id}] Reset failed. Restarting...")
                    env.unwrapped.vm_executor.restart()  # should set booted to true if succeeded
                r_i = 0
                while not env.unwrapped.vm_executor._booted:
                    try:
                        r_i += 1
                        print(f"[!] [W{t_id}] Restart failed {r_i} times. Retrying...")
                        env.unwrapped.vm_executor.restart()
                        print(f"[*] [W{t_id}] Restart complete.")
                    except:
                        time.sleep(2.0)
                if not isinstance(e, RuntimeError) and self.debug:
                    exit()
                print(f"[*] [ W{t_id} {env.unwrapped.vm_executor.base_id} ] Worker ready after reset/restart.")
                print("--------------------")
                continue  # throw out and retry

            with self.all_append_lock:
                self.dataset_tries += 1  # always increment to break out in case no new seqs before max size
                self._log_dataset_entry(all_infos)
                if len(self.dataset_write_cache) >= self.dataset_store_every:
                    self._save_clear_dataset_cache()
                    print(f"[*] Dataset saved at {len(self.dataset_write_seqs)} entries")

    def run_dataset_creation(self, save_all: bool = True) -> None:
        """
        Construct and execute commands according to policy, saving env dict returned from ShIOEnv
        """
        os.makedirs(self.datadir, exist_ok=True)
        self.dataset_tries = 0  # reset in case of intermed_dataset
        self.session_id = 0
        env_cfg = copy.deepcopy(self.config['env'])  # turn off intermediate execution, only need redundancy score
        try:
            with open(os.path.join(str(self.datadir), self.dataset_seq_file), 'r', encoding='utf-8') as f:
                self.dataset_write_seqs = set(json.load(f))
                self.session_id = len(self.dataset_write_seqs)
                self.dataset_tries = len(self.dataset_write_seqs)
                self.dataset_ctr = len(os.listdir(self.datadir))-1  # -1 for homefile
                print(f"[*] Loaded {self.dataset_tries} samples from previous run ({self.datadir})")
                print(f"[*] Datset counter set to {self.dataset_ctr}")
        except FileNotFoundError:
            self.dataset_write_seqs = set()
        if self.debug:
            exec_cfg = self.exec_configs_for_workers(self.config_iterables, self.n_workers)[0]
            env = gym.make('vm-env', env_config=env_cfg, exec_config=exec_cfg, working_dirs=self.working_dirs[exec_cfg['base_id']], verbose=self.verbose)
            self._run_dataset_ep_thread(env, 0, save_all)
        else:
            with multiprocessing.pool.ThreadPool(self.n_workers) as pool:  # new pool since NUM_VMS was possibly modified
                envs = []
                print(f"[*] starting {self.n_workers} dataset building environment processes")
                for _, exec_cfg in enumerate(self.exec_configs_for_workers(self.config_iterables, self.n_workers)):
                    env = None
                    while not env:
                        try:
                            env = gym.make('vm-env', env_config=env_cfg, exec_config=exec_cfg, working_dirs=self.working_dirs[exec_cfg['base_id']], verbose=self.verbose)
                        except Exception as e:
                            print("[!] Error spinning up env")
                            time.sleep(2.0)
                    envs.append(env)
                    print(f"[*] Env {_} started")
                    time.sleep(0.1)
                [result.wait() for result in [pool.apply_async(self._run_dataset_ep_thread, [env, _, save_all, True]) for _, env in enumerate(envs)]]
            self._save_clear_dataset_cache()  # last save

    def _run_dataset_ep_thread(self, env: ShIOEnv, t_id: int = -1, save_all: bool = True, go_forever: bool=False) -> None:
        seed = self.config['runner'].get('seed', random.randint(1, 1000))
        random.seed(seed)
        while len(self.dataset_write_seqs) < self.dataset_size:
            if self.dataset_tries > self.max_dataset_tries and not go_forever:
                break
            try:
                final_obs, done, trunc, all_infos = self.run_episode_multi_syntax(env)
            except Exception as e:
                print(traceback.format_exc())
                print(f"last env state: {env.unwrapped.last_state}")
                print(f"last env action: {env.unwrapped.last_action}")
                print("Throwing out episode and retrying.")
                try:
                    env.unwrapped.vm_executor.reset()
                    print(f"[*] [W{t_id}] Reset complete.")
                except:
                    print(f"[!] [W{t_id}] Reset failed. Restarting...")
                    env.unwrapped.vm_executor.restart()  # should set booted to true if succeeded
                r_i = 0
                while not env.unwrapped.vm_executor._booted:
                    try:
                        r_i += 1
                        print(f"[!] [W{t_id}] Restart failed {r_i} times. Retrying...")
                        env.unwrapped.vm_executor.restart()
                        print(f"[*] [W{t_id}] Restart complete.")
                    except:
                        time.sleep(2.0)
                if not isinstance(e, RuntimeError) and self.debug:
                    exit()
                print(f"[*] [ W{t_id} {env.unwrapped.vm_executor.base_id} ] Worker ready after reset/restart.")
                print("--------------------")
                continue  # throw out and retry

            with self.all_append_lock:
                self.dataset_tries += 1  # always increment to break out in case no new seqs before max size
                if save_all or all([info["redundancy_score"] > 0.999 for info in all_infos]):
                    self._log_dataset_entry(all_infos)
                    if len(self.dataset_write_cache) >= self.dataset_store_every:
                        self._save_clear_dataset_cache()
                        print(f"[*] Dataset saved at {len(self.dataset_write_seqs)} entries")

            if (t_id == 0 or self.dataset_tries % 100 == 0):
                print(f"[W{t_id:02d}] [ {len(self.dataset_write_seqs)} / {self.dataset_tries} ] {all_infos[-1]['input'][:30]}{'...' if len(all_infos[-1]['input']) > 30 else ''}")
                if 'skip_sample' not in all_infos[-1]:
                    print(f"      score: {all_infos[-1]['redundancy_score']:.3f} | err: {all_infos[-1]['redund_error']:.3f} | exit_code: {all_infos[-1]['exit_code']}")
        print(f"[W{t_id}] Done.")

    def _log_dataset_entry(self, infos: List[dict]) -> None:
        """ Add dataset info to running dataset if input not already added """
        _write_seq = False  # if new seq
        for info in infos:
            if f'{info["image"]}-{info["input"]}' not in self.dataset_write_seqs and 'skip_sample' not in info.keys():  # don't add repeated sequences
                info["session_id"] = self.session_id
                self.dataset_write_cache.append({
                    'input': info['input'],
                    'input_args': info['input_args'],
                    'output': info['output'],
                    'context_patch': info['context_patch'],
                    'exit_code': info['exit_code']
                })
                self.dataset_write_seqs.update([f'{info["image"]}-{info["input"]}'])
                _write_seq = True
        if _write_seq:
            self.session_id += 1

    def _save_clear_dataset_cache(self, iterpath: bool=True) -> None:
        """ Save dataset and clear intermediate samples. Maintain running list of commands for presence checking """
        d_ctr = f"{self.dataset_ctr}-" if iterpath else ""
        append_to_ndjson(str(os.path.join(str(self.datadir), f'{d_ctr}{self.dataset_file}')), self.dataset_write_cache)
        with open(os.path.join(str(self.datadir), self.dataset_seq_file), 'w') as f:
            json.dump(list(self.dataset_write_seqs), f, indent=4)
        self.dataset_write_cache = []
        self.dataset_ctr += 1
