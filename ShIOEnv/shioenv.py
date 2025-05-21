import statistics
import time
from itertools import combinations
from time import sleep
from typing import Optional, Tuple, List, Any, Dict
from collections import defaultdict

import gymnasium as gym
import random

from ShIOEnv.utils import calc_lev_sim_single, get_env_context, simple_send_cmds_docker, get_context_diff, \
    squeeze_sequence, _strip_dynamic_env_vars, _normalise_output, _WS_RE, update_cwd


class ShIOEnv(gym.Env):
    def __init__(self, config: dict, image_name: str, verbose: bool = False):

        self.config = config

        self.MAX_TRIES = 3

        self.cache_maxlen = config.get("cache_maxlen", 3)
        # key -> list[(output, exit_code, exec_time)]
        self._seq_cache = defaultdict(list)
        self.noise_runs = config.get("noise_runs", 5)  # repeat full seq N times to get noise estimate
        self.noise_sigmaK = config.get("noise_sigmaK", 2.0)  # k*std below the mean for beta
        self.margin = config.get("margin", 0.0)

        self.intermed_exec = config.get("intermed_exec", True)  # set to false to turn off intermediate execution for if only seqs and no intermediate redundancy checks needed.

        self.verbose = verbose

        self.max_global = max(1, config.get('global_horizon', 1)) + 1
        self.max_local = config.get('local_horizon', 5) + 1  # max size for each input seq

        self.global_horizon = max(self.max_global, 2)  # first index taken by directory move
        self.local_horizon = self.max_local - 4  # -4 handles give 2 arg buffer after repeats (repeat productions dont stop sampling)

        # final seq caching for final exec_action
        self.prev_output = ""
        self.prev_exit_code = 0
        self.prev_exec_time = 0.0
        self.prev_context = {}
        self.prev_redund_score = []

        # global context of previous command (cd for first), provides baseline for changes
        self.prev_global_context = {}

        # position counters
        self.curr_global = 0
        self.curr_local = 1

        # docker exec timeout before trap
        self.timeout = config.get('timeout', 10)

        # exec details
        self.image_name = image_name
        self.snapshot_name = config.get("snapshot_name", "base-snapshot")
        # starting directories, uniformly sampled
        self.working_dirs = config.get("working_dirs", ["/"])

        # changed on reset
        self.start_cwd = self.working_dirs[random.randint(0, len(self.working_dirs) - 1)]
        # current state
        self.constructed_input = ()


        """
        input_addition: new argument to be added to command sequence.
        exec_action: Flag for if sequence is ready to be run (triggered no arguments left)
        new_global: If input_addition is start of new input in sequence (add to global tuple)
        """
        self.max_arg_len = 128
        self.action_space = gym.spaces.Dict({
            "input_addition": gym.spaces.Text(max_length=self.max_arg_len),
            "exec_action": gym.spaces.Discrete(2),  # 0 or 1
            "new_global": gym.spaces.Discrete(2),  # 0 or 1
        })

        # monitored by constructed_input
        self.observation_space = gym.spaces.Tuple([
                gym.spaces.Tuple([gym.spaces.Text(max_length=self.max_arg_len) for _ in range(self.max_local)])
                for _ in range(self.max_global)
            ])

    def step(self, action: dict) -> Tuple[tuple, float, bool, bool, dict]:
        trunc = False
        done = False

        # Validate action conditions
        if action["exec_action"] == 1:
            if action["new_global"] == 1:
                return (
                    self.constructed_input,
                    -10.,  # Apply a penalty for invalid action
                    done,
                    trunc,
                    {"error": "exec_action flag and new_global flag cannot both be set."},
                )
            if action["input_addition"]:
                return (
                    self.constructed_input,
                    -10.,  # Apply a penalty for invalid action
                    done,
                    trunc,
                    {"error": "input_addition must be empty when exec_action flag is set."},
                )
        if len(action["input_addition"]) > self.max_arg_len:
            return (
                self.constructed_input,
                -10.,  # Apply a penalty for invalid action
                done,
                trunc,
                {"error": f"input_addition must be < {self.max_arg_len}."},
            )

        # Convert constructed_input to list for modification
        self.constructed_input = self._tuple_to_list(self.constructed_input)

        # Ensure constructed_input is initialized
        if not self.constructed_input:
            self.constructed_input = [list([""] * self.max_local) for _ in range(self.global_horizon)]

        if action["exec_action"] == 0:  # Append to constructed_input
            if action["new_global"] == 0:  # Append to current local command
                # Check if the local horizon is exceeded
                # Append input_addition to the current local sequence
                self.curr_local += 1  # Increment local position

                if self.curr_local >= self.local_horizon:
                    trunc = True

                self.constructed_input[self.curr_global][self.curr_local] = action["input_addition"]  # add arg to constructed input state
                if self.intermed_exec:
                    post_output, post_output_code, _et = self.get_cmd_output()  # execute seq
                    post_context_code, post_context = get_env_context(self.image_name,
                                                                      [" ".join(_) for _ in
                                                                       self.clean_input_seq()],
                                                                      timeout=self.timeout,
                                                                      verbose=self.verbose)

                    if post_output_code != 0:  # if error sequence, immediate err reward
                        redund_delta = 0.0  # set max redundant, let error silently propagate to end check or correct with future args
                        self.prev_redund_score = [0.0, 0.0]  # if constant error, will propagate out
                    else:
                        l_opts = self.curr_local
                        output_redundancy_score = 1.0 - self.get_n_output_redundant(post_output, post_output_code) / l_opts  # get ratio of options redundant with respect to final output
                        context_redundancy_score = 1.0 - self.get_n_context_redundant(post_context) / l_opts  # get ratio of options redundant with respect to final context
                        redundancy_reward = max(output_redundancy_score, context_redundancy_score)  # MAX OR EQUAL AMOUNT?
                        # proportion went down, new arg is redundant
                        redund_delta = redundancy_reward - max(self.prev_redund_score[0], self.prev_redund_score[1])
                        if redund_delta == 0.0 and redundancy_reward == 1.0:
                            redund_delta = 1.0  # no drop in perfect sequence

                        self.prev_redund_score = [output_redundancy_score, context_redundancy_score]

                    self.prev_output = post_output
                    self.prev_exit_code = post_output_code
                    self.prev_context = post_context
                    self.prev_exec_time = _et

                    reward = self._get_intermed(self.curr_local) * redund_delta
                else:
                    reward = 0.0
                # immutable type for gym return
                self.constructed_input = self._list_to_tuple(self.constructed_input)

                return (
                   self.constructed_input,
                   reward,
                   done,
                   trunc,
                   {}
                )

            elif action["new_global"] == 1:  # Start a new global command
                self.curr_local = 0
                self.curr_global += 1

                self.prev_output = ""
                self.prev_exit_code = 0
                self.prev_exec_time = 0.0
                self.prev_context = {}
                self.prev_redund_score = []
                if self.curr_global >= self.global_horizon:  # if over global horizon from last addition, end early
                    self.constructed_input = self._list_to_tuple(self.constructed_input)
                    return (
                        self.constructed_input,
                        0.0,
                        True,
                        True,
                        {}
                    )
                else:
                    _pgc, self.prev_global_context = get_env_context(self.image_name, [" ".join(_) for _ in self.clean_input_seq()], timeout=self.timeout, verbose=self.verbose)
                    self.constructed_input[self.curr_global][self.curr_local] = action["input_addition"]
                    if self.intermed_exec:
                        post_output, post_output_code, _et = self.get_cmd_output()
                        post_context_code, post_context = get_env_context(self.image_name, [" ".join(_) for _ in self.clean_input_seq()], timeout=self.timeout, verbose=self.verbose)
                        self.prev_output = post_output
                        self.prev_exit_code = post_output_code
                        self.prev_context = post_context
                        self.prev_exec_time = _et
                        self.prev_redund_score = [0.0, 0.0]  # nothing to compare to
                    self.constructed_input = self._list_to_tuple(self.constructed_input)

                    return (
                        self.constructed_input,
                        0.0,  # small punish to drive towards arg selection
                        done,
                        trunc,
                        {}
                    )
            else:
                raise ValueError("Invalid new_global flag value.")

        # full intermed exec, not cutoff
        if self.intermed_exec:
            output = self.prev_output
            code = self.prev_exit_code
            final_context = self.prev_context
            exec_time = self.prev_exec_time
            output_redundancy_score = self.prev_redund_score[0]
            context_redundancy_score = self.prev_redund_score[1]
        else:
            output, code, exec_time = self.get_cmd_output()
            final_context_code, final_context = get_env_context(self.image_name, [" ".join(_) for _ in self.clean_input_seq()], timeout=self.timeout, verbose=self.verbose)
            try:
                l_opts = self.curr_local
                output_redundancy_score = 1.0 - self.get_n_output_redundant(output, code) / l_opts  # get ratio of options redundant with respect to final output
                context_redundancy_score = 1.0 - self.get_n_context_redundant(final_context) / l_opts  # get ratio of options redundant with respect to final context

            except ZeroDivisionError:  # only utility in seq, nothing to compare to
                output_redundancy_score = 0.5
                context_redundancy_score = 0.5

        if code == 0:  # successful exec
            redundancy_reward = max(output_redundancy_score, context_redundancy_score)  # MAX OR EQUAL AMOUNT?
            # final diff information to be passed back in info
            _b, context_key, context_value = self.compare_diff_context(e1=self.prev_global_context, e2=final_context)
            exit_reward = 1.0  # don't need to check since handled by conditional
            exit_pen = 0.0
        else:  # error code
            output_redundancy_score = 0.0
            context_redundancy_score = 0.0
            context_key, context_value = "none", ""

            redundancy_reward = 0.0
            exit_reward = -1.0
            exit_pen = -1.0

        trunc = True  # trunc to clear agent buffer and move on to next seq or end
        self.constructed_input = self._list_to_tuple(self.constructed_input)
        info_input_seq = squeeze_sequence(self.constructed_input)

        # scale by final redunancy calc, subtract for error code
        reward = self._get_final_mass() * (redundancy_reward - self.margin) + exit_pen

        return (
            self.constructed_input,
            reward,
            done,
            trunc,
            {
                "input": info_input_seq,
                "output": output, "output_len": len(output),
                "exit_code": code, "exit_score": exit_reward,
                "redundancy_score": redundancy_reward,
                "output_redundancy_score": output_redundancy_score, "context_redundancy_score": context_redundancy_score,
                "context_key": context_key,
                "context_value": context_value,
                "exec_time": exec_time,
                "cwd": self.get_start_cwd(),
                "image": self.get_image_name()},
        )

        # return observation, reward, terminated, truncated, info

    def get_n_context_redundant(self, curr_context: dict) -> int:
        """ Checks for context change before and after last command. If different, does the same with each option to find redundant options in sequence """
        *prev_cmds, last_cmd = self.clean_input_seq()  # all but last command to be sent. last command will be edited

        is_same_context, context_key, context_value = self.compare_diff_context(e1=self.prev_global_context, e2=curr_context)
        # Checks context of command before last sequence and after last sequence.
        if is_same_context:
            return self.curr_local  # context did not change, no reason to check option redundancies, return max redundant

        n_redundant = 0
        for opt in range(1, len(last_cmd)):  # change 1 to 0 if don't want to hold utility constant
            ref_cmd = last_cmd[:opt] + last_cmd[opt + 1:]  # iterate removing one option from list, run subseq, compare env. if same -> increment redundant count.
            test_seq = prev_cmds + [ref_cmd]

            n_tries = 0
            t_code, t_c = get_env_context(self.image_name, [" ".join(_) for _ in test_seq], timeout=self.timeout, verbose=self.verbose)
            while t_code != 0 and n_tries < self.MAX_TRIES:
                n_tries += 1
                t_code, t_c = get_env_context(self.image_name, [" ".join(_) for _ in test_seq], timeout=self.timeout, verbose=self.verbose)
                self.raise_try_limit(n_tries)

            if self.compare_diff_context(e1=self.prev_global_context, e2=t_c) == self.compare_diff_context(e1=self.prev_global_context, e2=curr_context):
                n_redundant += 1
        return n_redundant

    def compare_diff_context(self, e1: dict, e2:dict) -> Tuple[bool, str, str]:
        """ Returns true if every element in both environments match. Returns key and environment matches for episode info dict"""
        try:
            for _ in e1.keys():
                if type(e1[_]) != type(e2[_]):  # not of same type, most likely error throwing off alignment
                    raise TypeError(f"Type mismatch: {type(_[0])}, {type(_[1])}")
                elif isinstance(e1[_], (int, float)):
                    if e1[_] != e2[_]:
                        return False, _, f"{e1[_]} - {e2[_]}"
                elif isinstance(e1[_], str):
                    if _ == "env":  # only for the env-block, strip changing var (OLDPWD,add if more found)
                        e1[_] = _strip_dynamic_env_vars(e1[_])
                        e2[_] = _strip_dynamic_env_vars(e2[_])
                    diff_str = get_context_diff(e1[_], e2[_])
                    if diff_str:
                        return False, _, diff_str
                elif isinstance(e1[_], bool):
                    if not e1[_] == e2[_]:
                        return False, _, f"{e1[_]} - {e2[_]}"
                else:
                    raise TypeError(f"Type not handled: {type(_[0])}")
        except KeyError:  # context-gathering wrapper broke due to poorly formatted sequence, return True to not reward being different
            return True, "error", ""
        return True, "none", ""

    def get_n_output_redundant(self, ref_output: str, ref_code: int) -> int:
        """
        - Measures redundancy of options in sequence by comparing reference output with output from one option removed
        - Using beta derived from repeat execution noise (0.95 base)
        """
        *prev_cmds, last_cmd = self.clean_input_seq()
        if len(last_cmd) < 2:
            return self.curr_local  # nothing to test, only command, set max redundant to not reward early stop

        full_seq = prev_cmds + [last_cmd]
        beta = max(0.2, self._dynamic_beta([" ".join(_) for _ in full_seq]) - 0.05)  # -0.05 for margin

        n_redundant = 0
        for idx in range(1, len(last_cmd)):  # skip util at position 0
            test_cmd = last_cmd[:idx] + last_cmd[idx + 1:]
            test_seq = prev_cmds + [test_cmd]

            out, code, _ = self.get_cmd_output([" ".join(_) for _ in test_seq])
            if code != ref_code:  # removal caused error, not redundant
                continue
            sim = calc_lev_sim_single(_normalise_output(out), _normalise_output(ref_output), max_len=8192)
            if sim >= beta:  # sim above noise threshold, did not change output enough to be sure.
                n_redundant += 1

        return n_redundant

    def _dynamic_beta(self, cmd_seq) -> float:
        """
        Run cmd_seq self.noise_runs times in fresh containers,
        compute pair-wise Levenshtein similarities, and return beta = max(0.0, mean − k*std)

        so that any similarity >= beta is treated as noise.
        """
        if self.noise_runs < 2:
            return 1.0  # degenerates to “exact match”

        outs = []
        for _ in range(self.noise_runs):
            out, code, _ = self.get_cmd_output(cmd_seq, bypass_cache=True)  # get fresh outputs each time, skip cache
            outs.append(_normalise_output(out))

        # pair‑wise similarities between all runs of the same command
        sims = [calc_lev_sim_single(a, b, max_len=8192)
                for i, j in combinations(range(len(outs)), 2)
                for a, b in [(outs[i], outs[j])]]

        mean = statistics.mean(sims)
        std = statistics.stdev(sims) if len(sims) > 1 else 0.0
        beta = max(0.2, mean - self.noise_sigmaK * std)  # min diff beta at 0.2
        return beta

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Tuple[str], Dict[Any, Any]]:
        super().reset(seed=seed)
        self.start_cwd = self.working_dirs[random.randint(0, len(self.working_dirs) - 1)]
        self.constructed_input = tuple(
            tuple(["cd", self.start_cwd] + [""] * (self.max_local - 2)) if i == 0
            else tuple([""] * self.max_local)
            for i in range(self.max_global)
        )  # initialize with first command changing to start cwd path, empty command strings up to max_global
        self.curr_local = 0
        self.curr_global = 0

        # reset memory
        self.prev_output = ""
        self.prev_exit_code = 0
        self.prev_exec_time = 0.0
        self.prev_context = {}
        self.prev_redund_score = []
        self.prev_global_context = {}

        self._seq_cache.clear()  # fresh cache every episode

        return self.constructed_input, {}

    def clean_input_seq(self) -> List[List[str]]:
        """ Convert padded state to non-padded 2d list"""
        return [list(filter(None, sublist)) for sublist in self.constructed_input if any(sublist)]

    def get_start_cwd(self) -> str:
        return self.start_cwd

    def get_max_global(self) -> int:
        return self.global_horizon - 1

    def get_curr_cwd(self) -> str:
        """ Gets current cwd from start cwd given list of cmds"""
        curr_cwd = self.get_start_cwd()
        cmd_list = self.clean_input_seq()
        for _ in cmd_list:
            curr_cwd = update_cwd(" ".join(_), curr_cwd)
        return curr_cwd

    def get_prev_cwd(self) -> str:
        """ Gets cwd from second to most recent input """
        curr_cwd = self.get_start_cwd()
        cmd_list = self.clean_input_seq()[:-1]
        for _ in cmd_list:
            curr_cwd = update_cwd(" ".join(_), curr_cwd)
        return curr_cwd

    def get_cmd_output(self, send_seq: list = None, bypass_cache: bool = False) -> Tuple[str, int, float]:
        """ Get from cache or attempt send_seq MAX_TRIES times before returning error """
        send_seq = [" ".join(_) for _ in self.clean_input_seq()] if send_seq is None else send_seq
        cache_key = self._seq_key(send_seq)

        # Serve from cache
        if not bypass_cache and self._seq_cache[cache_key]:
            # pick a random past result so the timing distribution remains “noisy”
            out, code, t = random.choice(self._seq_cache[cache_key])
            return out, code, t

        # run in a fresh container
        n_tries, code, out, exec_time = 0, -1, "", 0.0
        while code == -1 and n_tries < self.MAX_TRIES:
            n_tries += 1
            start_t = time.time()
            out, code = simple_send_cmds_docker(
                image_name=self.image_name,
                cmds=send_seq,
                timeout=self.timeout,
                max_output=8192,
                verbose=False  # self.verbose
            )
            exec_time = time.time() - start_t
            if code == -1:
                sleep(0.1)
            self.raise_try_limit(n_tries)

        # Store (bounded) in cache
        bucket = self._seq_cache[cache_key]
        bucket.append((out, code, exec_time))
        if len(bucket) > self.cache_maxlen:
            bucket.pop(0)

        return out, code, exec_time

    def get_image_name(self) -> str:
        return self.image_name

    @staticmethod
    def _seq_key(cmds: List[str]) -> Tuple[str]:
        return tuple(_WS_RE.sub(" ", c).strip() for c in cmds)

    @staticmethod
    def _tuple_to_list(obs: Tuple[Tuple[str]]) -> List[List[str]]:
        return list(list(_) for _ in obs)

    @staticmethod
    def _list_to_tuple(obs: List[List[str]]) -> Tuple[Tuple[str]]:
        return tuple(tuple(_) for _ in obs)

    def raise_try_limit(self, n_tries: int):
        if n_tries >= self.MAX_TRIES:
            raise RuntimeError(f"cmd execution limit reached for {squeeze_sequence(self.constructed_input)}. Retry episode")

    def _get_intermed(self, loc) -> float:
        return 1 / self.max_local

    def _get_final_mass(self, m=None) -> float:
        m = self.curr_local if m is None else m
        return m / self.max_local
