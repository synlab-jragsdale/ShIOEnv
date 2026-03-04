import itertools
import json
import re
import statistics
from itertools import combinations
from time import sleep
from typing import Optional, Tuple, List, Any, Dict
from collections import OrderedDict
import jsonpatch

import gymnasium as gym
import random

from ShIOEnv.docker_executor import DockerExecutor
from ShIOEnv.firecracker_executor import FirecrackerExecutor
from ShIOEnv.utils import calc_lev_sim_single, squeeze_sequence, update_cwd, _normalize_output


class RetryError(Exception):
    pass


def _strip_dynamic_env_vars(env_str: str) -> str:
    """
    Remove variables whose value legitimately changes from one command to the next.
    In env variables
    """
    IGNORE_PREFIXES = ("OLDPWD=", "_=")  # add PWD= if tracked
    return "\n".join(
        line for line in env_str.splitlines()
        if not any([_ in line for _ in IGNORE_PREFIXES])
    )


class ShIOEnv(gym.Env):
    def __init__(self, env_config: dict, exec_config: dict, working_dirs: List[str], verbose: bool = False):
        self.env_config = env_config
        self.exec_config = exec_config

        self.MAX_TRIES = 3
        self.cache_size = env_config.get("cache_size", 1024)
        # Global LRU cache: key(str) -> (output, exit_code, context)
        self._seq_cache: OrderedDict[str, tuple[str, int, dict]] = OrderedDict()  # cache of execution traces to skip execution
        self._beta_cache = OrderedDict()  # cache of str: float for past beta calculations

        self.noise_runs = env_config.get("noise_runs", 5)  # repeat full seq N times to get noise estimate for beta
        self.noise_sigmaK = env_config.get("noise_sigmaK", 2.0)  # k*std below the mean for beta

        self.hit_horizon = False  # switch for if to apply horizon penalty (set to true if trunc)

        self.calc_redundant = env_config.get("calc_redundant", True)  # Check redundancy for full sequence

        self.verbose = verbose

        self.horizon_margin = env_config.get('horizon_margin', 6)  # how much leeway to allow to incomplete arguments when trunc is sent
        self.max_global = max(1, env_config.get('global_horizon', 1)) + 1
        self.max_local = env_config.get('local_horizon', 5) + self.horizon_margin  # max size for each input seq

        self.global_horizon = max(self.max_global, 2)  # first index taken by directory move
        self.local_horizon = env_config.get('local_horizon', 5)

        self._beta = 0.95
        # position counters
        self.curr_global = 0
        self.curr_local = 1

        # docker exec timeout before trap
        self.timeout = env_config.get('timeout', 10)

        exec_method = env_config.get('exec_method', 'docker')
        self.vm_executor = FirecrackerExecutor(**exec_config) if exec_method == 'firecracker' else DockerExecutor(**exec_config)
        self.vm_executor.setup()
        # starting directories, uniformly sampled
        self.working_dirs = working_dirs
        self.ctx_exclude_paths = ("/fs/./*", "/fs/../*", "/settings/PPID", "/usage/*", "/net_conns/**", "/datetime")
        self._compiled_ctx_exclude = self._compile_patterns(self.ctx_exclude_paths)
        # self.ctx_comp_exclude_paths = ("/fs/./*", "/fs/../*", "/net_ports/**", "/net_tx_bytes", "/net_rx_bytes", "/cpu_used", "/io_wait", "/datetime")

        # changed on reset
        self.pre_exec_context = self.vm_executor.get_start_context()
        self.start_cwd = self.working_dirs[random.randint(0, len(self.working_dirs) - 1)]
        # current state
        self.constructed_input = ()

        self.get_final_score = env_config.get("get_final_score", True)
        # sub-input sample budget (suffix samples)
        self.contrib_samples = int(env_config.get("contrib_samples", 16))
        self.sample_all_combs = int(env_config.get("sample_all_combs", False))

        """
        input_addition: new argument to be added to command sequence.
        exec_action: Flag for if sequence is ready to be run (triggered no arguments left)
        new_global: If input_addition is start of new input in sequence (add to global tuple)
        """
        self.last_action = None  # for error reporting
        self.last_state = None
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

    def close(self):
        self.vm_executor.close()
        self.vm_executor.shutdown()

    def step(self, action: dict) -> Tuple[tuple, float, bool, bool, dict]:
        trunc = False
        done = False
        self.last_action = action
        self.last_state = self.constructed_input

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
                if self.verbose:
                    print(f"[*] [ShIOEnv.step] local argument received: {action['input_addition']}")
                # Check if the local horizon is exceeded
                # Append input_addition to the current local sequence
                self.curr_local += 1  # Increment local position

                if self.curr_local >= self.local_horizon:
                    if self.verbose:
                        print(f"[*] [ShIOEnv.step] Soft local horizon limit reached")
                    self.hit_horizon = True
                    trunc = True

                self.constructed_input[self.curr_global][self.curr_local] = action["input_addition"]  # add arg to constructed input state
                # immutable type for gym return
                self.constructed_input = self._list_to_tuple(self.constructed_input)

                if self.verbose:
                    print(f"[*] [ShIOEnv.step] {action['input_addition']} added to state")

                return (
                   self.constructed_input,
                   0.0,
                   done,
                   trunc,
                   {"input": "; ".join([self.join_cmd_args(_) for _ in self.clean_input_seq()][1:]), "image": self.get_executor_id()}
                )

            elif action["new_global"] == 1:  # Start a new global command
                if self.verbose:
                    print(f"[*] [ShIOEnv.step] Global argument received: {action['input_addition']}")
                self.curr_local = 0
                self.curr_global += 1

                if self.curr_global >= self.global_horizon:  # if over global horizon from last addition, end early
                    if self.verbose:
                        print(f"[*] [ShIOEnv.step] Soft global horizon limit reached")
                    self.constructed_input = self._list_to_tuple(self.constructed_input)
                    return (
                        self.constructed_input,
                        0.0,
                        True,
                        True,
                        {"input": "; ".join([self.join_cmd_args(_) for _ in self.clean_input_seq()][1:]), "image": self.get_executor_id()}
                    )
                else:
                    self.constructed_input[self.curr_global][self.curr_local] = action["input_addition"]
                    self.constructed_input = self._list_to_tuple(self.constructed_input)
                    if self.verbose:
                        print(f"[*] [ShIOEnv.step] {action['input_addition']} added to state")

                    return (
                        self.constructed_input,
                        0.0,
                        done,
                        trunc,
                        {"input": "; ".join([self.join_cmd_args(_) for _ in self.clean_input_seq()][1:]), "image": self.get_executor_id()}
                    )
            else:
                raise ValueError("Invalid new_global flag value.")

        # operating context after last command's execution
        try:
            if self.verbose:
                print(f"[*] [ShIOEnv.step] Getting behavior for last 2 cmds ({self.clean_input_seq()[-1]})")
            full_seq = [self.join_cmd_args(_) for _ in self.clean_input_seq()]
            output, code, executing_context, post_ctx = self.vm_executor.get_env_context_pair(
                full_seq,
                pre_exec_context=self.pre_exec_context,
                timeout=max(self.timeout, 180),
            )
        except ValueError as e:
            print(self.clean_input_seq())
            raise e
        if self.verbose:
            print(f"[*] [ShIOEnv.step] Getting context diffs")
        ctx_patch = self.get_ctx_diffs(executing_context, post_ctx, exclude=self._compiled_ctx_exclude)  # self.ctx_exclude_paths
        reward = 0.0
        info_required = {"redundancy_score": 0.0}

        if self.get_final_score:
            if self.verbose:
                print(f"[*] [ShIOEnv.step] Estimating per-seq scores")
            base_reward = self._estimate_contribs_all_args(pre_exec_ctx=executing_context, sample_all=False)
            redund_score = base_reward
            if self.sample_all_combs:
                if self.verbose:
                    print(f"[*] [ShIOEnv.step] Getting true per-seq scores")
                redund_score = self._estimate_contribs_all_args(pre_exec_ctx=executing_context, sample_all=True)
            score_err = redund_score - base_reward
            reward = base_reward

            info_required = {
                "redundancy_score": redund_score,
                "redund_error": score_err,  # how far off contrib prediction was
                "output_redundancy_beta": self._beta,
            }

        trunc = True  # trunc to clear agent buffer and move on to next seq or end
        self.constructed_input = self._list_to_tuple(self.constructed_input)
        clean_seq = self.clean_input_seq()

        return (
            self.constructed_input,
            reward,
            done,
            trunc,
            {
                "input": "; ".join([self.join_cmd_args(_) for _ in clean_seq][1:]),  # 1: to get rid of cwd change
                "input_args": self.ensure_safe_cmd(clean_seq[-1]),
                "output": output,
                "exit_code": code,
                "context_patch": ctx_patch,
                **info_required,  # unified
                "image": self.get_executor_id(),
            }
        )

    @staticmethod
    def ensure_safe_cmd(cmd:List[str]) -> List[str]:
        """ Remove last parts of last arg if will break executor (unclosed redirect/lop)"""
        unsafe_end = [">", "|", "&", ";"]
        unsafe_arg = lambda x: any([x.endswith(u) and not x.endswith("<ns>") for u in unsafe_end])
        last_arg = cmd[-1]
        while unsafe_arg(last_arg):
            last_arg = last_arg[:-1]  # remove last char for next check
            if not last_arg:  # if remove makes empty string, pop off and check with prev arg
                cmd = cmd[:-1]
                if not cmd:  # if remove makes empty input, throw error
                    raise AttributeError("No safe arguments to end sequence")
                last_arg = cmd[-1]
        return cmd

    def join_cmd_args(self, cmd: List[str]) -> str:
        """
        Join a list of command arguments into a single string. whitespace separated unless <ns> on joining side
        e.g.,
        - [arg1<ns>, <ns>arg2] -> arg1arg2
        - [arg1<ns>, arg2] -> arg1 arg2
        """
        cmd = self.ensure_safe_cmd(cmd)  # pop off last arg if breaking
        r_cmd = cmd[0]  # base cmd (assumed)
        arg_end_adj = [True if _.endswith("<ns>") else False for _ in cmd]
        arg_start_adj = [True if _.startswith("<ns>") else False for _ in cmd]

        can_join = lambda i, j: arg_end_adj[i] and arg_start_adj[j]  # [arg1]<ns> <ns>[arg2] -> [arg1][arg2]
        right_ns = lambda i, j: not can_join(i, j) and arg_start_adj[j]  # [arg1] <ns>[arg2] -> [arg1] -[arg2]
        for i in range(1, len(cmd)):
            # Check for [i-1]<ns>, <ns>[i]
            if can_join(i - 1, i):
                r_cmd += cmd[i].replace("<ns>", "")
            elif right_ns(i - 1, i):
                r_cmd += " -" + cmd[i].replace("<ns>", "")
            else:
                r_cmd += " " + cmd[i].replace("<ns>", "")

        return r_cmd

    def get_arg_ranges(self, cmd: List[str]) -> List[Tuple[int, int]]:
        """
        Return half-open index ranges [start, end) for each independent portion of the
        command arguments, suitable for slicing: cmd[start:end].

        Rules implemented:
        - If an argument ends with ';', include it in the current range, then start a new range.
        - If an argument is exactly a sep_op, or ends with a sep_op (other than the ';' case above),
          do not include that argument in any range (neither current nor next).
        - If an argument contains a sep_op but does not end with one, end the current range
          (excluding this argument) and start the next range at this argument (i.e., include it in next).
        - Otherwise, include the argument in the current, un-terminated range.

        The returned ranges are always half-open [start, end). Empty ranges are never emitted.
        """
        sep_ops = ["|", "||", ";", "&&"]
        stop_ops = [">", ">>"]  # stop if ends argument

        if not cmd:
            return []

        def ends_with_any_op(s: str) -> bool:
            return any(s.endswith(op) for op in sep_ops)

        def contains_any_op(s: str) -> bool:
            return any(op in s for op in sep_ops)

        ranges: List[Tuple[int, int]] = []
        start = 0  # start index of the current (open) range

        for i, arg in enumerate(cmd):
            # ends with ';' but not only ';' - include it in current, then split after it
            if arg.endswith(';') and arg != ';':
                # include [start, i+1)
                if start <= i + 1 and start < i + 1:
                    ranges.append((start, i + 1))
                # next range starts at the next position
                start = i + 1
                continue
            if (arg.endswith(">") and not arg.endswith("<ns>")) or arg.endswith(">>") or arg in stop_ops:  # redirect ending arg or by itself. Do not include and continue with next pos
                if start <= i + 1 and start < i + 1:
                    ranges.append((start, i))
                # next range starts at the next position
                start = i + 1
                continue

            # arg is exactly a sep_op or ends with a sep_op (but not handled by ;)
            if arg in sep_ops or (ends_with_any_op(arg)):
                # close current before i (exclude this argument)
                if start < i:
                    ranges.append((start, i))
                # skip this argument entirely. next range starts after it
                start = i + 1
                continue

            # arg contains a sep_op but does NOT end with one - include in next cmd
            if contains_any_op(arg):
                # close current before i (exclude this argument)
                if start < i:
                    ranges.append((start, i))
                # next range starts at i (include this arg in the next range)
                start = i
                continue
            # normal arg, keep accumulating (no action needed)

        # Trailing open range
        if start < len(cmd):
            ranges.append((start, len(cmd)))

        return ranges

    def _sample_suffix_subsets(self, suffix_tokens: list[str], rng: random.Random | None = None, max_tests: int = 0, **kwargs):
        """
        Yield up to max_tests unique random subsets (as lists) of suffix_tokens,
        excluding empty and full. O(max_tests) time and O(max_tests) memory.
        """
        rng = rng or random
        n = len(suffix_tokens)
        if n <= 1 or max_tests <= 0:
            return
        seen = set()
        while len(seen) < max_tests:
            # random mask in [1, 2^n - 2] (exclude empty=0 and full=(1<<n)-1)
            mask = rng.randrange(1, (1 << n) - 1)
            if mask == (1 << n) - 1:
                continue
            if mask in seen:
                continue
            seen.add(mask)
            subset = [suffix_tokens[i] for i in range(n) if (mask >> i) & 1]
            yield subset

    def _iter_all_suffix_subsets(self, suffix_tokens: list[str], **kwargs):
        n = len(suffix_tokens)
        if n <= 1:
            return
        # exclude empty (r=0) and full (r=n)
        for r in range(1, n):
            for combo in itertools.combinations(suffix_tokens, r):
                yield list(combo)


    def _estimate_contribs_all_args(self, pre_exec_ctx: dict, sample_all: bool = True) -> float:
        """
        Expectation of argument contribution by counterfactual comparison of full to argument permutation.
        Holds prior commands fixed; samples argument subset s' subseteq s.
        Returns normalized count of permutations in which changes in execution behavior were observed.
        """
        def _range_capacity(start_idx: int, end_idx: int) -> int:
            m = max(0, end_idx - start_idx - 1)
            # _unique_suffix_subsets returns 2^m including [] and full. slice [1:-1] later
            return max(0, (1 << m) - 2)

        def _fair_budget_alloc(caps: list[int], B: int) -> list[int]:
            """
            Water-filling / balanced allocation:
            - Give everyone floor(B / k)
            - Distribute the remainder from the earliest range forward
            - Respect per-range caps at every step
            guarantees no later range holds budget while an earlier one still
            has unsatisfied capacity.
            """
            k = len(caps)
            if k == 0 or B <= 0:
                return [0] * k

            # First pass: baseline fill, capped
            base = B // k
            alloc = [min(c, base) for c in caps]
            used = sum(alloc)
            rem = B - used

            # If some ranges couldn’t take base, redistribute the slack to the left first
            # Pass 1: top-up left-to-right until each reaches min(c, base)
            # Pass 2: distribute remainder one-by-one, earliest first, respecting caps
            i = 0
            while rem > 0:
                if i >= k:  # wrap if needed
                    i = 0
                    # If no one can take more, stop
                    if all(alloc[j] >= caps[j] for j in range(k)):
                        break
                if alloc[i] < caps[i]:
                    alloc[i] += 1
                    rem -= 1
                i += 1
            return alloc

        *prev_cmds, last_cmd = self.clean_input_seq()
        if not last_cmd or len(last_cmd) < 2:
            return 0.0

        iter_fn = self._iter_all_suffix_subsets if sample_all else self._sample_suffix_subsets

        all_hits, all_t = 0, 0
        eval_ranges = self.get_arg_ranges(last_cmd)  # evaluate arguments by their commands

        caps = [_range_capacity(s, e) for (s, e) in eval_ranges]
        # total feasible samples across all ranges
        total_cap = sum(caps)
        # budget
        B = min(self.contrib_samples if not sample_all else total_cap, total_cap, 1 << 15)  # hard cap
        n_ranges = _fair_budget_alloc(caps, B)

        if self.verbose:
            print(f"[*] [ShIOEnv._estimate_contribs_all_args] Eval range indices for {last_cmd}: {eval_ranges}")
        if self.verbose:
            print(f"[*] [ShIOEnv._estimate_contribs_all_args] Evaluating seqs with per-range budgets {n_ranges}")

        for i, (arg_range, N) in enumerate(zip(eval_ranges, n_ranges)):
            # arg_range: [start, end]
            start_idx, end_idx = arg_range

            full_seq = prev_cmds + [last_cmd[:end_idx]]
            if self.verbose:
                print(f"[*] [ShIOEnv._estimate_contribs_all_args] Evaluating for base seq {last_cmd[:end_idx]}")

            self._beta = self._dynamic_beta(full_seq)
            # Build constant prefix of commands before the last one
            baseline_out, baseline_code, baseline_ctx = self.get_cmd_behavior(full_seq)
            base_patch = self.get_ctx_diffs(pre_exec_ctx, baseline_ctx, exclude=self._compiled_ctx_exclude)
            base_out_norm = _normalize_output(baseline_out)

            prefix = last_cmd[:start_idx+1]
            suffix_pool = last_cmd[start_idx+1:end_idx]  # skip command (first idx usually)

            # For each i, estimate E[ delta(F(P + {xi} + S), F(P + S)) ]
            hits, t = 1, 1
            for S in iter_fn(suffix_pool, max_tests=N):
                seq_wo = prev_cmds + [prefix + S]
                if self.verbose:
                    print(f"[*] [ShIOEnv._estimate_contribs_all_args] Evaluating reduced seq {prefix} [+] {S}")
                # compute only the "without" side

                out_0, code_0, ctx_0 = self.get_cmd_behavior(seq_wo)
                diff_code = (baseline_code != code_0)
                diff_out = diff_ctx = False
                if baseline_code == code_0 == 0:  # exit code matched, need to check artifacts
                    sim = calc_lev_sim_single(base_out_norm, _normalize_output(out_0), max_len=8192)
                    diff_out = (sim < self._beta)
                    if not diff_out:  # output same, need to check diff
                        diff_ctx = (base_patch != self.get_ctx_diffs(pre_exec_ctx, ctx_0, exclude=self._compiled_ctx_exclude))
                delta = int(diff_out or diff_ctx or diff_code)
                hits += delta * len(S)
                t += 1 * len(S)
            all_hits += hits
            all_t += t

        return all_hits / max(all_t, 1)

    def filter_json_patch_ops(self, ops_json_or_list, patterns, mode="exclude", check_from=True):
        ops = json.loads(ops_json_or_list) if isinstance(ops_json_or_list, str) else ops_json_or_list

        regs = patterns
        m = lambda op: ("path" in op and self._matches_any(op["path"], regs)) or (
                check_from and "from" in op and self._matches_any(op["from"], regs))

        kept = [o for o in ops if (not m(o) if mode == "exclude" else m(o))]
        return json.dumps(kept, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _compile_patterns(patterns):
        def pat2re(p):
            p = p if p.startswith("/") else "/" + p
            s = re.escape(p).replace(r"\*\*", ".*").replace(r"\*", "[^/]+")
            return re.compile("^" + s + "$")

        return [pat2re(p) for p in patterns]

    @staticmethod
    def _matches_any(path, regs):
        return any(r.match(path) for r in regs)

    def get_ctx_diffs(self, e1, e2, exclude=()):
        regs = exclude

        def esc(k):
            return k.replace("~", "~0").replace("/", "~1")

        def get(doc, ptr):
            cur = doc
            if not ptr or ptr == "/": return cur
            for t in ptr.split("/")[1:]:
                t = t.replace("~1", "/").replace("~0", "~")
                cur = cur[int(t)] if isinstance(cur, list) else cur[t]
            return cur

        is_excluded = lambda p: self._matches_any((p.rstrip("/") or "/"), regs)

        # a path has an excluded descendant if ANY exclude pattern can match some child of p
        has_ex_desc = lambda p: any(r.match(p) or r.match(p + "/x") for r in regs)

        ops = [op for op in jsonpatch.make_patch(e1, e2).patch if not is_excluded(op["path"])]

        out = []
        i = 0
        while i < len(ops):
            op = ops[i]
            i += 1
            if op["op"] != "replace":
                out.append(op)
                continue

            path = op["path"]
            if not has_ex_desc(path):
                out.append(op)
                continue

            old, new = get(e1, path), get(e2, path)
            if not (isinstance(old, dict) and isinstance(new, dict)):
                out.append(op)
                continue  # lists: keep coarse replace

            ok, nk = set(old), set(new)

            for k in ok - nk:
                p = f"{path}/{esc(k)}"
                if not is_excluded(p):
                    out.append({"op": "remove", "path": p})

            for k in nk - ok:
                p = f"{path}/{esc(k)}"
                if not is_excluded(p):
                    out.append({"op": "add", "path": p, "value": new[k]})

            for k in ok & nk:
                p = f"{path}/{esc(k)}"
                if is_excluded(p): continue
                ov, nv = old[k], new[k]
                if ov == nv: continue
                if isinstance(ov, dict) and isinstance(nv, dict) and has_ex_desc(p):
                    # queue a synthetic replace to expand recursively
                    ops.append({"op": "replace", "path": p, "value": nv})
                else:
                    out.append({"op": "replace", "path": p, "value": nv})

        return self.compact_patch_min(self.filter_json_patch_ops(out, patterns=exclude))

    # --- Minimal Compact RFC6902 (ops-only) ---
    # Encoded form (list of arrays):
    #   ["a", "/path", value]  # add
    #   ["=", "/path", value]  # replace
    #   ["t", "/path", value]  # test
    #   ["r", "/path"]         # remove
    #   ["m", "/from", "/to"]  # move
    #   ["c", "/from", "/to"]  # copy

    _OP2CODE = {"add": "a", "replace": "=", "test": "t", "remove": "r", "move": "m", "copy": "c"}
    _CODE2OP = {v: k for k, v in _OP2CODE.items()}

    def compact_patch_min(self, ops_json_or_list: str | list[dict]) -> str:
        """RFC6902 ops -> compact list-of-arrays JSON (minified string)."""
        ops = json.loads(ops_json_or_list) if isinstance(ops_json_or_list, str) else ops_json_or_list
        out = []
        for op in ops:
            code = self._OP2CODE.get(op["op"])
            if code in ("a", "=", "t"):
                out.append([code, op["path"], op.get("value")])
            elif code == "r":
                out.append([code, op["path"]])
            elif code in ("m", "c"):
                out.append([code, op["from"], op["path"]])
            else:
                raise ValueError(f"Unsupported op: {op['op']}")
        return json.dumps(out, separators=(",", ":"), ensure_ascii=False)

    def _dynamic_beta(self, seq: List[List[str]]) -> float:
        """
        Run cmd_seq self.noise_runs times in fresh containers,
        compute pair-wise Levenshtein similarities, and return beta = max(0.0, mean − k*std)

        so that any similarity >= beta is treated as noise.
        """
        # LRU check
        key = self._seq_key(seq)
        send_seq = [self.join_cmd_args(_) for _ in seq]

        if self.verbose:
            print(f"[*] [ShIOEnv._dynamic_beta] Getting output noise beta for input {send_seq[-1]}")
        if self.noise_runs < 2:
            return 1.0  # degenerates to exact match

        if key in self._beta_cache:
            v = self._beta_cache.pop(key)
            self._beta_cache[key] = v
            return v

        outs = self.vm_executor.run_block_repeat(send_seq, repeat=self.noise_runs, pre_exec_context=self.pre_exec_context, timeout=max(self.timeout, 180))
        outs = [_normalize_output(o) for o in outs]

        # pair‑wise similarities between all runs of the same command
        sims = [calc_lev_sim_single(a, b, max_len=8192)
                for i, j in combinations(range(len(outs)), 2)
                for a, b in [(outs[i], outs[j])]]

        mean = statistics.mean(sims)
        std = statistics.stdev(sims) if len(sims) > 1 else 0.0
        beta = max(0.2, mean - self.noise_sigmaK * std)  # min diff beta at 0.2

        # LRU insert
        if key in self._beta_cache: self._beta_cache.pop(key)
        self._beta_cache[key] = beta
        while len(self._beta_cache) > self.cache_size:
            self._beta_cache.popitem(last=False)

        if self.verbose:
            print(f"[*] [ShIOEnv._dynamic_beta] {send_seq[-1]} | BETA={beta} (n={self.noise_runs} mean={mean}, std={std})")
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
        self.hit_horizon = False
        self._beta = 0.95
        self.pre_exec_context = self.vm_executor.get_start_context()  # prelim commands to setup executing environment, call on each reset

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
            curr_cwd = update_cwd(self.join_cmd_args(_), curr_cwd)
        return curr_cwd

    def get_prev_cwd(self) -> str:
        """ Gets cwd from second to most recent input """
        curr_cwd = self.get_start_cwd()
        cmd_list = self.clean_input_seq()[:-1]
        for _ in cmd_list:
            curr_cwd = update_cwd(self.join_cmd_args(_), curr_cwd)
        return curr_cwd


    def get_cmd_behavior(self, seq: List[List[str]], bypass_cache: bool = False) -> Tuple[str, int, dict]:
        """
        Get behavior for a command sequence.
        Uses a global LRU cache to store traces for equivalent comparisons.
        If bypass_cache=True, execute without touching the cache (used for noise estimation).
        """
        cache_key = self._seq_key(seq)
        send_seq = [self.join_cmd_args(_) for _ in seq]
        if self.verbose:
            print(f"[*] [ShIOEnv.get_cmd_behavior] Getting execution behavior for {send_seq}")

        # Serve from global LRU cache
        if not bypass_cache and cache_key in self._seq_cache:
            if self.verbose:
                print(f"[*] [ShIOEnv.get_cmd_behavior] LRU cache hit (k={cache_key})")
            # Mark as recently used
            out, code, ctx = self._seq_cache.pop(cache_key)
            self._seq_cache[cache_key] = (out, code, ctx)
            return out, code, ctx


        n_tries, code, out, ctx = 0, -1, "", {}
        while code == -1 and n_tries < self.MAX_TRIES:
            n_tries += 1
            try:
                code, out, ctx = self.vm_executor.get_env_context(cmds=send_seq, pre_exec_context=self.pre_exec_context)
            except ValueError:
                print(f"Value error when parsing env features for {send_seq} (state: {self.clean_input_seq()}). Trying again...")
                pass
            if code == -1:
                sleep(0.1)
            self.raise_try_limit(n_tries, send_seq, f'OUT: {repr(out[:50])}\ncode: {code}')
        # Store in global LRU cache unless explicitly bypassed
        if not bypass_cache:
            # Insert / refresh as most recently used
            if cache_key in self._seq_cache:
                self._seq_cache.pop(cache_key)
            self._seq_cache[cache_key] = (out, code, ctx)
            # Enforce max length by evicting least-recently-used
            while len(self._seq_cache) > self.cache_size:
                # popitem(last=False) pops the oldest (LRU) entry
                self._seq_cache.popitem(last=False)

        return out, code, ctx

    def get_executor_id(self) -> str:
        return self.vm_executor.base_id

    @staticmethod
    def _seq_key(cmds: List[List[str]]) -> str:
        return '; '.join(
            ' '.join(arg.replace('<ns>', '').lstrip('-') for arg in cmd)
            for cmd in cmds
        )

    @staticmethod
    def _tuple_to_list(obs: Tuple[Tuple[str]]) -> List[List[str]]:
        return list(list(_) for _ in obs)

    @staticmethod
    def _list_to_tuple(obs: List[List[str]]) -> Tuple[Tuple[str]]:
        return tuple(tuple(_) for _ in obs)

    def raise_try_limit(self, n_tries: int, send_seq: list = None, msg:str="") -> None:
        if n_tries >= self.MAX_TRIES:
            raise RetryError(f"cmd execution limit reached for {send_seq or squeeze_sequence(self.constructed_input)}. Retry episode. Optional msg: {msg}")
