import difflib
import json
from collections import OrderedDict
from typing import Tuple, List, Dict
import re
from math import floor, log

import docker
import numpy as np
import Levenshtein
import torch

from ShIOEnv.placeholder_types import MAX_PH_EXEC_TRIES, UNLEARNED_PLACEHOLDERS

client = docker.from_env()

placeholder_pattern = re.compile(r'\[[a-zA-Z0-9]+]')  # Compile the regex once
LOCAL_RANDOM_PLACEHOLDERS = ["[File]",  "[Directory]",  "[Executable]"]
RANDOM_PLACEHOLDERS = ["[GlobalFile]", "[GlobalDirectory]", "[GlobalExecutable]", "[Interface]", "[Username]", "[Groupname]"]


def is_placeholder(s: str) -> bool:
    """
    Checks for placeholder pattern [str] substring in string s
    """
    return bool(placeholder_pattern.search(s))


def get_placeholders(s: str) -> list:
    """
    Returns all placeholder pattern [str] substrings in string s
    """
    return placeholder_pattern.findall(s)


def split_placeholders(s: str) -> list:
    """
    Split string s into substrings delimited by placeholder pattern and spaces.

    Example:
    "as[p1]df[p2] fdf[p3]pop" -> ["as", "[p1]", "df", "[p2]", " ", "fdf", "[p3]", "pop"]
    """

    # captures placeholders (group 1) or whitespace (group 2).
    pattern = re.compile(r'(\[[a-zA-Z0-9]{1,}\])|(\s+)')
    parts = pattern.split(s)
    # Filter out empty strings that may occur due to splitting
    return [p for p in parts if p]


def count_tr_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def print_model_parameters(model: torch.nn.Module) -> None:
    # Total parameters
    total_params = count_all_parameters(model)
    # Trainable parameters
    trainable_params = count_tr_parameters(model)
    # Non-trainable parameters
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


def human_format(number) -> str:
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%d%s' % (number // k**magnitude, units[magnitude])


def normalize_tensor(arr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardise tensor to zero‑mean, unit-var (no divide-by-zero)."""
    return (arr - arr.mean()) / (arr.std(unbiased=False) + eps)


def normalize_list(values: List[float]) -> List[float]:
    """Numpy implementation for quick normalisation of python lists."""
    values = np.asarray(values, dtype=np.float32)
    std = values.std()
    if std == 0.0:
        return [0.0] * len(values)
    return ((values - values.mean()) / std).tolist()


def stable_unique(seq) -> List[str]:
    """Return a list with duplicates removed, preserving first‑occurrence order."""
    return list(OrderedDict.fromkeys(seq))


def get_all_expansion_ids(base_input: str, util_map: dict) -> List[str]:
    """
    Gets all nonterminals that can be expanded from base_input to single list
    """
    expansion_ids = [base_input]
    expansion_stack = [base_input]
    while len(expansion_stack) > 0:
        nh = expansion_stack.pop(0)
        for action in util_map[nh]:
            for _ in split_placeholders(action):
                # if placeholder, not been expanded yet, and has defined expansions (non-continuous/random)
                if is_placeholder(_) and _ not in expansion_ids and _ in util_map.keys():
                    expansion_ids.append(_)
                    expansion_stack.append(_)
    return expansion_ids


def flatten_expansion_maps(nonterminals: List[str], util_map: dict) -> List[str]:
    """
    Flattens util map defining each nonterminal's expansion into a single action list for sampling post-masking
    """
    all_productions = []
    for nonterminal in nonterminals:
        if nonterminal in util_map.keys():
            for production in util_map[nonterminal]:
                if production not in all_productions:
                    all_productions.append(production)
    return all_productions


def get_expansion_heads(util_map: dict, starting_production: str) -> List[str]:
    new_head_names = []
    new_head_stack = [starting_production]

    while len(new_head_stack) > 0:
        nh = new_head_stack.pop()
        nh_split = split_placeholders(nh)
        for n in nh_split:
            if is_placeholder(n) and n not in new_head_names and not n in UNLEARNED_PLACEHOLDERS:
                new_head_names.append(n)
                if n in util_map.keys():
                    new_head_stack.append(n)
                    new_head_stack.extend(util_map[n])

    return new_head_names


def populate_d_mask(base_input: str, util_map: dict,
                    discrete_actions: List[str],
                    repeating_productions: List[str],) -> Dict[str, torch.Tensor]:
    """
    Populate d_outs for global action (cmd) and local actions (cmdOptions) and local local actions (PHs), all cmds have uique action heads
    """

    d_masks = {}
    new_head_names = get_expansion_heads(util_map, base_input)

    action_space = len(discrete_actions)
    action_space += 1  # +1 for terminal action for repeating productions

    for _ in new_head_names:  # init action masks for each nonterminal
        d_masks[_] = torch.Tensor([0] * action_space)

    term_action_index = len(discrete_actions)

    new_head_names = stable_unique(new_head_names)

    for _ in new_head_names:
        if _ in util_map.keys():  # discrete action space, set mask the valid productions
            for __ in util_map[_]:
                d_masks[_][discrete_actions.index(__)] = 1
            if _ in repeating_productions:
                d_masks[_][term_action_index] = 1

    return d_masks


def populate_d_mask_2layer(base_input: str, util_map: dict,
                           discrete_actions: List[str], starting_actions: List[str],
                           repeating_productions: List[str]) -> Dict[str, torch.Tensor]:
    """
    Populate d_masks for global action (cmd) and local actions (cmdOptions) and local local actions (PHs), all cmds have uique action heads

    :param base_input: starting grammar nonterminal token ("[Input]")
    :param util_map: dictionary with all defined nonterminal expansions.
    :param discrete_actions: List of all defined expansions reachable from the base_input token
    :param starting_actions: List of all reserved actions not to be reachable from repeating action space ("[Input]", "[df]", "[ls]", etc.)
    :param policy_type: Dict of 0/1 type for continuous/discrete action mask (last 2 for distribution), empty when passed in, will populate reference
    :param repeating_productions: List of all repeating nonterminal tokens to be added to the global action space
    """

    d_masks = {}
    new_head_names = get_expansion_heads(util_map, base_input)

    action_space = len(discrete_actions)
    action_space += 1  # term action

    for _ in new_head_names:  # init action masks for each nonterminal
        d_masks[_] = torch.Tensor([0] * action_space)

    term_action_index = len(discrete_actions)

    dont_mask_actions = []
    for _ in starting_actions:
        if _ in util_map.keys():
            for __ in util_map[_]:
                d_masks[_][discrete_actions.index(__)] = 1
                dont_mask_actions.append(__)

    for _ in new_head_names:
        if _ not in starting_actions and _ not in util_map.keys():
            for __ in discrete_actions:
                if __ not in dont_mask_actions:  # for special starting actions
                    d_masks[_][discrete_actions.index(__)] = 1
            if _ in repeating_productions:
                d_masks[_][term_action_index] = 1

    return d_masks


def expand_wildcard(image_name: str, wildcard_dir: str, d: int=3, ignore_hidden: bool = True) -> List[str]:
    """ Get all children directories up to depth d """
    hidden_filter = "-not -path \"*/.*\"" if ignore_hidden else ""

    find_dir_cmd = (f"find {wildcard_dir} -mindepth 0 -maxdepth {d} -type d "
                    f"{hidden_filter} "
                    f"-exec readlink -f {'{}'} \\;")
    exec_list = [f"cd /", find_dir_cmd]
    expand_output, expand_code = simple_send_cmds_docker(image_name, exec_list)
    print(expand_output)
    if expand_code == 0:
        return expand_output.strip().split()
    return []


def get_working_dirs(image_name: str, working_dirs: list) -> List[str]:
    """
    -Expand wildcards
    -Check if directory exists in image
    -Add to ret_dir
    """
    ret_dirs = []
    for w in working_dirs:
        if w.endswith("*"):
            wildcard_dirs = expand_wildcard(image_name, w.rstrip('*'))
            ret_dirs.extend(wildcard_dirs)
        else:
            find_dir_cmd = f"test -d {w}"
            exec_list = [f"cd /", find_dir_cmd]
            test_out, test_code = simple_send_cmds_docker(image_name, exec_list)
            if test_code == 0:  # directory exists
                ret_dirs.append(w)
    return ret_dirs


def prep_local_dir_content_dict() -> Dict[str, Dict[str, List[str]]]:
    """ Initializes placeholder dicts for a directory entry """
    return {"[File]": {},
            "[Directory]": {},
            "[Executable]": {},
            "[ZipFile]": {},
            "[TarFile]": {},
            }


def run_placeholder_exec_retry(placeholder, cmd_list, container_name) -> Tuple[str, int]:
    exec_output, exec_code = "", -1
    n_tries = 0
    while exec_code == -1 and n_tries < MAX_PH_EXEC_TRIES:  # container error
        n_tries += 1
        exec_output, exec_code = simple_send_cmds_docker(container_name, cmd_list)
    if exec_code != 0:  # == -1 for container error
        print(exec_output)
        print(exec_code)
        raise RuntimeError(f"Container failed to fetch critical placeholders: {placeholder}")
    return exec_output, exec_code


def get_local_zipfile_ph(cwd: str, rand_map: dict, file_map: dict, container_name: str) -> None:
    """ Populated rand_map/file_map inplace with [ZipFile] placeholders """
    zip_cmd = (
        'found=0; '
        'for f in ./*; do '
        '[ -f "$f" ] || continue; '
        'fname="${f#./}"; '
        'sig_hex=$(head -c 6 "$f" | od -An -t x1 | tr -d " \\n"); '
        'if echo "$sig_hex" | grep -q -i "^504b0304"; then '
        'echo "$fname"; '
        'found=1; '
        'elif echo "$sig_hex" | grep -q -i "^1f8b"; then '
        'echo "$fname"; '
        'found=1; '
        'elif echo "$sig_hex" | grep -q -i "^377abcaf271c"; then '
        'echo "$fname"; '
        'found=1; '
        'fi; '
        'done; '
        'exit 0'
    )
    # zip and gzip file signatures. Treating as same production may cause issues if zip is added
    zip_exec = [f"cd {cwd}", zip_cmd]
    zip_output, zip_code = run_placeholder_exec_retry("[ZipFile]", zip_exec, container_name)

    if zip_code == 0:
        archives = zip_output.strip().split()
        if cwd not in file_map[container_name]:
            file_map[container_name][cwd] = prep_local_dir_content_dict()
        file_map[container_name][cwd]["[ZipFile]"] = archives
        for _ in archives:  # dont let non-text files be picked for file placeholders
            try:
                file_map[container_name][cwd]["[File]"].pop(file_map[container_name][cwd]["[File]"].index(_))
            except KeyError:
                pass
        global_archives = [f"{cwd}/{_}" for _ in archives]
        rand_map[container_name]["[GlobalZipFile]"].extend(global_archives)


def get_local_tarfile_ph(cwd: str, rand_map: dict, file_map: dict, container_name: str) -> None:
    """ Populated rand_map/file_map inplace with [TarFile] placeholders """
    tar_cmd = (
        'found=0; '
        'for f in ./*; do '
        '[ -f "$f" ] || continue; '
        'fname="${f#./}"; '
        'if dd if="$f" bs=1 skip=257 count=5 2>/dev/null | grep -q "ustar"; then '
        'echo "$fname"; '
        'found=1; '
        'fi; '
        'done; '
        'exit 0'
    )
    # tar signature
    tar_exec = [f"cd {cwd}", tar_cmd]
    tar_output, tar_code = run_placeholder_exec_retry("[TarFile]", tar_exec, container_name)

    if tar_code == 0:
        tars = tar_output.strip().split()
        if cwd not in file_map[container_name]:
            file_map[container_name][cwd] = prep_local_dir_content_dict()
        file_map[container_name][cwd]["[TarFile]"] = tars
        for _ in tars:  # dont let non-text files be picked for file placeholders
            try:
                file_map[container_name][cwd]["[File]"].pop(file_map[container_name][cwd]["[File]"].index(_))
            except KeyError:
                pass
        global_tars = [f"{cwd}/{_}" for _ in tars]
        rand_map[container_name]["[GlobalTarFile]"].extend(global_tars)


def get_local_file_ph(cwd: str, rand_map: dict, dir_map: dict, container_name: str) -> None:
    """ Populated rand_map/file_map inplace with [File] placeholders """
    file_cmd = "find . -maxdepth 1 -type f -printf \"%f\n\""
    file_exec = [f"cd {cwd}", file_cmd]
    file_output, file_code = run_placeholder_exec_retry("[File]", file_exec, container_name)

    if file_code == 0:
        files = file_output.strip().split()
        if cwd not in dir_map[container_name].keys():
            dir_map[container_name][cwd] = prep_local_dir_content_dict()
        dir_map[container_name][cwd]["[File]"] = files
        global_files = [f"{cwd}/{_}" for _ in files]
        rand_map[container_name]["[GlobalFile]"].extend(global_files)
        rand_map[container_name]["[Path]"].extend(global_files)


def get_local_dir_ph(cwd: str, rand_map: dict, dir_map: dict, container_name: str) -> None:
    """ Populated rand_map/file_map inplace with [Directory] placeholders """
    dir_cmd = "find . -maxdepth 1 -type d -printf \"%f\n\""
    dir_exec = [f"cd {cwd}", dir_cmd]
    dir_output, dir_code = run_placeholder_exec_retry("[Directory]", dir_exec, container_name)

    if dir_code == 0:
        dirs = dir_output.strip().split()
        dirs.append("..")
        if cwd not in dir_map[container_name].keys():
            dir_map[container_name][cwd] = prep_local_dir_content_dict()
        dir_map[container_name][cwd]["[Directory]"] = dirs
        global_dirs = [f"{cwd}/{_}" for _ in dirs]
        rand_map[container_name]["[GlobalDirectory]"].extend(global_dirs)
        rand_map[container_name]["[Path]"].extend(global_dirs)


def get_local_exec_ph(cwd: str, rand_map: dict, dir_map: dict, container_name: str) -> None:
    """ Populated rand_map/file_map inplace with [Executable] placeholders """
    exec_cmd = "find . -maxdepth 1 -type f -executable -printf \"%f\n\""
    exec_exec = [f"cd {cwd}", exec_cmd]
    exec_output, exec_code = run_placeholder_exec_retry("[Executable]", exec_exec, container_name)

    if exec_code == 0:
        execs = exec_output.strip().split()
        if cwd not in dir_map[container_name].keys():
            dir_map[container_name][cwd] = prep_local_dir_content_dict()
        dir_map[container_name][cwd]["[Executable]"] = execs
        global_execs = [f"{cwd}/{_}" for _ in execs]
        rand_map[container_name]["[GlobalExecutable]"].extend(global_execs)
        rand_map[container_name]["[Path]"].extend(global_execs)

def get_local_ph(cwd: str, rand_map: dict, dir_map: dict, container_name: str) -> None:
    """ Populate dir_map and rand_map with current working directory's contents """
    get_local_dir_ph(cwd, rand_map, dir_map, container_name)
    get_local_file_ph(cwd, rand_map, dir_map, container_name)
    get_local_exec_ph(cwd, rand_map, dir_map, container_name)
    get_local_tarfile_ph(cwd, rand_map, dir_map, container_name)
    get_local_zipfile_ph(cwd, rand_map, dir_map, container_name)


def init_local_placeholders(rand_map: dict, dir_map: dict, container_name: str, working_dirs: list) -> None:
    """ Populate rand map with container artifacts for system-specific productions """
    if container_name not in rand_map.keys():
        rand_map[container_name] = {}
    if container_name not in dir_map.keys():
        dir_map[container_name] = {}
    exec_interface = ["cd /", "ip link show | grep -E \"^[0-9]+:\" | cut -d ':' -f 2 | xargs"]
    exec_user = ["cd /", "cat /etc/passwd | cut -d ':' -f 1"]
    exec_group = ["cd /", "cat /etc/group | cut -d ':' -f 1"]

    interface_output, int_code = run_placeholder_exec_retry("[Interface]", exec_interface, container_name)
    rand_map[container_name]["[Interface]"] = interface_output.split()

    user_output, user_code = run_placeholder_exec_retry("[Username]", exec_user, container_name)
    rand_map[container_name]["[Username]"] = user_output.split()

    group_output, group_code = run_placeholder_exec_retry("[Groupname]", exec_group, container_name)
    rand_map[container_name]["[Groupname]"] = group_output.split()

    rand_map[container_name]["[GlobalDirectory]"] = []
    rand_map[container_name]["[GlobalFile]"] = []
    rand_map[container_name]["[GlobalExecutable]"] = []
    rand_map[container_name]["[Path]"] = []
    rand_map[container_name]["[GlobalTarFile]"] = []
    rand_map[container_name]["[GlobalZipFile]"] = []

    for dir in working_dirs:
        get_local_ph(dir, rand_map=rand_map, dir_map=dir_map, container_name=container_name)

    print("All placeholders populated")


_NUM_RE = re.compile(r'\b\d{4,}\b')  # only 4+ digit numbers
_WS_RE  = re.compile(r'\s+')


def resolve_path(pathspec: str, cwd: str) -> str:
    """
    From cowrie
    """
    cwdpieces: list[str] = []
    # If a path within home directory is specified, convert it to an absolute path
    if pathspec.startswith("~/"):
        path = '/root/' + pathspec[2:]
    else:
        path = pathspec

    pieces = path.rstrip("/").split("/")

    if path[0] == "/" or path[0] in ['~', '~/']:  # added or for if just ~ send to reset like / directory
        cwdpieces = []
    else:
        cwdpieces = [x for x in cwd.split("/") if len(x) and x is not None]

    while 1:
        if not len(pieces):
            break
        piece = pieces.pop(0)
        if piece == "..":
            if len(cwdpieces):
                cwdpieces.pop()
            continue
        if piece in (".", ""):
            continue
        if piece == '~':  # ADDED to clear cwd to just home
            cwdpieces = ['~']
        else:
            cwdpieces.append('{}'.format(piece))

    if len(cwdpieces) > 0:
        return "/{}".format("/".join(cwdpieces)) if cwdpieces[0] != '~' else "/{}".format(
            "/".join(cwdpieces)).lstrip('/')
    else:
        return "/"


def _normalise_output(text: str) -> str:
    """
    Make command output comparable across fresh containers.

    Strip all purely numeric tokens (block counts, inode counts, pids).
    Collapse runs of whitespace, lower‑case everything.
    """
    if not text:
        return ""

    # remove volatile numbers
    text = _NUM_RE.sub("<NUM>", text)

    # canonical white‑space & case
    lines = [_WS_RE.sub(" ", ln.strip()).lower()        # '  123  KB' -> '<NUM> kb'
             for ln in text.splitlines()
             if ln.strip()]                             # keep non‑blank only

    return "\n".join(lines)

def update_cwd(cmd: str, cwd: str):
    """
    Update directory for placeholder grabbing
    """
    iter_list = cmd.split(' ')
    if iter_list[0] == 'cd':
        cwd = resolve_path(iter_list[-1], cwd)

    return cwd


def calc_lev_sim_single(o1: str, o2: str, max_len: int=8192) -> float:
    """ Sim ratio of command output strings """
    if not o1 and not o2:  # 1.0 if both strings are empty
        return 1.0
    if not o1 or not o2:  # 0.0 if only one string is empty
        return 0.0
    return Levenshtein.ratio(o1[:max_len], o2[:max_len])


def get_context_diff(c1: str, c2: str) -> str:
    """ Returns structural diff of strings from context-gathering wrapper command """
    if c1 == c2:
        return ""
    return ''.join(difflib.unified_diff(c1.splitlines(keepends=True), c2.splitlines(keepends=True), n=0))


def is_image(image_name: str) -> bool:
    try:
        image = client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def simple_send_cmds_docker(image_name: str, cmds: list, timeout: int = 10, max_output: int = -1, verbose: bool = True) -> Tuple[str, int]:
    """
    Executes a list of commands inside a Docker container, returns (logs, exit_code).

    If max_output is not -1, stops the container as soon as max_output characters
    have been observed.
    """
    # Silence all but the last command (session sim)
    for i, cmd in enumerate(cmds[:-1]):
        if '>' not in cmd:
            cmds[i] = f"{cmd} > /dev/null 2>&1"

    # Join commands into a single bash line
    cmd_send = "; ".join(cmds)

    # Wrap with timeout, capture and alter return code if timeout occurs
    wrapped_command = (
        f"/usr/bin/timeout {timeout}s /bin/bash -c '{cmd_send}'; "
    )
    final_command = ["/bin/bash", "-c", wrapped_command]

    try:
        container = client.containers.run(
            image=image_name,
            command=final_command,
            hostname="svr01",
            mem_limit="256m",
            nano_cpus=int(1e9),
            network_disabled=True,
            network_mode="none",
            privileged=True,
            auto_remove=False,  # no auto-remove to grab output on nonzero exit code
            detach=True,        # run in background
            tty=False
        )

        logs_output = ""
        if max_output != -1:
            # Stream logs and stop once we've collected max_output characters.
            for log in container.logs(stdout=True, stderr=True, stream=True):
                # Decode the log chunk (if needed) and accumulate it.
                chunk = log.decode("utf-8", errors="replace") if isinstance(log, bytes) else log
                logs_output += chunk

                # Check if we reached (or exceeded) the limit.
                if len(logs_output) >= max_output:
                    # Optionally, truncate to exactly max_output characters.
                    logs_output = logs_output[:max_output]
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
        if verbose:
            print(f"Error creating or running container: {e}")
        exit_code = -1
        logs_output = f"APIError: {e}"

    if exit_code == 137 or exit_code == 124:  # container kill from max_output reached or timeout reached
        if verbose:
            print(f"TIMEOUT OR MAX OUT\ncmd: {';'.join(cmds)}")
        exit_code = 0

    return logs_output, exit_code


def get_env_context(container_name: str, cmds: list, timeout: int = 30, verbose: bool = True) -> Tuple[int, Dict[str, str]]:
    """ Captures environment context after a command is run. Passed as {context key: output} """
    context_cmd = (
        "echo \"<del>\" && "  # in case timeout terminated output
        "pwd && echo \"<del>\" && "
        "ls -la && echo \"<del>\" && "
        "groups && echo \"<del>\" && "
        "env && echo \"<del>\" && "
        "set -o && echo \"<del>\" && "
        "shopt && echo \"<del>\" && "
        "ulimit -a && echo \"<del>\" && "
        "iptables -S"
    )
    context_keys = ["pwd", "ls", "groups", "env", "set", "shopt", "ulimit", "iptables"]

    escaped_context_cmd = context_cmd.replace('"', '\\"')

    # Build user commands with no output.
    user_cmds = '; '.join(f'{cmd} > /dev/null 2>&1' for cmd in cmds)

    # Outer quotes (from docker helper) will be single quotes, double quotes for the trap command.
    full_cmd = (
        f"trap \"{escaped_context_cmd}\" TERM; "  # Trap SIGTERM to output context.
        f"{user_cmds}; "  # Run user commands.
        f"{context_cmd}"  # Output context normally if no timeout.
    )
    context_out, context_code = simple_send_cmds_docker(container_name, [full_cmd], timeout=timeout, verbose=verbose)

    context_values = context_out.split("<del>")  # split outputs according to command ordering
    # dictionary mapping each context key to its corresponding value
    context_dict = dict(zip(context_keys, context_values[1:]))  # 1: for starting tag

    return context_code, context_dict


def _strip_dynamic_env_vars(env_str: str) -> str:
    """
    Remove variables whose value legitimately changes from one command to the next.
    In env variables
    """
    IGNORE_PREFIXES = ("OLDPWD=", "_=")  # add PWD= if you also track it via the 'pwd' key
    return "\n".join(
        line for line in env_str.splitlines()
        if not any([_ in line for _ in IGNORE_PREFIXES])
    )


def flatten_obs(obs: Tuple[Tuple], delim='\n') -> str:
    return delim.join(" ".join(_).strip() for _ in obs).strip()


def decode_sequence(tok_seq: torch.Tensor, vocabulary: dict) -> str:
    """
    Converts a tensor of token indices back to a text sequence using self.vocabulary.
    tok_seq (Tensor): Tensor of token indices, shape (sequence_length,) or (1, sequence_length)

    str: Decoded text sequence.
    """
    # If tok_seq is of shape (1, sequence_length), flatten it
    if tok_seq.dim() == 2 and tok_seq.size(0) == 1:
        tok_seq = tok_seq.squeeze(0)

    # Convert tensor to list of ints
    tok_indices = tok_seq.tolist()

    # Map indices to tokens using self.vocabulary['ids']
    tokens = []
    for idx in tok_indices:
        if idx == 0:
            continue  # Skip padding tokens (assuming 0 is the padding index)
        token = vocabulary['ids'].get(idx, '<unk>')  # Use '<unk>' for unknown indices
        tokens.append(token)

    # Join tokens to form a sequence
    text_sequence = ' '.join(tokens)

    return text_sequence


def tensorize_sequence(seq: List[int], device: torch.device) -> torch.Tensor:
    return torch.Tensor([seq]).to(device).type(torch.LongTensor)


def tokenize_sequence(seq: str, vocabulary: dict, input_size: int = 512) -> List[int]:
    """ Tokenizes the sequence using full or partial matches, falling back to characters. """
    tokens = []
    vocab = vocabulary["values"]

    for word in seq.split():
        if word in vocab:
            tokens.append(vocab[word])
        else:
            i = 0
            # Process the word with a greedy longest-match
            while i < len(word):
                match_found = False
                # Try to find longest substring starting at i.
                for j in range(len(word), i, -1):
                    substr = word[i:j]
                    if substr in vocab:
                        tokens.append(vocab[substr])
                        i = j  # move the index forward to the end of the match
                        match_found = True
                        break
                if not match_found:
                    # No partial match found, so tokenize character by character.
                    tokens.append(vocab.get(word[i], 1))  # 1: UNK token
                    i += 1

    return tokens[:input_size-1]


def squeeze_sequence(seq: Tuple[Tuple[str]], extra_input:str="") -> str:
    """ Converts state represented as tuple of tuples into single string """
    ret_seq = " ; ".join(
        [" ".join(filter(None, sublist)) for sublist in seq if any(sublist)]
    )

    # Remove excessive whitespace from the final output
    ret_seq = " ".join(ret_seq.split()) + extra_input

    return ret_seq


def prep_input(seq: Tuple[Tuple], output_buffer: str, device: torch.device, vocabulary: dict, input_size: int = 512) -> Tuple[torch.Tensor, int]:
    tok_seq = tokenize_sequence(squeeze_sequence(seq=seq, extra_input=output_buffer), vocabulary, input_size=input_size)
    seq_len = len(tok_seq)
    padded_tok_seq = tok_seq + [0] * (input_size - seq_len)

    return tensorize_sequence(padded_tok_seq, device), seq_len


def append_to_ndjson(file_path, data) -> None:
    """
    Append one or more dictionaries ndjson file

    Parameters:
        file_path (str): Path to the NDJSON file.
        data (dict or list of dict): A single dictionary or a list of dictionaries to append.
    """
    # Ensure that data is list of dictionaries
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Data must be a dict or a list of dicts")

    # add each record as a new line
    with open(file_path, 'a', encoding='utf-8') as f:
        for record in records:
            json_record = json.dumps(record)
            f.write(json_record + '\n')
