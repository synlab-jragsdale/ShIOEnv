import copy
import json
import shlex
from collections import OrderedDict, defaultdict
from itertools import product
from functools import lru_cache
from typing import Tuple, List, Dict, Sequence, Callable, Mapping
import re

import docker
import Levenshtein

from ShIOEnv.base_executor import BaseExecutor
from ShIOEnv.placeholder_types import MAX_PH_EXEC_TRIES, UNLEARNED_PLACEHOLDERS, RANDOM_STR_PLACEHOLDERS

client = docker.from_env()

placeholder_pattern = re.compile(r'\[[a-zA-Z0-9]+]')  # Compile the regex once
LOCAL_RANDOM_PLACEHOLDERS = ["[File]",  "[Directory]",  "[Executable]"]
RANDOM_PLACEHOLDERS = ["[GlobalFile]", "[GlobalDirectory]", "[GlobalExecutable]", "[Interface]", "[Username]", "[Groupname]"]


def is_placeholder(s: str) -> bool:
    """
    Checks for placeholder pattern [str] substring in string s
    """
    return bool(placeholder_pattern.search(s))


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


def stable_unique(seq) -> List[str]:
    """Return a list with duplicates removed, preserving first‑occurrence order."""
    return list(OrderedDict.fromkeys(seq))


def build_local_action_spaces_for_cmd(cmd_root: str, util_map: dict) -> Dict[str, List[str]]:
    """
    Returns: head -> compact list of productions (strings) for this command’s reachable heads.
    """
    local: Dict[str, List[str]] = {}
    heads = get_expansion_heads(util_map, cmd_root)
    for h in heads:
        if h in util_map:
            prods = list(util_map[h])
            local[h] = prods
    return local


def is_repeat_terminal(token: str) -> bool:
    return token.startswith("__[") and token.endswith("]__TERM__")


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


def expand_wildcard(executor: BaseExecutor, wildcard_pattern: str, ignore_hidden: bool = True) -> List[str]:
    """Expand trailing wildcard directory patterns.

    Supported forms
    ---------------
    /dir/*
        Expands to every directory directly under /dir.
    /dir/*/*
        Expands to every directory two levels beneath /dir.

    Any additional trailing /* groups are ignored so that the
    expansion depth never exceeds two.

    executor
        Object that can run shell commands inside the target image
        (must expose run_cmd).
    wildcard_pattern
        A path that ends with one or two * components separated by
        /.  If the path contains no *, the path is returned
        unchanged.
    ignore_hidden
        When True (default) directories whose full path contains a
        hidden component (``/.``) are filtered‑out.

    Returns
    -------
    list[str]
        Absolute, readlink‑resolved directory paths that matched the
        supplied pattern.  An empty list is returned when the expansion
        fails or the pattern matches nothing.
    """

    # Normalise the pattern so it never ends with ``/``.
    pattern = wildcard_pattern.rstrip("/")

    # If there is no wildcard, nothing to expand.
    if "*" not in pattern:
        return [pattern]

    # Count trailing "/*" groups (max 2 as per the requirement).
    wild_cnt = 0
    while pattern.endswith("*") and wild_cnt < 2:
        wild_cnt += 1
        pattern = pattern.rsplit("/", 1)[0]  # strip the last "/*"

    base_dir = pattern or "/"  # Ensure the root path is at least "/".

    # Depth == number of trailing wildcards that we consumed
    depth = wild_cnt

    hidden_filter = "-not -path '*/.*'" if ignore_hidden else ""

    find_cmd = (
        f"find {shlex.quote(base_dir)} -mindepth {depth} -maxdepth {depth} "
        f"-type d {hidden_filter} -exec readlink -f '{{}}' \\;"
    )

    exec_list = ["cd /", find_cmd]
    expand_output, expand_code = executor.run_cmd(exec_list, max_output_len=0, timeout=180)

    return expand_output.strip().split() if expand_code == 0 else []


def get_working_dirs(
    executor: BaseExecutor,
    starting_dirs: list,
    *,
    include_intermediate: bool = True,
) -> List[str]:
    """Expand wildcard expressions, optionally include parent levels, and
    validate that directories exist.

    Steps
    -----
    1. Expand any trailing wildcard patterns (via expand_wildcard).
    2. Optionally add the first‑level expansion when the pattern
       contains two trailing wildcards (/dir/*/*) and include_intermediate is True.
    3. Verify that a non‑wildcard path is a directory inside the image (test -d).
    4. Deduplicate the resulting list while preserving order.

    Parameters
    ----------
    executor : BaseExecutor
    starting_dirs : list[str] Paths that may contain trailing * or */* components.
    include_intermediate : bool, default True
        When True and the pattern contains two trailing wildcards, the first‑level expansion
        (e.g. /dir/foo from /dir/*/*) is also included in the result set.
    """

    ret_dirs: List[str] = []

    for pattern in starting_dirs:
        if "*" in pattern:
            # Expand deepest level first.
            ret_dirs.extend(expand_wildcard(executor, pattern))

            if include_intermediate:
                # Count trailing '*' segments.
                parts = pattern.rstrip("/").split("/")
                star_count = 0
                for seg in reversed(parts):
                    if seg == "*":
                        star_count += 1
                    else:
                        break

                # Only add one intermediate level (from /*/* patterns).
                if star_count >= 2:
                    parent_pattern_parts = parts[:-1]  # drop the last '*'
                    parent_pattern = "/".join(parent_pattern_parts)
                    # Ensure it ends with a single '*'.
                    if not parent_pattern.endswith("*"):
                        parent_pattern += "/*"
                    ret_dirs.extend(expand_wildcard(executor, parent_pattern))
        else:
            exec_list = ["cd /", f"test -d {pattern}"]
            _, code = executor.run_cmd(exec_list, max_output_len=0, timeout=180)
            if code == 0:
                ret_dirs.append(pattern)

    # Remove duplicates while preserving original order.
    seen: set[str] = set()
    return [d for d in ret_dirs if not (d in seen or seen.add(d))]

def prep_local_dir_content_dict() -> Dict[str, Dict[str, List[str]]]:
    """ Initializes placeholder dicts for a directory entry """
    return {"[File]": {},
            "[Directory]": {},
            "[Executable]": {},
            "[ZipFile]": {},
            "[TarFile]": {},
            }


def run_placeholder_exec_retry(placeholder: str, cmd_list: List[str], executor: BaseExecutor) -> Tuple[str, int]:
    exec_output, exec_code = "", -1
    n_tries = 0
    while exec_code == -1 and n_tries < MAX_PH_EXEC_TRIES:  # container error
        n_tries += 1
        exec_output, exec_code = executor.run_cmd(cmd_list, max_output_len=0, timeout=180)
    if exec_code != 0:  # == -1 for container error
        print(exec_output)
        print(exec_code)
        raise RuntimeError(f"Container failed to fetch critical placeholders: {placeholder}")
    return exec_output, exec_code


def get_local_zipfile_ph(cwd: str, rand_map: dict, file_map: dict, executor: BaseExecutor) -> None:
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
    zip_output, zip_code = run_placeholder_exec_retry("[ZipFile]", zip_exec, executor)

    if zip_code == 0:
        archives = zip_output.strip().split()
        if cwd not in file_map[executor.base_id]:
            file_map[executor.base_id][cwd] = prep_local_dir_content_dict()
        file_map[executor.base_id][cwd]["[ZipFile]"] = archives
        for _ in archives:  # dont let non-text files be picked for file placeholders
            try:
                file_map[executor.base_id][cwd]["[File]"].pop(file_map[executor.base_id][cwd]["[File]"].index(_))
            except KeyError:
                pass
        global_archives = [f"{cwd}/{_}" for _ in archives]
        rand_map[executor.base_id]["[GlobalZipFile]"].extend(global_archives)


def get_local_tarfile_ph(cwd: str, rand_map: dict, file_map: dict, executor: BaseExecutor) -> None:
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
    tar_output, tar_code = run_placeholder_exec_retry("[TarFile]", tar_exec, executor)

    if tar_code == 0:
        tars = tar_output.strip().split()
        if cwd not in file_map[executor.base_id]:
            file_map[executor.base_id][cwd] = prep_local_dir_content_dict()
        file_map[executor.base_id][cwd]["[TarFile]"] = tars
        for _ in tars:  # dont let non-text files be picked for file placeholders
            try:
                file_map[executor.base_id][cwd]["[File]"].pop(file_map[executor.base_id][cwd]["[File]"].index(_))
            except KeyError:
                pass
        global_tars = [f"{cwd}/{_}" for _ in tars]
        rand_map[executor.base_id]["[GlobalTarFile]"].extend(global_tars)


def get_local_file_ph(cwd: str, rand_map: dict, dir_map: dict, executor: BaseExecutor) -> None:
    """ Populated rand_map/file_map inplace with [File] placeholders """
    file_cmd = "find . -maxdepth 1 -type f -printf \"%f\n\""
    file_exec = [f"cd {cwd}", file_cmd]
    file_output, file_code = run_placeholder_exec_retry("[File]", file_exec, executor)

    if file_code == 0:
        files = file_output.strip().split()
        if cwd not in dir_map[executor.base_id].keys():
            dir_map[executor.base_id][cwd] = prep_local_dir_content_dict()
        dir_map[executor.base_id][cwd]["[File]"] = files
        global_files = [f"{cwd}/{_}" for _ in files]
        rand_map[executor.base_id]["[GlobalFile]"].extend(global_files)
        rand_map[executor.base_id]["[Path]"].extend(global_files)


def get_local_dir_ph(cwd: str, rand_map: dict, dir_map: dict, executor: BaseExecutor) -> None:
    """ Populated rand_map/file_map inplace with [Directory] placeholders """
    dir_cmd = "find . -maxdepth 1 -type d -printf \"%f\n\""
    dir_exec = [f"cd {cwd}", dir_cmd]
    dir_output, dir_code = run_placeholder_exec_retry("[Directory]", dir_exec, executor)

    if dir_code == 0:
        dirs = dir_output.strip().split()
        dirs.append("..")
        if cwd not in dir_map[executor.base_id].keys():
            dir_map[executor.base_id][cwd] = prep_local_dir_content_dict()
        dir_map[executor.base_id][cwd]["[Directory]"] = dirs
        global_dirs = [f"{cwd}/{_}" for _ in dirs]
        rand_map[executor.base_id]["[GlobalDirectory]"].extend(global_dirs)
        rand_map[executor.base_id]["[Path]"].extend(global_dirs)


def get_local_exec_ph(cwd: str, rand_map: dict, dir_map: dict, executor: BaseExecutor) -> None:
    """ Populated rand_map/file_map inplace with [Executable] placeholders """
    exec_cmd = "find . -maxdepth 1 -type f -executable -printf \"%f\n\""
    exec_exec = [f"cd {cwd}", exec_cmd]
    exec_output, exec_code = run_placeholder_exec_retry("[Executable]", exec_exec, executor)

    if exec_code == 0:
        execs = exec_output.strip().split()
        if cwd not in dir_map[executor.base_id].keys():
            dir_map[executor.base_id][cwd] = prep_local_dir_content_dict()
        dir_map[executor.base_id][cwd]["[Executable]"] = execs
        global_execs = [f"{cwd}/{_}" for _ in execs]
        rand_map[executor.base_id]["[GlobalExecutable]"].extend(global_execs)
        rand_map[executor.base_id]["[Path]"].extend(global_execs)

def get_local_ph(cwd: str, rand_map: dict, dir_map: dict, executor: BaseExecutor) -> None:
    """ Populate dir_map and rand_map with current working directory's contents """
    get_local_dir_ph(cwd, rand_map, dir_map, executor)
    get_local_file_ph(cwd, rand_map, dir_map, executor)
    get_local_exec_ph(cwd, rand_map, dir_map, executor)
    get_local_tarfile_ph(cwd, rand_map, dir_map, executor)
    get_local_zipfile_ph(cwd, rand_map, dir_map, executor)


_PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")
ChoiceSpec = Sequence[str] | str | Callable[[], Sequence[str] | str]


def make_expander(mapping: Mapping[str, ChoiceSpec]):
    """
    Returns a cached expand(template) closure whose cache is valid as
    mapping is captured and fixed for the lifetime of the closure.
    """

    def _choices(ph: str) -> list[str]:
        if ph not in mapping:
            return [ph]  # unknown placeholder: leave it unchanged

        spec = mapping[ph]
        if callable(spec):
            spec = spec()

        if isinstance(spec, str):
            return [spec]

        # If the placeholder is explicitly mapped to an empty sequence, interpret it
        # as deletion of the placeholder.
        return list(spec) if spec else [""]

    @lru_cache(maxsize=None)
    def expand(template: str) -> list[str]:
        phs = _PLACEHOLDER_RE.findall(template)
        if not phs:
            return [template]

        # Preserve first occurrence order, but deduplicate to ensure consistency.
        uniq_phs = list(dict.fromkeys(phs))
        choice_lists = [_choices(ph) for ph in uniq_phs]

        out: list[str] = []
        for combo in product(*choice_lists):
            subst = dict(zip(uniq_phs, combo))

            def repl(m: re.Match[str]) -> str:
                tok = m.group(0)
                return subst.get(tok, tok)

            out.append(_PLACEHOLDER_RE.sub(repl, template))
        return out

    return expand


def expand_context(pre_exec_context_base: dict, mapping: Mapping[str, ChoiceSpec]) -> dict:
    expand = make_expander(mapping)

    out: dict = {}
    for cmd_type, info in pre_exec_context_base.items():
        # Preserve all non-command metadata (repeatable, root, etc).
        meta = {k: v for k, v in info.items() if k != "command"}

        cmd_acc: defaultdict[str, list[str]] = defaultdict(list)
        for orig_cmd, val_list in info["command"].items():
            exp_cmds = expand(orig_cmd)

            # Expand each value template; empty val_list stays empty.
            exp_vals = [ev for v in val_list for ev in expand(v)]

            # Accumulate to avoid overwriting if multiple expansions collide.
            for ec in exp_cmds:
                cmd_acc[ec].extend(exp_vals)

        meta["command"] = dict(cmd_acc)
        out[cmd_type] = meta

    return out


def init_local_placeholders(rand_map: dict, dir_map: dict, pre_exec_context: dict, pre_exec_context_base: dict,
                            executor: BaseExecutor, working_dirs: list) -> None:
    """ Populate rand map with container artifacts for system-specific productions """
    if executor.base_id not in rand_map.keys():
        rand_map[executor.base_id] = {}
    if executor.base_id not in dir_map.keys():
        dir_map[executor.base_id] = {}
    if executor.base_id not in pre_exec_context.keys():
        pre_exec_context[executor.base_id] = {}

    exec_interface = ["cd /", "ip link show | grep -E \"^[0-9]+:\" | cut -d ':' -f 2 | xargs"]
    # exec_user = ["cd /", "cat /etc/passwd | cut -d ':' -f 1"]  # all users
    exec_user = ["cat /etc/passwd | grep -E \"(/usr)?/bin/(ba)?sh$\" | cut -d ':' -f 1"]  # only login-able users
    exec_group = ["cd /", "cat /etc/group | cut -d ':' -f 1"]

    interface_output, int_code = run_placeholder_exec_retry("[Interface]", exec_interface, executor)
    rand_map[executor.base_id]["[Interface]"] = interface_output.split()

    user_output, user_code = run_placeholder_exec_retry("[Username]", exec_user, executor)
    rand_map[executor.base_id]["[Username]"] = user_output.split()

    group_output, group_code = run_placeholder_exec_retry("[Groupname]", exec_group, executor)
    rand_map[executor.base_id]["[Groupname]"] = group_output.split()

    rand_map[executor.base_id]["[GlobalDirectory]"] = []
    rand_map[executor.base_id]["[GlobalFile]"] = []
    rand_map[executor.base_id]["[GlobalExecutable]"] = []
    rand_map[executor.base_id]["[Path]"] = []
    rand_map[executor.base_id]["[GlobalTarFile]"] = []
    rand_map[executor.base_id]["[GlobalZipFile]"] = []

    for dir in working_dirs:
        get_local_ph(dir, rand_map=rand_map, dir_map=dir_map, executor=executor)

    mapping = {**rand_map[executor.base_id], **RANDOM_STR_PLACEHOLDERS}
    # rebuild entire dict in one comprehension
    pre_exec_context[executor.base_id] = expand_context(pre_exec_context_base, mapping)

    print(f"[*] {executor.base_id} All placeholders populated (n={sum([len(rand_map[executor.base_id][_]) for _ in rand_map[executor.base_id].keys()])})")


# _NUM_RE = re.compile(r'\b\d{3,}\b')  # only 3+ digit numbers
_NUM_RE = re.compile(r'\b\d+\b')  # all digit numbers
_WS_RE = re.compile(r'\s+')


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


def _normalize_output(text: str) -> str:
    """
    Make command output comparable across fresh containers.

    Strip all purely numeric tokens (block counts, inode counts, pids),
    but annotate each replacement with the number of characters replaced.
    Collapse runs of whitespace, lower-case everything.

    Example: '  12345  KB' -> '<NUM:5> kb'
    """
    if not text:
        return ""

        # Map each distinct numeric string to a stable, small ID (1,2,3,...)
    id_map: dict[str, int] = {}
    next_id = 1

    def _num_repl(m):
        nonlocal next_id
        s = m.group(0)
        i = id_map.get(s)
        if i is None:
            i = next_id
            id_map[s] = i
            next_id += 1
        return f"<NUM:{i}:{len(s)}>"

    # replace volatile numbers with annotated tokens
    text = _NUM_RE.sub(_num_repl, text)

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


def make_exec_cfg(template: dict, **overrides) -> dict:
    """
    Return a new exec-config built from template with the keyword overrides applied.
    """
    cfg = copy.deepcopy(template)
    cfg.update(overrides)
    return cfg


def is_image(image_name: str) -> bool:
    try:
        image = client.images.get(image_name)
        return True
    except docker.errors.APIError:
        return False


def squeeze_sequence(seq: Tuple[Tuple[str]], extra_input:str="") -> str:
    """ Converts state represented as tuple of tuples into single string """
    ret_seq = " ; ".join(
        [" ".join(filter(None, sublist)) for sublist in seq if any(sublist)]
    )

    # Remove excessive whitespace from the final output
    ret_seq = " ".join(ret_seq.split()) + extra_input

    return ret_seq


def ipv4_to_int(ip_address_str: str):
    """
    Converts an IPv4 address string to its integer representation.

    Args:
        ip_address_str (str): The IPv4 address in dotted-decimal format (e.g., "192.168.1.1").

    Returns:
        int: The integer representation of the IPv4 address.
    """
    octets = [int(octet) for octet in ip_address_str.split('.')]

    # Combine octets into a single 32-bit integer
    ip_integer = (octets[0] << 24) | \
                 (octets[1] << 16) | \
                 (octets[2] << 8) | \
                 octets[3]
    return ip_integer


def int_to_ipv4(ip_integer: int):
    """
    Converts a 32-bit integer to its IPv4 dotted-decimal string.

    Args:
        ip_integer (int): Integer in the range 0..4294967295 (0x00000000..0xFFFFFFFF).

    Returns:
        str: The IPv4 address in dotted-decimal format (e.g., "192.168.1.1").
    """
    if not isinstance(ip_integer, int):
        raise TypeError("ip_integer must be an int")
    if ip_integer < 0 or ip_integer > 0xFFFFFFFF:
        raise ValueError("ip_integer must be in the range 0..4294967295")

    return ".".join(str((ip_integer >> shift) & 0xFF) for shift in (24, 16, 8, 0))


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
