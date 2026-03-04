# ShIOEnv: A Command Evaluation Environment for Grammar-Constrained Synthesis and Execution Behavior Modeling

## Overview

ShIOEnv is a behavior-capturing evaluation environment for synthesized shell commands.
It is implemented as a `gymnasium.Env` object that models command synthesis as a sequential argument appending process terminated by an explicit execution action.
The environment executes candidates inside an isolated backend, returning structured feedback that summarizes both 
observable execution artifacts and changes to the system state.

The primary use case is dataset curation for command synthesis and analysis, including the estimation of argument irreducibility relative to observed execution behavior.

We include sample data files for each method in the event code is not able to be run.

A dataset of synthesized samples is available for download on
[HuggingFace](https://huggingface.co/datasets/jragsdale1/ShIO-bash-26.1).

## Key Capabilities

### Behavior capture
ShIOEnv records command output and exit status and extracts a normalized snapshot of operating system state. The current implementation tracks, among other signals:
- Local filesystem changes
- active user groups
- environment variables
- shell options
- system limits
- firewall rules via `iptables`

### Irreducibility and redundancy scoring
Given a completed command sequence, ShIOEnv can estimate argument-level redundancy by comparing the final observed behavior against behavior induced by argument-omitted subsequences. The implementation supports repeated executions to estimate behavioral noise and to compute conservative thresholds.

### Execution backends
ShIOEnv supports two interchangeable execution backends:
- **Docker** via the local Docker daemon
- **Firecracker MicroVMs** via a locally installed Firecracker binary, kernel images, and root filesystems

Backend selection is controlled by `env.exec_method` in `config.json` and by the `--docker` flag in the provided runners.

## Repository Structure

- `ShIOEnv/`
  - `shioenv.py`: `gymnasium` environment implementation and irreducibility scoring logic
  - `agent.py`: dataset collection driver, concurrency, and grammar-based sampling
  - `docker_executor.py`: Docker-based isolated execution
  - `firecracker_executor.py`: Firecracker-based isolated execution
  - `utils.py`: placeholder expansion, filesystem enumeration, utilities
  - `placeholder_types.py`: placeholder vocabularies and sampling distributions
- `maps/`
  - `util_map.json`: context-free grammar specifying command generation productions
  - `repeat_productions.json`: configuration for repeatable productions
  - `sys_context_maps.json`: executor context presets and exclusions
  - `77utils.cmd`: a utility list used to select start symbols for collection
- `fc_store/`: Contains a sample .ext4 filesystem and kernel for execution.
- `docker/`: Contains baseline Ubuntu image and home directory mirroring the firecracker microVM's contents.
- `config.json`: default configuration for executors, environment limits, runner options, and dataset output

## Requirements

- Python 3.10+
- One supported execution backend:
  - Docker: Docker Engine and a locally running Docker daemon
  - Firecracker: Firecracker binary, kernel images, rootfs images, and appropriate host permissions

Python dependencies are specified in `requirements.txt`

Install:
```bash
pip install -r requirements.txt
````

## Setup

### Docker backend

Build the provided image:

```bash
docker build -t testubuntu -f docker/Dockerfile docker
```

Ensure `config.json` contains the image name in `executor.docker_images`:

```json
  {"docker_images": ["testubuntu"]}
```

### Firecracker backend

Firecracker execution requires external assets for which we provide an Ubuntu24.04 filesystem and modified Linux v6.8 kernel:

* a Firecracker binary path (`executor.fc_path`)
* a directory of kernel images (`executor.fc_kernel_store`)
* a directory of root filesystem images (`executor.fc_rootfs_store`)

These paths and resource parameters must be set in `config.json`. The agent performs a compatibility check across available kernel and rootfs pairs before dataset collection.

Refer to the [Firecracker repository](https://github.com/firecracker-microvm/firecracker) For issues related to the Firecracker implementation.

## Configuration

All hyperparameters and paths are specified in `config.json`.

### `executor`

Controls backend resources and environment discovery.

* `working_dirs`: seed directories used to enumerate candidate working directories inside the executor
* `pre_exec_context_path`: JSON file defining context values for initial execution context (e.g., users, environment variables).
* Docker:

  * `docker_images`: list of allowed base images
* Firecracker:

  * `fc_rootfs_store`, `fc_kernel_store`: stores for rootfs and kernels
  * `fc_vcpus`, `fc_mem_mib`: VM sizing parameters
  * `fc_path`: Firecracker binary path
  * networking fields such as `tap_host` and `fc_tap_dns` if enabled

### `env`

Controls the environment state space and scoring procedure.

* `global_horizon`, `local_horizon`: maximum number of commands and arguments per command
* `horizon_margin`: allowance used when truncation occurs mid-argument sequence
* `timeout`: execution timeout per command
* `cache_size`: LRU cache capacity for execution traces
* `noise_runs`, `noise_sigmaK`: repeated execution parameters for noise-aware scoring
* `contrib_samples`, `get_final_score`: subsequence sampling and scoring controls
* `exec_method`: `docker` or `firecracker`

### `runner`

Controls dataset collection and grammar constraints.

* `n_workers`: parallelism level
* `utility_map`: grammar production file, typically `maps/util_map.json`
* `repeat_productions`: configuration for repeatable expansions
* `grammar_mask`: toggles constrained expansion behavior in the agent
* `random_arg_weight`, `early_stop_repeated`: controls repeat termination heuristics
* `runprefix`: run identifier used in output paths

### `dataset`

Controls output size and persistence.

* `dataset_size`: number of examples to generate
* `dataset_store_every`: write frequency
* output filenames such as `data.ndjson` and `seqs.json`

Dataset output is written under `data/<runprefix>/` by default.

## Usage

### Grammar-driven dataset collection

Run collection with Firecracker (default):

```bash
python run_collection.py -l <run_name> -n 10
```

Run collection with Docker:

```bash
python run_collection.py -l <run_name> -n 10 --docker
```

Common options:

* `-c, --config`: path to `config.json`
* `-u, --util`: path to a utility list such as `maps/77utils.cmd`
* `-m, --max`: local argument horizon
* `-a, --approx`: number of subsequence permutations evaluated for irreducibility
* `-w, --workers`: concurrency level
* `--nomask`: disable grammar masking
* `--no-score`: Collect samples without calculating irreducibility

### Adapting existing input-only dataset for ShIOEnv collection

The `run_adapted_collection.py` runner consumes a JSON file containing pre-specified command structures or templates and executes them under the same backend instrumentation.

**Note**: Irreducibility calculation is unable to be performed for string samples due to lack of argument delimitation. 
```bash
python run_adapted_collection.py -l <run_name> -r path/to/inputs.json --docker
```

## Outputs

A run produces a directory `data/<runprefix>/` containing:

* `data.ndjson`: line-delimited JSON records containing command sequences and captured behaviors
* `seqs.json`: a deduplication set used to prevent repeated command sequences

The exact schema of each record is defined by the agent logging code and may include execution artifacts, normalized context diffs, and irreducibility estimates.

## Security Notes

This repository executes shell commands inside isolated environments. Execute only on hosts where Docker or Firecracker usage is permitted, and treat generated commands and captured artifacts as potentially sensitive. When using Firecracker, ensure the host configuration prevents resource collisions across concurrent VMs.

## Contributing

Bug reports and pull requests are welcome. Please include a minimal reproduction and the relevant `config.json` section when reporting backend or execution issues.

## Citation

If this dataset is used in support of research, please use the following citation:

```bibtex
@misc{ragsdale2025shioenv,
      title={ShIOEnv: A CLI Behavior-Capturing Environment Enabling Grammar-Guided Command Synthesis for Dataset Curation}, 
      author={Jarrod Ragsdale and Rajendra Boppana},
      year={2025},
      eprint={2505.18374},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18374}, 
}
