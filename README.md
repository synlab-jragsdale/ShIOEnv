# ShIOEnv: A CLI Behavior-Capturing Environment Enabling Grammar-Guided Command Synthesis for Dataset Curation

## Overview

ShIOEnv is an environment interface for evaluating synthesized shell commands agnostic of synthesis method. 
ShIOEnv provides execution feedback using Docker containers to observe output, exit code, and effect on the environment.
ShIOEnv is a custom OpenAI Gym environment that models command construction as a sequential argument state space with an argument appending action space

## ShIOEnv

* **Intermediate execution feedback**: Optionally execute partial command sequences to compute redundancy-based intermediate rewards.
* **Multi behavior observation**: Measures output and effect on operating system. Currently measured effects:
  * change in current working directory
  * change in current directory's contents
  * active user's groups
  * environment variables
  * shell options
  * system limits
  * iptables rules
* **Adjustable session limits**: hyperparameters to adjust maximum argument and command length, enabling multi-input analysis.
* **Argument redundancy calculation**: Calculate minimality of candidate command with respect to the final observed behavior and argument-omitted subsequences.

## Agent Features

* To demonstrate `ShIOEnv`, a `ShIOAgent` class is implemented with two synthesis techniques to evaluate convergence and generated dataset quality.
* The agent is provided a `util_map` file to build a grammar from which a nonterminal start token is expanded until only terminal tokens remain.
* The agent employs one of two policies using one of two action abstractions:
  * **Policies**:
    * `PolicyNetwork`: Transformer-based actor-critic network.
    * `PolicyRandom`: Baseline random policy using uniform sampling of provided actions.
  * **Action abstractions**:
    * `grammar_mask=True`: Limit grammar productions to the leftmost nonterminal token.
    * `grammar_mask=False`: Select from any production until a global argument is emitted.
* This agent orchestrates dataset collection(`ShIOAgent.run_dataset_creation()`), rollouts (`ShIOAgent._run_episode_multi_syntax()`), and PPO updates (`ShIOAgent.ppo_update(steps)`).

## Requirements

* Python 3.8+
* Docker daemon running locally

See `requirements.txt` for Python dependencies:

```text
docker==7.1.0
docker-compose==1.29.2
gym==0.26.2
nltk==3.9.1
numpy==1.24.4
python_Levenshtein==0.25.1
sacrebleu==2.5.1
scikit_learn==1.3.2
torch==2.4.1
tqdm==4.67.1
transformers==4.46.3
```

Install via:

```bash
pip install -r requirements.txt
```

## Configuration

All hyperparameters and settings are specified in `policy_config.json`. Key sections:

* **model**: Transformer policy network dimensions (input size, embed dim, hidden dim, layers, heads).
* **env**: RL environment settings (horizons, cache size, noise runs, Docker image, working directories).
* **runner**: Training & logging options (episodes, steps, learning rates, PPO coefficients, grammar masking, log directories).
* **dataset**: Dataset collection parameters (dataset size, store frequency, file paths).

Customize these values to match your experiment setup.

## Usage

1. **Prepare Docker environment**:

    ```bash
    bash docker build -t testubuntu docker/
    ```

2. **Edit configuration**: Adjust `policy_config.json` to your needs.
3. **Run sample runner**:

   ```bash
   python run_exp.py -c path/to/policy_config.json
   ```
4. **Monitor logs**: Training logs and model checkpoints will be saved to `runner.logdir` (default `./policy_runlog`).

## Contributing

Contributions, bug reports, and feature requests are welcome. Please fork the repo and submit a pull request.

## License

See license file.
