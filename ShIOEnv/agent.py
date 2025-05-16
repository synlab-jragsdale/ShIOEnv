import copy
import json
import multiprocessing.pool
import os
import random
import string
import threading
import traceback
from datetime import datetime
from threading import Lock
from typing import List, Tuple

from tqdm import tqdm
import numpy as np
import torch
import gym
from torch import nn

from ShIOEnv.policyMasked import PolicyNetwork
from ShIOEnv.policyRandom import PolicyRandom
from ShIOEnv.shioenv import ShIOEnv
from ShIOEnv.utils import prep_input, is_placeholder, split_placeholders, squeeze_sequence, \
    print_model_parameters, init_local_placeholders, normalize_list, is_image, get_working_dirs, \
    get_local_ph, append_to_ndjson, populate_d_mask, get_all_expansion_ids, flatten_expansion_maps, \
    populate_d_mask_2layer, normalize_tensor, stable_unique, human_format
from ShIOEnv.placeholder_types import *


class ShIOAgent:
    def __init__(self, config: dict) -> None:

        if not all([_ in config.keys() for _ in ["model", "env", "runner", "dataset"]]):
            raise KeyError("Missing at least 1 inner config dicts: ", '["model", "env", "runner", "dataset"]')

        self.config = config
        self.verbose = self.config['runner'].get('verbose', True)
        self.debug = self.config['runner'].get('debug', False)
        self.log = self.config['runner'].get('ep_log', True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        gym.register(id='vm-env', entry_point=ShIOEnv, nondeterministic=True)

        if not os.path.isdir(config["runner"]["logdir"]):
            os.makedirs(config["runner"]["logdir"])

        self.image = config["env"]["image"]

        if not is_image(self.image):
            raise NameError(f"docker image {self.image} not defined")

        self.config['env']['working_dirs'] = get_working_dirs(image_name=self.image, working_dirs=config["env"]["working_dirs"])

        """ CONURRENCY VARIABLES """
        self.n_workers = config["runner"]["n_workers"]

        self.all_append_lock = Lock()  # lock for post-episode logging+caching
        self.barrier = threading.Barrier(self.n_workers) if not self.debug else threading.Barrier(1)  # update barrier to ensure current policy usage

        self.step_buffer = []  # holds (obs, len, act, return, adv, logp, pid)
        self.step_counter = 0
        self.global_step_counter = 0

        self.ep_counter = 0  # per-worker update step counter/collector
        self.tr_counter = 0  # global counter
        self.n_datasets = 1  # global dataset loop counter

        self.n_episodes = self.config['runner'].get('n_episodes', 10000)  # total number of episodes to rollout
        self.n_steps = self.config['runner'].get('n_steps', 1000000)  # total number of steps to rollout

        self.update_every = self.config['runner'].get('update_every', 2048)  # train using x steps
        self.n_epochs = self.config['runner'].get("n_epochs", 3)
        self.b_size = self.config['runner'].get('b_size', 128)

        self.base_lr = config['runner'].get('policy_lr', 1e-4)
        self.r_gamma = self.config['runner'].get('r_gamma', 0.99)

        self.early_stop_repeated = self.config["runner"].get("early_stop_repeated", False)  # whether to early stop random policy's repeated actions

        self.log_cache = []

        """ DATASET VARIABLES """
        self.session_id = 0  # iterate with full sequence write
        self.dataset_write_cache = []
        self.dataset_write_seqs = []
        self.dataset_dir = self.config['dataset'].get('datadir', './dataset')
        self.dataset_file = self.config["dataset"].get("datafile", "shio_data.json")  # dataset file
        self.dataset_seq_file = self.config["dataset"].get("seqsfile", "shio_seqs.json")  # used to ensure uniqueness of commands
        self.dataset_size = self.config["dataset"].get("dataset_size", 5000)
        self.dataset_store_every = self.config["dataset"].get("dataset_store_every", 500)
        self.dataset_every = self.config['dataset'].get('dataset_every', 100000)

        self.dataset_tries = 0  # number of actual generations
        self.max_dataset_tries = self.config['dataset'].get('max_trials', self.dataset_size * 5)

        """ GRAMMAR POP """
        with open(config["runner"]["utility_map"], "r") as f:
            self.util_map = json.load(f)

        self.rand_map = {}  # global placeholders to be selected randomly
        self.dir_map = {}  # directory specific placeholders to be selected randomly

        # populate rand_map (randomly chosen placeholders) and dir_map (randomly chosen cwd-specific placeholders
        init_local_placeholders(rand_map=self.rand_map, dir_map=self.dir_map, container_name=self.image,
                                working_dirs=self.config['env']['working_dirs'])

        self.vocabulary = {"ids": {}, "values": {}}
        self.populate_vocab()

        """ UTIL/ACTION MAP POPULATE + MASKING """
        self.util_map["[Input]"] = self.config["runner"]["test_cmd"][:]  # str: list assignment
        self.util_map["[Command]"] = self.config["runner"]["test_cmd"][:]  # same as input, but inner nested (send as local placeholder)

        # which strs to send to new_global_action (may break randomly with some terminals)
        self.global_cmds = [_.lstrip("[").rstrip("]") for _ in self.config["runner"]["test_cmd"]]
        repeating_options = [f"[{_}Options]" for _ in self.global_cmds if f"[{_}Options]" in self.util_map.keys()]
        """ list of nonterminals that can repeat and need a termininating action. Include all cmdOptions by default """
        with open(config["runner"]["repeat_productions"], "r") as f:  # add known repeating placeholders that need terminating action
            self.repeat_productions = list(json.load(f))
        self.repeat_productions = list(set(self.repeat_productions + repeating_options))

        self.all_discrete_nonterminals = stable_unique(get_all_expansion_ids(base_input="[Input]", util_map=self.util_map))
        self.all_discrete_actions = stable_unique(flatten_expansion_maps(nonterminals=self.all_discrete_nonterminals, util_map=self.util_map))
        self.ACTION_DIM = len(self.all_discrete_actions)
        print(f"Total discrete actions: {self.ACTION_DIM + 1}")
        # if sampled masked action for discrete == ACTION_DIM, terminal action

        self.grammar_mask = self.config['runner'].get('grammar_mask', False)  # whether to mask action space according to NT production rule
        self.random_sample = self.config["runner"].get("random_sample", False)  # whether to use trained policy or select from uniform dist

        # populates d_masks (PH: tensor mapping) using util_map (find placeholder, populate tensor mask with expansions)
        if self.grammar_mask:  # if mask according to production rules
            d_masks = populate_d_mask(base_input="[Input]", util_map=self.util_map,
                                      discrete_actions=self.all_discrete_actions,
                                      repeating_productions=self.repeat_productions)
        else:
            self.starting_actions = ["[Input]"] + self.util_map["[Input]"]  # mask as commands to give some starting point for command, everything else full action space
            d_masks = populate_d_mask_2layer(base_input="[Input]", util_map=self.util_map,
                                             discrete_actions=self.all_discrete_actions,
                                             starting_actions=self.starting_actions,
                                             repeating_productions=self.repeat_productions)

        def linear_decay(step):
            return 1.0 - max(0, float(step) / self.n_steps)

        if self.random_sample:
            self.policy = PolicyRandom(vocab_size=self.get_vocab_size(),
                                              config=config['model'],
                                              production_masks=d_masks,
                                              cmd_key="[Input]",
                                              max_action_dim=len(d_masks["[Input]"])).to(self.device)
            self.optimizer = None
            self.lr_scheduler = None

        else:
            self.policy = PolicyNetwork(vocab_size=self.get_vocab_size(),
                                               config=config['model'],
                                               production_masks=d_masks,
                                               cmd_key="[Input]",
                                               max_action_dim=len(d_masks["[Input]"])).to(self.device)  # FOR TESTING WITH SINGLE COMMAND
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.base_lr)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_decay)

        self.model_dir = config['runner'].get('model_dir', './policymodel')
        if config['runner'].get('try_restore', False):
            self._restore_runtime()

        if self.verbose:
            print_model_parameters(self.policy)
            print(f"Grammar Masked: {self.grammar_mask}")
            print(f"Random Sampling: {self.random_sample}")
            print("Handler initialized")

    def get_vocab_size(self) -> int:
        """ Returns number of unique values in placeholder/option dicts """
        return len(self.vocabulary["ids"].keys())

    def add_token(self, token_id: int, token_value: str) -> None:
        self.vocabulary["ids"][token_id] = token_value
        self.vocabulary["values"][token_value] = token_id

    def populate_vocab(self) -> None:
        """ Build vocab from all unique terminal tokens in grammar random space """
        v_c = 0
        self.add_token(token_id=v_c, token_value="<pad>")
        v_c += 1
        self.add_token(token_id=v_c, token_value="<cls>")
        v_c += 1
        self.add_token(token_id=v_c, token_value="<unk>")
        v_c += 1
        for c in string.printable:
            self.add_token(token_id=v_c, token_value=c)
            v_c += 1
        for placeholder_list in self.util_map.values(): # iterates through syntax expansions for non-terminal tokens (non-PH regex)
            for pl in placeholder_list:
                parts = split_placeholders(pl)  # splits action string into non-placeholder segments
                pl_split = [part.strip() for part in parts if part.strip() and not is_placeholder(part)]  # gets all non-placeholder parts of action sequence
                for pl_ in pl_split:
                    if pl_ not in self.vocabulary["values"].keys():
                        self.add_token(token_id=v_c, token_value=pl_)
                        v_c += 1
        for cwd in self.config['env']['working_dirs']:
            for cw in cwd.split("/"):
                if cw not in self.vocabulary["values"].keys():
                    self.add_token(token_id=v_c, token_value=cw)
                    v_c += 1

        for img_global_dict in self.rand_map.values():  # dir_map['testubuntu']
            for ph_list in img_global_dict.values():  # dir_map['testubuntu']['[Username]']
                for ph_token in ph_list:
                    self.add_token(token_id=v_c, token_value=ph_token)
                    v_c += 1

        for img_local_dict in self.dir_map.values():  # dir_map['testubuntu']
            for ph_cwd_dict in img_local_dict.values():  # dir_map['testubuntu']['/home']
                for ph_list in ph_cwd_dict.values():  # dir_map['testubuntu']['/home']['[Directory]']
                    for ph_token in ph_list:  # dir_map['testubuntu']['/home']['[Directory]']['Pictures/']
                        self.add_token(token_id=v_c, token_value=ph_token)
                        v_c += 1

    def run_periodic(self) -> None:
        """
        Initialize sample workers. Breaks out for dataset collection if intermed_dataset is True.
        """
        prev_exec = self.config['env']['intermed_exec']
        self.datdir_base = self.config['dataset']['datadir']

        while self.global_step_counter < self.n_steps and self.tr_counter < self.n_episodes:  # global loop
            if self.config['dataset'].get("intermed_dataset", False) and self.tr_counter > self.n_datasets * self.dataset_every:
                self.dataset_dir = self.datdir_base + f"{human_format(self.n_datasets * self.dataset_every)}"
                self.run_dataset_creation()
                self.n_datasets += 1

            print(f"[*] self.tr_counter: {self.tr_counter}")
            if self.debug:
                if self.verbose:
                    print(f"[*] starting 1 train environment processes")
                env_cfg = copy.deepcopy(self.config['env'])
                env_cfg['intermed_exec'] = prev_exec
                env = gym.make('vm-env', config=env_cfg, image_name=self.image, verbose=self.verbose)
                self._run_ep_thread(env, 0)
            else:
                with multiprocessing.pool.ThreadPool(self.n_workers) as pool:  # new pool since NUM_VMS was possibly modified
                    results = []
                    if self.verbose:
                        print(f"[*] starting {self.n_workers} train environment processes")
                    for _ in range(self.n_workers):
                        env_cfg = copy.deepcopy(self.config['env'])
                        env_cfg['intermed_exec'] = prev_exec
                        env = gym.make('vm-env', config=env_cfg, image_name=self.image)
                        results.append(pool.apply_async(self._run_ep_thread, [env, _]))
                    [result.wait() for result in results]

    def _run_ep_thread(self, env: gym.Env, t_id: int = -1) -> None:
        """
        Worker process handling episode rollouts and global buffer management. Waits for first worker to update every update_every
        """
        # will still be some stochasticity from container exec times
        seed = self.config['runner'].get('seed', random.randint(1, 1000))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        while True:
            #  synchronize with all other threads, do the PPO update, then continue.
            if self.step_counter >= self.update_every:
                # Wait at the barrier until *all* threads are done with this batch
                self.barrier.wait()

                # Let thread 0 do the update
                if t_id == 0:
                    if not self.random_sample:  # trained policy
                        if self.verbose:
                            print(f"Updating network with {self.ep_counter} episodes ({self.step_counter} steps) (~{self.step_counter / self.ep_counter:.2f} steps/episode)")
                        self.ppo_update(self.step_buffer)
                        self.step_buffer.clear()
                        self.step_counter = 0

                        if self.config['runner'].get('checkpoint_training', False):
                            self._save_model()
                        if self.verbose:
                            print(f"Networks updated at episode {self.tr_counter + self.ep_counter}")
                    if self.log:  # (Optional) Save logs with each update
                        self._save_log_cache()
                        if self.verbose:
                            print(f"Episodes up to {self.tr_counter + self.ep_counter} saved to logfile")
                    self.tr_counter += self.ep_counter
                    self.ep_counter = 0
                # Wait again so all threads see the updated policy before continuing
                self.barrier.wait()
                # breakout check for intermediate dataset right after all works are synchronized
                if self.global_step_counter >= self.n_steps or self.tr_counter >= self.dataset_every * self.n_datasets:
                    # if steps done or time for dataset, break
                    break

            # Run single episode with the current policy
            try:
                final_obs, done, trunc, states, actions, rewards, exec_seq_rewards, old_log_probs, value_estimates, controlling_policies, all_infos = \
                    self.run_episode_multi_syntax(env)
            except Exception as e:
                if self.verbose:
                    print(traceback.format_exc())
                    print("Throwing out episode and retrying.")
                    if not isinstance(e, RuntimeError) and self.debug:
                        exit()
                continue  # throw out and retry

            G = self.calc_G(rewards, normalise=self.config["runner"].get("normalize_rewards", False))
            advantages = self.calculate_advantages(rewards, value_estimates, lambd=0.95)  # normalize at ppo

            with self.all_append_lock:
                # Accumulate and flatten for ppo batching
                step_records = []
                for i, (obs_tensor, length) in enumerate(states):
                    step_records.append(
                        (obs_tensor, length,
                         actions[i], G[i], advantages[i],
                         old_log_probs[i], controlling_policies[i])
                    )
                if not self.random_sample:  # trained policy, store accumulated actions for updating
                    self.step_buffer.extend(step_records)  # Increment counters for barrier sync

                self.step_counter += len(step_records)
                self.global_step_counter += len(step_records)
                self.ep_counter += 1

                # base worker
                if t_id == 0 and self.verbose:
                    self._print_ep_info(self.tr_counter + self.ep_counter + 1, done, trunc, rewards, final_obs, all_infos[-1])

                # Log finished episode
                if self.log:
                    self._log_episode(i_episode=self.tr_counter + self.ep_counter, t_id=t_id, done=done, trunc=trunc,
                                      actions=actions, rewards=rewards, redund_scores=exec_seq_rewards, G=G,
                                      controlling_policies=controlling_policies, final_obs=final_obs)

    def calc_G(self, rewards: List[float], normalise: bool = False) -> List[float]:
        g = 0.0
        G = []
        for r in reversed(rewards):
            g = r + self.r_gamma * g
            G.insert(0, g)
        return normalize_list(G) if normalise else G

    def calculate_advantages(self, rewards: List[float], value_estimates: List[float], lambd: float = 0.9) -> List[float]:
        """
        GAE
        """
        advantages = []
        gae = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            next_value = value_estimates[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + self.r_gamma * next_value - value_estimates[t]
            gae = delta + self.r_gamma * lambd * gae
            advantages.insert(0, gae)
        return advantages

    def ppo_update(self, episodes: List[Tuple]) -> None:
        """
        Perform PPO update with global advantage normalization (sb3 default)
        """
        dataset_size = len(episodes)
        clip_eps = self.config["runner"].get("clip_eps", 0.2)
        value_coeff = self.config["runner"].get("value_coeff", 0.5)

        entropy_coeff = self.config["runner"].get("entropy_coeff", 0.01)

        # If b_size is too large, treat as batch update
        b_size = dataset_size if self.b_size >= dataset_size else self.b_size

        adv_all = torch.tensor([ep[4] for ep in episodes], device=self.device)
        adv_all = normalize_tensor(adv_all)

        for epoch in range(self.n_epochs):
            # random.shuffle(episodes)
            idxs = list(range(dataset_size))
            random.shuffle(idxs)

            # running statistics for epoch
            kl_running = 0
            clip_running = 0
            ent_running = 0.0
            value_running = 0.0
            samples = 0

            # Go through mini-batches of episodes
            for start in tqdm(range(0, dataset_size, b_size), leave=False, desc=f"[ {epoch} / {self.n_epochs} ]"):
                # batch = episodes[start : start + b_size]
                batch_idxs = idxs[start:start + b_size]
                batch = [episodes[i] for i in batch_idxs]

                # Stack tensors for the whole minibatch
                states = torch.stack([ep[0] for ep in batch]).squeeze(1).to(self.device)  # (B, ...)
                actions = torch.tensor([ep[2] for ep in batch], device=self.device)  # dtype=torch.float
                returns = torch.tensor([ep[3] for ep in batch], device=self.device)
                advantages = adv_all[batch_idxs]
                old_logp = torch.tensor([ep[5] for ep in batch], device=self.device)
                policy_ids = [ep[6] for ep in batch]

                # Forward pass
                logits, values = self.policy(states, head_keys=policy_ids)

                # current policy distributions
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropies = dist.entropy()

                # PPO objective
                ratio = torch.exp(log_probs - old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # combined loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)
                entropy_loss = -entropy_coeff * entropies.mean()
                loss = policy_loss + value_coeff * value_loss + entropy_loss

                # Optimiser step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                with torch.no_grad():  # bookkeeping
                    kl = (old_logp - log_probs).mean()
                    clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()

                    kl_running += kl.item() * len(batch)
                    clip_running += clip_frac.item() * len(batch)
                    samples += len(batch)
                    ent_running += entropies.sum().item()
                    value_running += value_loss.item() * len(batch)

            if self.verbose:
                print(f"[{epoch + 1}/{self.n_epochs}] "
                      f"mean_KL={kl_running / max(1, samples):.4f} | "
                      f"clip%={100 * clip_running / max(1, samples):.1f} | "
                      f"entropy={ent_running / max(1, samples):.3f} | "
                      f"value_loss={value_running / max(1, samples):.3f}")

        self.lr_scheduler.step()

# obs, done, trunc, states, actions, rewards, exec_seq_rewards, old_log_probs, value_estimates, controlling_policies, all_infos
    def run_episode_multi_syntax(self, env):
        """
        Main sampler for episode. Policy builds arguments from grammar expansions and sends to environment
        """
        def _add_to_trajectory(state,
                               iaction,
                               policy_id,
                               log_prob,
                               value_est,
                               ireward):
            states.append(state)
            actions.append(iaction)
            controlling_policies.append(policy_id)
            old_log_probs.append(log_prob)
            value_estimates.append(value_est)
            rewards.append(ireward)

        obs, info = env.reset()
        done, trunc = False, False
        in_cmd = False  # check for if a new command production is nested or global in session
        ARG_START = "<ARG>"
        ARG_END = "</ARG>"
        arg_depth = 0  # counter for when when argument is built

        n_seqs = 0  # counter for trunc reset for next seq if multicommand session

        env_name = env.get_image_name()
        rand_map = copy.deepcopy(self.rand_map)  # create local instance to prevent created ghost files on reset
        dir_map = copy.deepcopy(self.dir_map)

        policy_control_id = "[Input]"  # starting nonterminal
        policy_output_stack = ["[Input]"]
        policy_output_buffer = ""

        states, actions, rewards, exec_seq_rewards, old_log_probs, value_estimates, controlling_policies, all_infos = ([] for _ in range(8))

        while not done:
            try:
                if trunc:  # option adding hit soft limit, pop items before [cmdOptions] and option PH, continue as normal, set done to true after
                    if arg_depth == 0:  # no more items, stack is empty popped, can end
                        _ = ""
                        if len(policy_output_buffer) > 0:
                            action = {"exec_action": 0,
                                      "new_global": 0,
                                      "input_addition": policy_output_buffer,
                                      }
                            obs, reward, done, trunc, info = env.step(action)
                            policy_output_buffer = ""
                            rewards[-1] += reward

                        action = {  # last exec for final reward
                            "exec_action": 1,
                            "new_global": 0,
                            "input_addition": "",
                        }
                        in_cmd = False
                        obs, reward, done, trunc, info = env.step(action)
                        all_infos.append(info)
                        exec_seq_rewards.append(info["redundancy_score"])  # redundancy score
                        rewards[-1] += reward

                        cwd = env.get_curr_cwd()  # updating cwd after exec_action (full cmd in sequence)
                        if cwd not in dir_map[env_name].keys():
                            get_local_ph(cwd, rand_map=rand_map, dir_map=dir_map,
                                              container_name=env_name)

                        if n_seqs >= env.get_max_global():  # final command is done, exit loop
                            done = True
                        else:
                            trunc = False  # reset loop for next iteration
                            policy_control_id = "[Input]"
                            policy_output_stack = ["[Input]"]
                            policy_output_buffer = ""
                            arg_depth = 0
                    else:  # some arguments (e.g. file that need to be added
                        if arg_depth == 2:  # in global view, pop out any repeating productions  # changed from 1 with no uniform stack init
                            while any([__ in self.repeat_productions for __ in policy_output_stack]):  # remove repeating args and pop out
                                policy_output_stack.pop(0)
                        _ = policy_output_stack.pop(0)
                else:
                    _ = policy_output_stack.pop(0)
            except IndexError:  # last item is in buffer, stack is empty, set trunc to clear buffer and exec in next iter
                trunc = True
                continue

            if _ == ARG_START:
                arg_depth += 1

            elif _ == ARG_END:
                arg_depth -= 1
                # Only flush when closed the outermost ARG
                # changed from 1 with no uniform stack init
                # protect against characters expecting special inputs
                if arg_depth <= 2 and policy_output_buffer: # and policy_output_buffer.strip()[-1] not in ['>', '|', '=', '&', '+']:
                    action = {
                        "exec_action": 0,
                        "new_global": 0,
                        "input_addition": policy_output_buffer,
                    }
                    obs, reward, done, trunc, info = env.step(action)
                    policy_output_buffer = ""
                    rewards[-1] += reward

            elif _ in self.global_cmds and policy_control_id == f"[{_}]" and not in_cmd:  # not a nested command
                # Checks if command (and policy control on global level, not nested [Input] call
                action = {"exec_action": 0,
                          "new_global": 1,
                          "input_addition": _,
                          }
                in_cmd = True  # set to false with new command
                n_seqs += 1
                obs, reward, done, trunc, info = env.step(action)
                rewards[-1] += reward
                policy_output_buffer = ""

            # control handoff (if not already given, generate option -> if not term option, add to stack (split)
            elif _ in self.repeat_productions:  # special handling to check for terminating token and add nonterminal
                policy_control_id = _
                tok_seq, seq_len = prep_input(seq=obs,
                                              output_buffer=policy_output_buffer,
                                              device=self.device,
                                              vocabulary=self.vocabulary,
                                              input_size=self.config["model"].get("input_size", 64))
                logits, value = self.policy(inputs=tok_seq.to(self.device), head_keys=[policy_control_id])
                dist = torch.distributions.Categorical(logits=logits)
                r_action = dist.sample()

                if (self.random_sample and self.early_stop_repeated) or self.early_stop_repeated:  # end repeat sampling early for random (can't learn optimal length
                    if not random.randint(0,4):
                        r_action = torch.Tensor([self.ACTION_DIM]).to(self.device)

                if r_action.item() != self.ACTION_DIM:  # not terminal action for repeat production
                    seq_split = split_placeholders(self.all_discrete_actions[r_action.item()])  # split to seperate terminals/nonterminals
                    policy_output_stack.insert(0, _)  # non-terminating action, add option and repeat option gen PH [CmdOptions]
                    policy_output_stack.insert(0, ARG_END)  # depth handlers
                    for __ in reversed(seq_split):  # reverse for FIFO
                        policy_output_stack.insert(0, __)
                    policy_output_stack.insert(0, ARG_START)

                reward = 0.0  # intermed reward seen in env local input_addition and replaces most recent expansion
                logp = dist.log_prob(r_action)
                _add_to_trajectory(state=(tok_seq, seq_len), iaction=r_action.item(),
                                   policy_id=policy_control_id, log_prob=logp.detach(),
                                   value_est=value.item(), ireward=reward)

            elif is_placeholder(_):
                if _ in RANDOM_PLACEHOLDERS:
                    try:
                        new_seq = rand_map[env_name][_][random.randint(0, len(rand_map[env_name][_]) - 1)]
                    except ValueError:  # no valid placeholder in current directory, create random string
                        new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                elif _ in LOCAL_RANDOM_PLACEHOLDERS:
                    # get second to most previous cwd in case current command is cd (will be an invalid key since cd current cmd may not be fully built)
                    cwd = env.get_prev_cwd()
                    try:
                        new_seq = dir_map[env_name][cwd][_][random.randint(0, len(dir_map[env_name][cwd][_]) - 1)]
                    except ValueError:  # no valid placeholder in current directory, create random string
                        new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                elif _ in RANDOM_STR_PLACEHOLDERS:
                    new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                elif _ == "[Character]":
                    new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=1))
                elif _ in RANDOM_NUM_PLACEHOLDERS.keys():
                    new_seq = str(random.randint(RANDOM_NUM_PLACEHOLDERS[_][0], RANDOM_NUM_PLACEHOLDERS[_][1]))
                elif _ not in self.util_map.keys():  # unknown placeholder, random str + debug
                    print(f"[*] Unknown nonterminal detected: {_}")
                    new_seq = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 8)))
                else:  # non-defined utils captured above
                    policy_control_id = _
                    if policy_control_id == "[Input]":  # enforce random command selection to prevent picking "easy" commands
                        new_seq = self.util_map["[Input]"][random.randint(0, len(self.util_map["[Input]"]) - 1)]
                    else:
                        tok_seq, seq_len = prep_input(seq=obs,
                                                      output_buffer=policy_output_buffer,
                                                      device=self.device,
                                                      vocabulary=self.vocabulary,
                                                      input_size=self.config["model"].get("input_size", 64))
                        logits, value = self.policy(inputs=tok_seq.to(self.device), head_keys=[policy_control_id])
                        dist = torch.distributions.Categorical(logits=logits)
                        r_action = dist.sample()

                        new_seq = self.all_discrete_actions[r_action.item()]
                        logp = dist.log_prob(r_action)

                        reward = 0.0  # intermed reward seen in env local input_addition and replaces most recent expansion
                        _add_to_trajectory(state=(tok_seq, seq_len), iaction=r_action.item(),
                                           policy_id=policy_control_id, log_prob=logp.detach(),
                                           value_est=value.item(), ireward=reward)

                seq_split = split_placeholders(new_seq)  # split to seperate terminals/nonterminals
                policy_output_stack.insert(0, ARG_END)  # depth handlers
                for __ in reversed(seq_split):  # reverse for FIFO
                    policy_output_stack.insert(0, __)
                policy_output_stack.insert(0, ARG_START)
            else:
                if len(policy_output_buffer) > 0 or _ != " ":  # don't append space to buffer if buffer is empty
                    policy_output_buffer += _

        return obs, done, trunc, states, actions, rewards, exec_seq_rewards, old_log_probs, value_estimates, controlling_policies, all_infos

    def run_dataset_creation(self) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.dataset_tries = 0  # reset in case of intermed_dataset
        self.session_id = 0
        env_cfg = copy.deepcopy(self.config['env'])  # turn off intermediate execution, only need redundancy score
        env_cfg['intermed_exec'] = False
        try:
            with open(os.path.join(self.dataset_dir, "shioppo_seqs.ndjson"), 'r', encoding='utf-8') as f:
                self.dataset_write_seqs = json.load(f)
                self.session_id = len(self.dataset_write_seqs)
        except FileNotFoundError:
            self.dataset_write_seqs = []
        if self.debug:
            env = gym.make('vm-env', config=env_cfg,
                           image_name=self.image,
                           verbose=self.verbose)
            self._run_dataset_ep_thread(env, 0)
        else:
            with multiprocessing.pool.ThreadPool(self.n_workers) as pool:  # new pool since NUM_VMS was possibly modified
                results = []
                if self.verbose:
                    print(f"[*] starting {self.n_workers} dataset building environment processes")
                for _ in range(self.n_workers):
                    env = gym.make('vm-env', config=env_cfg, image_name=self.image)
                    results.append(pool.apply_async(self._run_dataset_ep_thread, [env, _]))
                [result.wait() for result in results]

    def _run_dataset_ep_thread(self, env: gym.Env, t_id: int = -1) -> None:
        seed = self.config['runner'].get('seed', random.randint(1, 1000))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        while True:
            if len(self.dataset_write_cache) >= self.dataset_store_every:
                # Wait at the barrier until all workers are done
                self.barrier.wait()
                # Let worker 0 do the update
                if t_id == 0:
                    self._save_clear_dataset_cache()
                self.barrier.wait()  # wait for clear
                if len(self.dataset_write_seqs) >= self.dataset_size or self.dataset_tries >= self.max_dataset_tries:
                    # If reached or exceeded the total desired dataset entries or no new seqs are being added, break
                    break
            try:
                final_obs, done, trunc, states, actions, rewards, exec_seq_rewards, old_log_probs, value_estimates, controlling_policies, all_infos = \
                    self.run_episode_multi_syntax(env)

            except Exception as e:
                if self.verbose:
                    print(traceback.format_exc())
                    if not isinstance(e, RuntimeError) and self.debug:
                        exit()
                continue  # throw out and retry

            with self.all_append_lock:
                self._log_dataset_entry(all_infos)
                if t_id == 0 and self.verbose:
                    print(all_infos)
                    print(f"Dataset Size: {len(self.dataset_write_seqs)}")
                    print(f"Dataset Rollouts: {self.dataset_tries}")
                    print(f"rewards: {rewards}")
                    print(f"done: {done}\t\ttrunc: {trunc}")
                    print(f"output_score: {all_infos[-1]['output_redundancy_score']:.5f}\tcontext_score: {all_infos[-1]['context_redundancy_score']:.5f}\texit_score: {all_infos[-1]['exit_score']}")
                    print(f"context key: {all_infos[-1]['context_key']}")
                    print(f"final commands: {squeeze_sequence(final_obs)}")
                    print(f"----------")

    def _print_step_info(self, action: dict, obs: tuple, reward: float, done: bool, trunc: bool, info: dict) -> None:
        print(f"-----\naction: {action}\n\nDone: {done}, trunc: {trunc}\n\nreward: {reward}\n-----\nobs: {obs}\n-----\ninfo: {info}\n-----")

    def _print_ep_info(self, i_ep: int, done: bool, trunc: bool, rewards, final_obs, info: dict) -> None:
        final_input = squeeze_sequence(final_obs)
        raw_G = self.calc_G(rewards, normalise=False)  # only for printing
        print(final_obs)
        print(done)
        print(trunc)
        print(info)
        print(f"episode: {i_ep}")
        print(f"step: {self.global_step_counter}")
        print(f"value: {raw_G[0]:.5f}")
        print(f"rewards: {rewards}")
        print(f"done: {done}\t\ttrunc: {trunc}")
        print(f"output_score: {info['output_redundancy_score']:.5f}\tcontext_score: {info['context_redundancy_score']:.5f}\texit_score: {info['exit_score']}")
        print(f"context key: {info['context_key']}")
        print(f"final commands: {final_input}")
        print(f"----------")

    def _log_episode(self, i_episode: int, t_id: int, done: bool, trunc: bool, actions, rewards, redund_scores, G, controlling_policies, final_obs) -> None:
        final_input = squeeze_sequence(final_obs)
        self.log_cache.append({
            "episode": i_episode,
            "step": self.global_step_counter,
            "timestamp": datetime.today().isoformat(),
            "worker_id": t_id,
            "rewards": rewards,
            "redundancy_score": redund_scores,
            "done": done,
            "trunc": trunc,
            "G": G,
            "control_ids": controlling_policies,
            "n_actions": len(actions),
            "actions": actions,
            "final_input": final_input,
            "final_obs": [x for x in prep_input(seq=final_obs, output_buffer="", device=self.device, vocabulary=self.vocabulary,
                                              input_size=self.config["model"].get("input_size", 64))[0].squeeze().tolist() if x != 0]
        })

    def _save_log_cache(self) -> None:
        with open(os.path.join(self.config["runner"]["logdir"], self.config["runner"]["logfile"]), 'w') as f:
            json.dump(self.log_cache, f, indent=4)

    def _log_dataset_entry(self, infos: List[dict]) -> None:
        """ Add dataset info to running dataset if input not already added """
        self.dataset_tries += 1  # always increment to break out in case no new seqs before max size
        _write_seq = False  # if new seq
        for info in infos:
            if info["input"] not in self.dataset_write_seqs:  # don't add repeated sequences
                info["session_id"] = self.session_id
                self.dataset_write_cache.append(info)
                self.dataset_write_seqs.append(info["input"])
                _write_seq = True
        if _write_seq:
            self.session_id += 1

    def _save_clear_dataset_cache(self) -> None:
        """ Save dataset and clear intermediate samples. Maintain running list of commands for presence checking """
        append_to_ndjson(str(os.path.join(self.dataset_dir, self.dataset_file)), self.dataset_write_cache)
        with open(os.path.join(self.dataset_dir, self.dataset_seq_file), 'w') as f:
            json.dump(self.dataset_write_seqs, f, indent=4)
        self.dataset_write_cache = []

    def _restore_runtime(self) -> None:
        """ Restore saved model and running counters for lr decay """
        self._load_model()
        self._load_counters()

    def _load_counters(self) -> None:
        try:
            with open(os.path.join(self.config["runner"]["logdir"], self.config["runner"]["logfile"]), 'r') as f:
                self.log_cache = json.load(f)
                self.tr_counter = self.log_cache[-1]["episode"]
                self.step_counter = self.log_cache[-1]["step"]
        except FileNotFoundError:
            self.log_cache = []
            self.tr_counter = 0
            self.step_counter = 0
        if self.config["dataset"].get("intermed_dataset", False):  # if logging dataset
            self.n_datasets = (self.tr_counter // self.config["dataset"].get("dataset_every", 10000)) + 1


    def _load_model(self) -> None:
        # Load model from defined path
        model_path = os.path.join(self.model_dir, "policy.pth")
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path))
            print(f"Loaded policy network from {model_path}.")
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"{os.path.join(self.model_dir, 'config.json')} not found.")

    def _save_model(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        # Save the policy model
        model_path = os.path.join(self.model_dir, "policy.pth")
        torch.save(self.policy.state_dict(), model_path)
        with open(os.path.join(self.model_dir, "config.json"), 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Saved policy network to {model_path}.")
