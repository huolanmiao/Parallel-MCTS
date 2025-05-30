import math
import random
import copy
import time
from multiprocessing import Pool, cpu_count
import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper function for multiprocessing pool.
def _simulate_rollout_task(env_name, task_data,
                           rollout_depth_limit, possible_actions_indices,
                           worker_seed, gamma=0.95):
    """
    Performs a random rollout from a given state for a worker process.
    task_data can be:
        - cloned_env_state (for 'ale' or 'generic_clone')
        - (action_sequence_to_node, initial_obs_for_replay_verification) for 'action_replay'
    """
    env = gym.make(env_name, render_mode=None)
    # Seed the environment for this specific rollout to ensure diverse rollouts if desired,
    # or for deterministic replay if action sequences are used.
    current_obs, info = env.reset(seed=worker_seed)

    is_rollout_start_terminal = False

    action_sequence, _ = task_data
    for action in action_sequence:
        current_obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            is_rollout_start_terminal = True
            break

    total_rollout_reward = 0.0
    if is_rollout_start_terminal: # If state reconstruction led to a terminal state
        env.close()
        return total_rollout_reward # Which is 0.0, or a specific terminal value if known differently

    # Perform the random rollout
    current_discount = 1.0
    for _ in range(rollout_depth_limit):
        if not possible_actions_indices: # Should not happen for non-terminal states with actions
             break
        action = random.choice(possible_actions_indices)
        current_obs, reward, terminated, truncated, _info = env.step(action)
        total_rollout_reward += current_discount * reward
        current_discount *= gamma
        if terminated or truncated:
            break
    
    env.close()
    return total_rollout_reward

class Node:
    def __init__(self, parent=None, action_that_led_here=None,
                 possible_actions_indices=None, ucb_exploration_const=1.41,
                 observation=None,
                 action_sequence_from_root=None,
                 # Properties of this node's state
                 is_terminal_state=False,
                 reward_for_reaching_this_state=0.0):

        self.parent = parent
        self.action_that_led_here = action_that_led_here

        self.children = []
        self.untried_actions = possible_actions_indices.copy() if possible_actions_indices else []
        if self.untried_actions: # Ensure random exploration order
            random.shuffle(self.untried_actions)


        self.wins = 0.0
        self.visits = 0
        self.C = ucb_exploration_const

        # State information (one of these will be primarily used based on cloning_method)
        self.observation = observation
        self.action_sequence_from_root = action_sequence_from_root if action_sequence_from_root is not None else []

        self.is_terminal = is_terminal_state
        # Reward obtained for the action that *led* to this node's state
        self.reward_at_node = reward_for_reaching_this_state

    def is_fully_expanded(self):
        return not self.untried_actions

    def add_child(self, child_node):
        self.children.append(child_node)

    def select_best_child_ucb1(self):
        if not self.children:
            return None
        best_child = max(self.children, key=lambda child:
                         (child.wins / child.visits if child.visits > 0 else float('inf')) +
                         self.C * math.sqrt(math.log(self.visits) / child.visits if child.visits > 0 else float('inf')))
        return best_child


class MCTS_Parallel_Simulations:
    def __init__(self, env_name, num_workers=None, ucb_exploration_const=1.41, rollout_depth=100, num_simulation=1000, fixed_initial_seed=None):        
        self.env_name = env_name
        self.ucb_exploration_const = ucb_exploration_const
        self.num_simulation = num_simulation
        self.rollout_depth = rollout_depth
        self.master_env_for_expansion = None
        
        self.cur_root_node = None # Current root node for MCTS search


        if num_workers is None:
            self.num_workers = cpu_count()
        else:
            self.num_workers = num_workers
            
        if self.num_workers > 1:
            self.pool = Pool(processes=self.num_workers)
        else:
            self.pool = None

        _temp_env = gym.make(self.env_name)
        if not isinstance(_temp_env.action_space, gym.spaces.Discrete):
            _temp_env.close()
            raise ValueError("MCTS currently supports Discrete action spaces only.")
        self.action_space_n = _temp_env.action_space.n
        self.possible_actions_indices = list(range(self.action_space_n))

        
        self.master_env_for_expansion = gym.make(self.env_name)
        
        if fixed_initial_seed is not None:
            self.base_seed = fixed_initial_seed
        else:
            self.base_seed = random.randint(0, 1_000_000) # Base seed for MCTS search consistency if needed

        _temp_env.close()

    def _select_promising_node(self, root_node):
        node = root_node
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node
            else:
                if not node.children: return node
                node = node.select_best_child_ucb1()
                if node is None: return root_node # Should not happen
        return node

    def _expand_node(self, parent_node, seed):
        action_to_try = parent_node.untried_actions.pop()

        child_obs, child_reward, child_term, child_trunc = None, 0.0, False, False
        child_action_sequence = parent_node.action_sequence_from_root + [action_to_try]
        
        
        # Use the persistent master_env_for_expansion
        self.master_env_for_expansion.reset(seed=seed)
        # Replay actions to reach parent_node's state
        for action_idx in parent_node.action_sequence_from_root:
            self.master_env_for_expansion.step(action_idx)
        child_obs, child_reward, child_term, child_trunc, _ = self.master_env_for_expansion.step(action_to_try)


        child_node = Node(parent=parent_node, action_that_led_here=action_to_try,
                          possible_actions_indices=self.possible_actions_indices if not (child_term or child_trunc) else [],
                          ucb_exploration_const=self.ucb_exploration_const,
                          observation=child_obs,
                          action_sequence_from_root=child_action_sequence,
                          is_terminal_state=(child_term or child_trunc),
                          reward_for_reaching_this_state=child_reward)
        parent_node.add_child(child_node)
        return child_node

    def _backpropagate(self, node, reward_from_rollout, gamma=0.95):
        current_node = node
        cur_value_estimation = reward_from_rollout
        while current_node is not None:
            current_node.visits += 1
            # The reward backpropagated should be the future rewards (rollout)
            # The immediate reward for reaching current_node is stored in current_node.reward_at_node
            # Standard MCTS backpropagates the outcome of the playout.
            cur_value_estimation = node.reward_at_node + gamma * cur_value_estimation 
            current_node.wins += cur_value_estimation
            current_node = current_node.parent

    def search(self, current_observation, current_info, is_current_state_terminated, is_current_state_truncated, history_actions, seed, reuse_tree=True): 
        
        if self.cur_root_node is None:
            root_params = {
                'observation': current_observation,
                'action_sequence_from_root': history_actions,
                'is_terminal_state': is_current_state_terminated or is_current_state_truncated,
                'reward_for_reaching_this_state': 0.0, # Root node has no prior action leading to it
                'possible_actions_indices': self.possible_actions_indices if not (is_current_state_terminated or is_current_state_truncated) else [],
                'ucb_exploration_const': self.ucb_exploration_const
            }
            
            root_node = Node(**root_params)
            self.cur_root_node = root_node
        else:            
            root_node = self.cur_root_node
            
        # print(len(root_node.action_sequence_from_root))
        # print(f"state in MCTS search: {root_node.observation}")

        if root_node.is_terminal:
            print("Root node is terminal, no further actions possible.")
            return random.choice(self.possible_actions_indices) if self.possible_actions_indices else 0

        simulations_done = 0
        num_total_simulations_for_this_move = self.num_simulation 

        while simulations_done < num_total_simulations_for_this_move:
            num_to_batch = self.num_workers if self.pool else 1
            actual_batch_size = min(num_to_batch, num_total_simulations_for_this_move - simulations_done)
            

            batch_simulation_tasks_data = [] # Holds (node_for_bp, task_args_for_pool)
            
            for i in range(actual_batch_size):
                promising_node = self._select_promising_node(root_node)
                node_to_evaluate = promising_node

                if not promising_node.is_terminal:
                    node_to_evaluate = self._expand_node(promising_node, seed)
                
                # If after selection/expansion, the node is terminal, backpropagate its intrinsic reward
                if node_to_evaluate.is_terminal:
                    self._backpropagate(node_to_evaluate, node_to_evaluate.reward_at_node) # Or 0 if terminal means end of game with no further reward
                    simulations_done += 1
                else:
                    # Prepare data for parallel simulation task
                    task_data_for_worker = (node_to_evaluate.action_sequence_from_root, root_node.observation)
                    
                    worker_seed = seed

                    task_args = (self.env_name, task_data_for_worker,
                                 self.rollout_depth, self.possible_actions_indices, worker_seed)
                    # print(f"Preparing task for worker {i + 1}/{actual_batch_size} with seed {worker_seed} for action {node_to_evaluate.action_that_led_here} that lead to node {node_to_evaluate.observation} (terminal: {node_to_evaluate.is_terminal})")
                    batch_simulation_tasks_data.append({'node_for_bp': node_to_evaluate, 'task_args': task_args})
            
            if not batch_simulation_tasks_data: # All paths in batch led to terminal or failed expansions
                if simulations_done >= num_total_simulations_for_this_move: break
                else: continue # Try another batch if budget remains

            rollout_rewards = []
            if self.pool:
                pool_tasks = [item['task_args'] for item in batch_simulation_tasks_data]
                rollout_rewards = self.pool.starmap(_simulate_rollout_task, pool_tasks)
            else:
                for item in batch_simulation_tasks_data:
                    rollout_rewards.append(_simulate_rollout_task(*item['task_args']))
            
            for i, item in enumerate(batch_simulation_tasks_data):
                self._backpropagate(item['node_for_bp'], rollout_rewards[i])
            simulations_done += len(batch_simulation_tasks_data)


        best_child = max(root_node.children, key=lambda child: child.visits, default=None)
        if reuse_tree:
            self.cur_root_node = best_child
            self.cur_root_node.parent = None # Reset parent to None for the new search
        else:
            self.cur_root_node = None
        return best_child.action_that_led_here

    def shutdown_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
        if self.master_env_for_expansion:
            self.master_env_for_expansion.close()
            self.master_env_for_expansion = None


def evaluate_mcts_on_env(env_name, num_episodes=10, rollout_depth=10, simulations_per_move=50, 
                         num_workers=None, reuse_tree=True, render_actual_game=False, record_video=False, 
                         experiment_seed=42, video_folder_suffix=""):
    logger.info(f"Evaluating MCTS on {env_name}: {num_episodes} episodes, "
                f"Sim/move: {simulations_per_move}, Rollout Depth: {rollout_depth}, "
                f"Workers: {num_workers if num_workers is not None else cpu_count()}, Seed: {experiment_seed}")
    
    mcts_agent = MCTS_Parallel_Simulations(env_name, 
                                           num_workers=num_workers, 
                                           rollout_depth=rollout_depth,
                                           num_simulation=simulations_per_move,
                                           fixed_initial_seed=experiment_seed) # Pass experiment_seed for MCTS internal base seed

    render_mode_env = 'human' if render_actual_game else ('rgb_array' if record_video else None)
    
    try:
        env = gym.make(env_name, render_mode=render_mode_env)
    except Exception as e:
        logger.warning(f"Could not set render_mode '{render_mode_env}'. Trying without. Error: {e}")
        env = gym.make(env_name)


    if record_video:
        video_folder_name = f"videos/{env_name.replace('/', '_')}{video_folder_suffix}"
        try:
            env = RecordVideo(env, video_folder=video_folder_name, 
                              name_prefix=f"sim{simulations_per_move}_depth{rollout_depth}_seed{experiment_seed}",
                              episode_trigger=lambda x: True) # Records all episodes for this config
        except Exception as e:
            logger.error(f"Failed to initialize RecordVideo: {e}. Video recording will be disabled.")
            # Fallback: ensure env is still wrapped with RecordEpisodeStatistics
            if not isinstance(env, RecordEpisodeStatistics):
                 env = RecordEpisodeStatistics(env, buffer_length=num_episodes)


    if not isinstance(env, RecordEpisodeStatistics): # Ensure it's always wrapped for stats
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)


    total_rewards_all_episodes = []

    # Collect time for each search
    search_times = []
    for episode_num in range(num_episodes):
        current_episode_seed = experiment_seed + episode_num # Consistent seed for each episode across different MCTS params
        obs, info = env.reset(seed=current_episode_seed) 
        
        history_actions = []
        terminated = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        mcts_agent.cur_root_node = None # Reset MCTS tree for each new episode

        search_time_per_episode = []
        while not (terminated or truncated):
            start_time = time.time()
            
            action = mcts_agent.search(obs, info, terminated, truncated, history_actions, seed=current_episode_seed, reuse_tree=reuse_tree)
            
            search_time = time.time() - start_time
            search_time_per_episode.append(search_time)
            
            history_actions.append(action)
            
            obs, reward, terminated, truncated, new_info = env.step(action)
            info = new_info 
            
            episode_reward += reward
            step_count += 1
            
            if render_actual_game: 
                try:
                    env.render()
                except: pass # Some envs might not support render or have issues if not rgb_array mode initially
            
            if terminated or truncated:
                break
        
        # Record the search time for this episode
        # logger.info(f"Episode {episode_num + 1} average search time per move: {np.mean(search_time_per_episode):.4f} seconds (total: {sum(search_time_per_episode):.4f} seconds), std: {np.std(search_time_per_episode):.4f} seconds")
        search_times.extend(search_time_per_episode)
        
        total_rewards_all_episodes.append(episode_reward)
        mcts_agent.cur_root_node = None

    
    
    mcts_agent.shutdown_pool()
    env.close()

    avg_reward = np.mean(total_rewards_all_episodes) if num_episodes > 0 else 0.0
    std_reward = np.std(total_rewards_all_episodes) if num_episodes > 0 else 0.0
    search_time_per_move = np.mean(search_times)
    search_time_std = np.std(search_times)
    
    logger.info(f"Finished evaluation for {env_name} with Sim/move: {simulations_per_move}, Rollout Depth: {rollout_depth}")
    logger.info(f"Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f} (over {num_episodes} episodes)")
    logger.info(f"Individual Rewards: {total_rewards_all_episodes}")
    logger.info(f"Total search time for all episodes: {sum(search_times):.4f} seconds")
    logger.info(f"Average search time per move: {search_time_per_move:.4f} seconds (std: {search_time_std:.4f} seconds)")
    
    return total_rewards_all_episodes, avg_reward, std_reward, search_time_per_move, search_time_std

# --- Plotting Functions ---
def plot_experiment_results(results_list, env_name, fixed_param_name, varying_param_name, fixed_param_values, varying_param_values):
    """
    Generic plotting function.
    Example: fixed_param_name='Rollout Depth', varying_param_name='Simulations per Move'
    """
    plt.figure(figsize=(12, 7))
    
    for fixed_val in fixed_param_values:
        rewards_for_this_fixed_val = []
        rewards_std_for_this_fixed_val = []
        
        actual_varying_param_values_for_plot = []

        for vary_val in varying_param_values:
            found = False
            for res in results_list:
                # Check which parameter is fixed and which is varying
                if fixed_param_name == 'Rollout Depth':
                    if res['rollout_depth'] == fixed_val and res['sim_per_move'] == vary_val:
                        rewards_for_this_fixed_val.append(res['avg_reward'])
                        rewards_std_for_this_fixed_val.append(res['std_reward'])
                        if vary_val not in actual_varying_param_values_for_plot:
                             actual_varying_param_values_for_plot.append(vary_val)
                        found = True
                        break
                elif fixed_param_name == 'Simulations per Move':
                     if res['sim_per_move'] == fixed_val and res['rollout_depth'] == vary_val:
                        rewards_for_this_fixed_val.append(res['avg_reward'])
                        rewards_std_for_this_fixed_val.append(res['std_reward'])
                        if vary_val not in actual_varying_param_values_for_plot:
                             actual_varying_param_values_for_plot.append(vary_val)
                        found = True
                        break
            if not found: # If a specific combination was not run, or data is missing
                pass # Or append NaN, or handle as needed: rewards_for_this_fixed_val.append(np.nan)

        # Ensure x-values are sorted for correct plotting
        sorted_indices = np.argsort(actual_varying_param_values_for_plot)
        plot_x = np.array(actual_varying_param_values_for_plot)[sorted_indices]
        plot_y = np.array(rewards_for_this_fixed_val)[sorted_indices]
        plot_y_err = np.array(rewards_std_for_this_fixed_val)[sorted_indices]

        if len(plot_x) == len(plot_y): # Only plot if data is consistent
            plt.plot(plot_x, plot_y, marker='o', label=f'{fixed_param_name} = {fixed_val}')
            plt.fill_between(plot_x, plot_y - plot_y_err, plot_y + plot_y_err, alpha=0.2)
        else:
            logger.warning(f"Skipping plot line for {fixed_param_name} = {fixed_val} due to data mismatch (x:{len(plot_x)}, y:{len(plot_y)})")


    plt.title(f'Effect of {varying_param_name} on Average Episode Reward ({env_name})')
    plt.xlabel(varying_param_name)
    plt.ylabel('Average Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{env_name.replace('/', '_')}_effect_of_{varying_param_name.replace(' ', '_')}.png")
    # plt.show()


if __name__ == '__main__':
    random.seed(42) 
    # --- Configuration for Experiment ---
    # ATARI_ENV_NAME = 'ALE/Pong-v5'
    ATARI_ENV_NAME = 'ALE/Breakout-v5' 
    CLASSIC_CONTROL_ENV_NAME = 'CartPole-v1'
    # CLASSIC_CONTROL_ENV_NAME = 'LunarLander-v2'

    SELECTED_ENV_NAME = CLASSIC_CONTROL_ENV_NAME 

    # Reduce episodes for quicker experiment; increase for more stable results
    NUM_EPISODES_PER_CONFIG = 5  # Number of episodes per parameter combination
    
    # NUM_MCTS_WORKERS = cpu_count() // 2 if cpu_count() > 1 else 1 # Use half cores or 1 if single core
    NUM_MCTS_WORKERS = 1 # Set to 1 for simpler debugging, can be increased.
                        # Note: if >1, ensure MCTS parallel rollout seeding is robust. (Included a fix for this)
    
    # Reuse MCTS tree across episodes for each config
    REUSE_TREE = True
    
    # Disable rendering and video for batch experiments
    RENDER_GAME_DURING_EXPERIMENT = False 
    RECORD_VIDEO_DURING_EXPERIMENT = False

    MAIN_EXPERIMENT_SEED = 42 # Master seed for the entire experiment for reproducibility

    # --- Start Experiment ---
    logger.info(f"Starting MCTS experiment on {SELECTED_ENV_NAME} with {NUM_MCTS_WORKERS} worker(s).")
    logger.info(f"Master Seed for experiment: {MAIN_EXPERIMENT_SEED}")
    logger.info(f"Reuse MCTS tree across episodes: {REUSE_TREE}\n")

    # --- Parameter Ranges ---
    logger.info("Defining parameter ranges for the experiment...")
    # Define parameter ranges for the experiment
    simulations_per_move_options = [1, 2, 5, 10, 20, 35, 55] # More simulations usually better, but slower
    rollout_depth_options = [5, 10, 15, 20, 25, 30]         # Deeper rollouts might be better for some envs

    # --DEBUG--
    # simulations_per_move_options = [1, 2] # More simulations usually better, but slower
    # rollout_depth_options = [5, 10]         # Deeper rollouts might be better for some envs
    
    # For Atari, these values might need to be much larger. For CartPole, they are reasonable.
    if SELECTED_ENV_NAME.startswith("ALE/"):
        simulations_per_move_options = [50, 100, 200]
        rollout_depth_options = [20, 50, 80]
        NUM_EPISODES_PER_CONFIG = 3 # Atari is slower
        logger.info("Using adjusted parameters for Atari environment.")

    logger.info(f"Simulations per Move Options: {simulations_per_move_options}")
    logger.info(f"Rollout Depth Options: {rollout_depth_options}\n")
    
    # --- Run Experiment ---
    logger.info("Starting the MCTS experiment with the defined parameter ranges...")
    # Prepare to collect results
    experiment_results_list = []
    total_configs = len(simulations_per_move_options) * len(rollout_depth_options)
    current_config = 0
    
    for sim_per_move in simulations_per_move_options:
        for rollout_depth_val in rollout_depth_options:
            current_config += 1
            logger.info(f"--- Running Config {current_config}/{total_configs}: "
                        f"Sim/move={sim_per_move}, RolloutDepth={rollout_depth_val} ---")
            
            # Each configuration runs with the same base seed for fair comparison of episode sequences
            all_rewards, avg_reward, std_reward, search_time_per_move, search_time_std = evaluate_mcts_on_env(
                env_name=SELECTED_ENV_NAME,
                num_episodes=NUM_EPISODES_PER_CONFIG,
                rollout_depth=rollout_depth_val,
                simulations_per_move=sim_per_move,
                num_workers=NUM_MCTS_WORKERS,
                reuse_tree=REUSE_TREE,  # Reuse MCTS tree across episodes for each config
                render_actual_game=RENDER_GAME_DURING_EXPERIMENT, # For experiment, usually False
                record_video=RECORD_VIDEO_DURING_EXPERIMENT,   # For experiment, usually False
                experiment_seed=MAIN_EXPERIMENT_SEED, # Consistent seed for this set of runs
                video_folder_suffix=f"_sim{sim_per_move}_depth{rollout_depth_val}"
            )
            
            experiment_results_list.append({
                'sim_per_move': sim_per_move,
                'rollout_depth': rollout_depth_val,
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'all_rewards': all_rewards,
                'search_time_per_move': search_time_per_move,
                'search_time_std': search_time_std,
            })
            logger.info(f"--- Finished Config {current_config}/{total_configs} ---\n\n")

    # --- Output Original Data ---
    logger.info("\n\n--- Experiment Finished: Summary of Results ---")
    print("===================================================================================")
    print(f"{'Sim/Move':<12} | {'Rollout Depth':<15} | {'Avg Reward':<15} | {'Std Reward':<15} | {'All Rewards'}")
    print("-----------------------------------------------------------------------------------")
    for res in experiment_results_list:
        rewards_str = ", ".join([f"{r:.1f}" for r in res['all_rewards']])
        print(f"{res['sim_per_move']:<12} | {res['rollout_depth']:<15} | {res['avg_reward']:<15.2f} | {res['std_reward']:<15.2f} | {rewards_str} ")
    print("===================================================================================")
    # summary of time
    print("===================================================================================")
    print(f"{'Sim/Move':<12} | {'Rollout Depth':<15} | {'Avg Search Time':<15} | {'Search Time Std':<15}")
    print("-----------------------------------------------------------------------------------")
    for res in experiment_results_list:
        print(f"{res['sim_per_move']:<12} | {res['rollout_depth']:<15} | {res['search_time_per_move']:<15.4f} | {res['search_time_std']:<15.4f}")
    print("===================================================================================")
    logger.info("Experiment results printed above. Now plotting...")

    # --- Plotting ---
    if experiment_results_list:
        # Plot 1: Effect of Simulations per Move (lines are different Rollout Depths)
        plot_experiment_results(
            results_list=experiment_results_list,
            env_name=SELECTED_ENV_NAME,
            fixed_param_name='Rollout Depth',
            varying_param_name='Simulations per Move',
            fixed_param_values=rollout_depth_options,
            varying_param_values=simulations_per_move_options
        )

        # Plot 2: Effect of Rollout Depth (lines are different Simulations per Move)
        plot_experiment_results(
            results_list=experiment_results_list,
            env_name=SELECTED_ENV_NAME,
            fixed_param_name='Simulations per Move',
            varying_param_name='Rollout Depth',
            fixed_param_values=simulations_per_move_options,
            varying_param_values=rollout_depth_options
        )
    else:
        logger.info("No results to plot.")

    logger.info("Experiment script finished.")
    
    # nohup python MCTS_basic_experiment.py > cart_pole_experiment.log 2>&1 &