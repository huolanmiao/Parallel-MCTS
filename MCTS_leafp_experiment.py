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
                           worker_seed, random_seed, gamma=0.95):
    """
    Performs a random rollout from a given state for a worker process.
    task_data can be:
        - cloned_env_state (for 'ale' or 'generic_clone')
        - (action_sequence_to_node, initial_obs_for_replay_verification) for 'action_replay'
    """
    # print(f"Simulated rollout for task with action sequence: {task_data[0]} using worker seed {worker_seed}")
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
    def __init__(self, env_name, num_workers=None, ucb_exploration_const=1.41, rollout_depth=100, num_simulation=1000):
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
        
        # self.base_seed = random.randint(0, 1_000_000) # Base seed for MCTS search consistency if needed
        self.base_seed = 42

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
            current_node.wins += cur_value_estimation
            cur_value_estimation = node.reward_at_node + gamma * cur_value_estimation 
            current_node = current_node.parent

    def search(self, current_observation, current_info, is_current_state_terminated, is_current_state_truncated, history_actions, seed, reuse_tree=False): 
        
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
            num_to_batch = self.num_workers
            actual_batch_size = min(num_to_batch, num_total_simulations_for_this_move - simulations_done)
            

            batch_simulation_tasks_data = [] # Holds (node_for_bp, task_args_for_pool)
            
            promising_node = self._select_promising_node(root_node)
            node_to_evaluate = promising_node
            # print(f"promising node: {promising_node.action_sequence_from_root}")
            if not promising_node.is_terminal:
                node_to_evaluate = self._expand_node(promising_node, seed)
            
            # If after selection/expansion, the node is terminal, backpropagate its intrinsic reward
            if node_to_evaluate.is_terminal:
                self._backpropagate(node_to_evaluate, node_to_evaluate.reward_at_node) # Or 0 if terminal means end of game with no further reward
                simulations_done += 1
                continue # No need to simulate rollouts for terminal nodes
            
            task_data_for_worker = (node_to_evaluate.action_sequence_from_root, root_node.observation)            
            worker_seed = seed
            # change seed for each task to ensure diverse rollouts
            batch_simulation_tasks_data = [(self.env_name, task_data_for_worker,
                            self.rollout_depth, self.possible_actions_indices, worker_seed, seed) for seed in range(worker_seed, worker_seed + actual_batch_size)]
            
            rollout_rewards = []
            if self.pool:                
                rollout_rewards = self.pool.starmap(_simulate_rollout_task, batch_simulation_tasks_data)
            else:
                for task_args in batch_simulation_tasks_data:
                    rollout_rewards.append(_simulate_rollout_task(*task_args))
            
            # Backpropagate the rewards from the rollouts
            mean_rollout_reward = np.mean(rollout_rewards)
            std_rollout_reward = np.std(rollout_rewards)
            # if std_rollout_reward != 0:
            #     print(f"Mean Rollout Reward: {mean_rollout_reward:.2f}, Std Dev: {std_rollout_reward:.2f}")
            self._backpropagate(node_to_evaluate, mean_rollout_reward)
            
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
                           num_workers=None, render=False, reuse_tree=False):
    logger.info(f"Initializing MCTS agent for {env_name}...")
    
    mcts_rollout_depth = rollout_depth # Default depth of random rollouts within MCTS
    mcts_sim_budget = simulations_per_move # Number of tree traversals/simulations per move

    mcts_agent = MCTS_Parallel_Simulations(env_name, 
                                           num_workers=num_workers, 
                                           rollout_depth=mcts_rollout_depth,
                                           num_simulation=mcts_sim_budget,
                                           ) 

    render_mode = 'rgb_array' if render else None
    env = gym.make(env_name, render_mode=render_mode)
    if render:
        env = RecordVideo(env, video_folder=f"{env_name}-agent", name_prefix="eval",
                    episode_trigger=lambda x: True)
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)
        
    total_rewards_all_episodes = []

    # Collect time for each search
    search_times = []
    
    logger.info(f"Starting evaluation for {num_episodes} episodes on {env_name}...")
    for episode_num in range(num_episodes):
        seed = mcts_agent.base_seed + episode_num
        # obs, info = env.reset(seed=mcts_agent.base_seed + episode_num) # Seed for episode reproducibility
        obs, info = env.reset(seed=seed)
        
        history_actions = []
        terminated = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        search_time_per_episode = []  # Track search time for this episode

        while not (terminated or truncated):
            start_time = time.time()
            action = mcts_agent.search(obs, info, terminated, truncated, history_actions, seed, reuse_tree=reuse_tree)
            search_time = time.time() - start_time
            search_time_per_episode.append(search_time)
            history_actions.append(action)
            

            obs, reward, terminated, truncated, new_info = env.step(action)
            info = new_info # Update info for next search call
            
            episode_reward += reward
            step_count += 1
            
            # if render: env.render()
            
            # logger.debug(f"  Action: {action}, Reward: {reward}, Total Ep Reward: {episode_reward}")
            if terminated or truncated:
                logger.info(f"Episode {episode_num + 1} finished after {step_count} steps. Reward: {episode_reward}")
                break
        
        search_times.extend(search_time_per_episode)
        
        total_rewards_all_episodes.append(episode_reward)
        mcts_agent.cur_root_node = None  # Reset root node for next episode

    mcts_agent.shutdown_pool()
    env.close()

    avg_reward = sum(total_rewards_all_episodes) / num_episodes if num_episodes > 0 else 0
    std_reward = np.std(total_rewards_all_episodes) if num_episodes > 0 else 0.0
    avg_search_time_per_move = np.mean(search_times)
    std_search_time_per_move = np.std(search_times)
    
    logger.info(f"Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}")
    logger.info(f"Avg Search Time/Move: {avg_search_time_per_move:.4f}s, Std Search Time: {std_search_time_per_move:.4f}s\n")
    
    return avg_reward, std_reward, total_rewards_all_episodes, avg_search_time_per_move, std_search_time_per_move

def plot_experiment_data(results_list, env_name,
                         fixed_line_param_config,
                         varying_x_axis_param_config,
                         fixed_line_param_values,
                         varying_x_axis_param_values,
                         y_metric_config,
                         plot_title_prefix="Effect of"):
    """
    为实验结果生成图表的通用函数。
    示例 fixed_line_param_config: {'display_name': 'Number of Workers', 'dict_key': 'workers'}
    示例 varying_x_axis_param_config: {'display_name': 'Simulations per Move', 'dict_key': 'simulations'}
    示例 y_metric_config: {'value_key': 'avg_reward', 'error_key': 'std_reward', 'axis_label': 'Average Episode Reward'}
    """
    plt.figure(figsize=(12, 7))

    fixed_param_display_name = fixed_line_param_config['display_name']
    fixed_param_dict_key = fixed_line_param_config['dict_key']

    varying_param_display_name = varying_x_axis_param_config['display_name']
    varying_param_dict_key = varying_x_axis_param_config['dict_key']

    y_value_key = y_metric_config['value_key']
    y_error_key = y_metric_config['error_key']
    y_axis_label = y_metric_config['axis_label']

    for fixed_val in fixed_line_param_values:
        # 临时列表，用于存储找到的数据点
        temp_x_values = []
        temp_y_values = []
        temp_y_errors = []

        # 遍历所有可能的 varying parameter 值以查找匹配项
        for vary_val in varying_x_axis_param_values: # 此处无需排序
            for res_item in results_list:
                # 检查 res_item 中的值是否为数字类型，以避免与 None 或其他类型比较时出错
                res_fixed_val = res_item.get(fixed_param_dict_key)
                res_varying_val = res_item.get(varying_param_dict_key)

                if isinstance(res_fixed_val, (int, float)) and \
                   isinstance(res_varying_val, (int, float)) and \
                   res_fixed_val == fixed_val and \
                   res_varying_val == vary_val:
                    
                    temp_x_values.append(vary_val)
                    temp_y_values.append(res_item[y_value_key])
                    temp_y_errors.append(res_item[y_error_key])
                    # 假设 results_list 中的 (fixed_val, vary_val) 组合是唯一的
                    break 
        
        if temp_x_values: # 如果为此 fixed_val 找到了任何数据
            # 根据 x 值 (vary_val) 对收集到的数据点进行排序
            sorted_indices = np.argsort(temp_x_values)
            
            plot_x = np.array(temp_x_values)[sorted_indices]
            plot_y = np.array(temp_y_values)[sorted_indices]
            plot_y_err = np.array(temp_y_errors)[sorted_indices]

            if len(plot_x) == len(plot_y) and len(plot_x) == len(plot_y_err): # 如果这样构造，应该始终为 true
                plt.plot(plot_x, plot_y, marker='o', linestyle='-', label=f'{fixed_param_display_name} = {fixed_val}')
                plt.fill_between(plot_x, np.array(plot_y) - np.array(plot_y_err), np.array(plot_y) + np.array(plot_y_err), alpha=0.2)
            else:
                # 当前逻辑不应发生这种情况。
                logger.warning(f"Data consistency issue for {fixed_param_display_name} = {fixed_val}. X:{len(plot_x)}, Y:{len(plot_y)}, Err:{len(plot_y_err)}")
        else:
            logger.info(f"No data found for line with {fixed_param_display_name} = {fixed_val}. Skipping this line.")

    plt.title(f'{plot_title_prefix} {varying_param_display_name} on {y_axis_label} ({env_name})')
    plt.xlabel(varying_param_display_name)
    plt.ylabel(y_axis_label)
    # Only show legend if there are plotted lines
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 清理文件名
    safe_env_name = env_name.replace('/', '_').replace('\\', '_')
    safe_varying_param_name = varying_param_display_name.replace(' ', '_')
    safe_y_label_for_fname = y_axis_label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    safe_fixed_param_name = fixed_param_display_name.replace(' ', '_')

    filename = f"{safe_env_name}_plot_X_{safe_varying_param_name}_Y_{safe_y_label_for_fname}_lines_{safe_fixed_param_name}.png"
    plt.savefig(filename)
    logger.info(f"Plot saved as {filename}")
    # plt.show() # 在脚本中通常注释掉
    plt.close() # 关闭图形以释放内存，如果生成许多图表则很重要
    
if __name__ == '__main__':
    random.seed(42) 
    # --- Configuration for Evaluation ---
    # ATARI_ENV_NAME = 'ALE/Pong-v5'
    ATARI_ENV_NAME = 'ALE/Breakout-v5' 
    SELECTED_ENV_NAME = 'CartPole-v1' # A simple environment for testing non-Atari compatibility

    # Experiment Parameters
    NUM_EPISODES_FOR_EXPERIMENT = 5  # Keep low for faster experiments
    ROLLOUT_DEPTH_FOR_EXPERIMENT = 15 # MCTS internal rollout depth

    # simulations_options = [50, 100, 200, 400, 600, 1000] # SIMULATIONS_PER_MOVE    
    # worker_options = [1, 2, 4, 8, 16, 32] # Number of MCTS workers to test
    
    # DEGUB
    simulations_options = [50, 100] # SIMULATIONS_PER_MOVE    
    worker_options = [1, 2] # Number of MCTS workers to test
    # --------------------------------------

    logger.info(f"Starting MCTS search time experiments on {SELECTED_ENV_NAME}")
    logger.info(f"Worker options: {worker_options}")
    logger.info(f"Simulations per move options: {simulations_options}")
    logger.info(f"Episodes per setting: {NUM_EPISODES_FOR_EXPERIMENT}, Rollout depth: {ROLLOUT_DEPTH_FOR_EXPERIMENT}\n\n")

    experiment_results = []

    if SELECTED_ENV_NAME.startswith("ALE/"):
        try:
            import ale_py # Check if Atari ROMs are available
            logger.info("ALE environment selected. Ensure you have 'pip install gymnasium[accept-rom-license]' if needed.")
        except ImportError:
            logger.error("ALE environment selected, but 'ale_py' not found. Skipping Atari experiments.")
            SELECTED_ENV_NAME = 'CartPole-v1' # Fallback
            logger.warning(f"Falling back to {SELECTED_ENV_NAME} for experiments.")
    
    total_experiments = len(simulations_options) * len(worker_options)
    current_experiment = 0

    for workers in worker_options:
        for sims in simulations_options:
            current_experiment += 1
            logger.info(f"\n------------- Running Experiment {current_experiment}/{total_experiments} ----------------------------------")
            logger.info(f"Sim/move={sims}, Workers={workers}")
            
            avg_r, std_r, all_rewards, avg_search_t, std_search_t = evaluate_mcts_on_env(
                env_name=SELECTED_ENV_NAME,
                num_episodes=NUM_EPISODES_FOR_EXPERIMENT,
                rollout_depth=ROLLOUT_DEPTH_FOR_EXPERIMENT,
                simulations_per_move=sims,
                num_workers=workers,
                render=False, # Disable rendering for speed during experiments
                reuse_tree=False # Isolate search time for each move decision
            )
            experiment_results.append({
                'workers': workers,
                'simulations': sims,
                'avg_search_time': avg_search_t,
                'std_search_time': std_search_t,
                'avg_reward': avg_r,
                'std_reward': std_r,
                'all_rewards': all_rewards
            })

    # --- Output Original Data ---
    logger.info("\n\n--- Experiment Finished: Summary of Results ---")
    print("===================================================================================")
    print(f"{'workers':<12} | {'sim_per_move':<15} | {'avg_reward':<15} | {'std_reward':<15} | {'all_rewards'} ")
    print("-----------------------------------------------------------------------------------")
    for res in experiment_results:
        rewards_str = ", ".join([f"{r:.1f}" for r in res['all_rewards']])
        print(f"{res['workers']:<12} | {res['simulations']:<15} | {res['avg_reward']:<15.2f} | {res['std_reward']:<15.2f} | {rewards_str}")
    print("===================================================================================")
    # summary of time
    print("===================================================================================")
    print(f"{'workers':<12} | {'sim_per_move':<15} | {'search_time_per_move':<15} | {'search_time_std':<15}")
    print("-----------------------------------------------------------------------------------")
    for res in experiment_results:
        print(f"{res['workers']:<12} | {res['simulations']:<15} | {res['avg_search_time']:<15.4f} | {res['std_search_time']:<15.4f}")
    print("===================================================================================")
    
    # --- 绘制结果图表 ---
    logger.info("Generating plots...")

    # 定义Y轴指标的配置
    y_metric_reward_config = {'value_key': 'avg_reward', 'error_key': 'std_reward', 'axis_label': 'Average Episode Reward'}
    y_metric_search_time_config = {'value_key': 'avg_search_time', 'error_key': 'std_search_time', 'axis_label': 'Average Search Time (s)'}

    # 定义参数配置
    workers_config = {'display_name': 'Number of Workers', 'dict_key': 'workers'}
    simulations_config = {'display_name': 'Simulations per Move', 'dict_key': 'simulations'}

    # 图表1: X轴=Simulations per Move, 线条=Number of Workers, Y轴=Average Reward
    plot_experiment_data(
        results_list=experiment_results,
        env_name=SELECTED_ENV_NAME,
        fixed_line_param_config=workers_config,
        varying_x_axis_param_config=simulations_config,
        fixed_line_param_values=worker_options,
        varying_x_axis_param_values=simulations_options,
        y_metric_config=y_metric_reward_config,
        plot_title_prefix="Effect of"
    )

    # 图表2: X轴=Simulations per Move, 线条=Number of Workers, Y轴=Average Search Time
    plot_experiment_data(
        results_list=experiment_results,
        env_name=SELECTED_ENV_NAME,
        fixed_line_param_config=workers_config,
        varying_x_axis_param_config=simulations_config,
        fixed_line_param_values=worker_options,
        varying_x_axis_param_values=simulations_options,
        y_metric_config=y_metric_search_time_config,
        plot_title_prefix="Effect of"
    )

    # 图表3: X轴=Number of Workers, 线条=Simulations per Move, Y轴=Average Reward
    plot_experiment_data(
        results_list=experiment_results,
        env_name=SELECTED_ENV_NAME,
        fixed_line_param_config=simulations_config,
        varying_x_axis_param_config=workers_config,
        fixed_line_param_values=simulations_options,
        varying_x_axis_param_values=worker_options,
        y_metric_config=y_metric_reward_config,
        plot_title_prefix="Effect of"
    )

    # 图表4: X轴=Number of Workers, 线条=Simulations per Move, Y轴=Average Search Time
    plot_experiment_data(
        results_list=experiment_results,
        env_name=SELECTED_ENV_NAME,
        fixed_line_param_config=simulations_config,
        varying_x_axis_param_config=workers_config,
        fixed_line_param_values=simulations_options,
        varying_x_axis_param_values=worker_options,
        y_metric_config=y_metric_search_time_config,
        plot_title_prefix="Effect of"
    )
    
    logger.info("Experiment run and plotting complete. Check for .png files for graphs.")
    # nohup python MCTS_leafp_experiment.py > cart_pole_leafp_exp.log 2>&1 &

