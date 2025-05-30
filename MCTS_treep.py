import math
import random
import copy
import time
from multiprocessing import Pool, cpu_count
import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np

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
    search_time_per_move = np.mean(search_times)
    search_time_std = np.std(search_times)
    
    logger.info(f"\nEvaluation Complete for {env_name}.")
    logger.info(f"Number of Episodes: {num_episodes}")
    logger.info(f"Simulations per Move (MCTS budget): {mcts_sim_budget}")
    logger.info(f"Rollout Depth (within simulation): {mcts_rollout_depth}")
    logger.info(f"Number of Parallel Workers: {mcts_agent.num_workers if mcts_agent.num_workers > 0 else 'Serial'}")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Standard Deviation of Rewards: {std_reward:.2f}")
    logger.info(f"Individual Rewards: {total_rewards_all_episodes}")
    logger.info(f"Average Search Time per Move: {search_time_per_move:.4f} seconds")
    logger.info(f"Search Time Standard Deviation: {search_time_std:.4f} seconds")
    return total_rewards_all_episodes


if __name__ == '__main__':
    random.seed(42) 
    # --- Configuration for Evaluation ---
    # ATARI_ENV_NAME = 'ALE/Pong-v5'
    ATARI_ENV_NAME = 'ALE/Breakout-v5' 
    CLASSIC_CONTROL_ENV_NAME = 'CartPole-v1' # A simple environment for testing non-Atari compatibility
    # CLASSIC_CONTROL_ENV_NAME = 'LunarLander-v2' # More complex, discrete actions

    SELECTED_ENV_NAME = CLASSIC_CONTROL_ENV_NAME # Change this to test different environments

    NUM_EPISODES_TO_RUN = 5
    SIMULATIONS_PER_MOVE = 1000 # Budget for MCTS search (number of tree traversals/sims)
    ROLLOUT_DEPTH_MCTS = 100 # Max depth for random rollouts within each simulation 
    # NUM_MCTS_WORKERS = cpu_count() // 2 if cpu_count() > 1 else 0 # Use half cores or run serially
    NUM_MCTS_WORKERS = 1
    RENDER_GAME = True

    logger.info(f"Starting MCTS evaluation on {SELECTED_ENV_NAME} with {NUM_MCTS_WORKERS} worker(s).")
    
    # Ensure gymnasium[accept-rom-license] is installed if using Atari
    if SELECTED_ENV_NAME.startswith("ALE/"):
            pass # Assume license is accepted via pip install gymnasium[accept-rom-license]

    evaluate_mcts_on_env(
        env_name=SELECTED_ENV_NAME,
        num_episodes=NUM_EPISODES_TO_RUN,
        rollout_depth=ROLLOUT_DEPTH_MCTS,
        simulations_per_move=SIMULATIONS_PER_MOVE,
        num_workers=NUM_MCTS_WORKERS,
        render=RENDER_GAME,
        reuse_tree=False,
    )
    
    # nohup python MCTS_basic_mont.py > test_1.log 2>&1 &

