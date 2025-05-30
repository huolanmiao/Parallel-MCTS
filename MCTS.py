import math
import random
import copy
import time
from multiprocessing import Pool, cpu_count

import gymnasium as gym
import ale_py
from ale_py import ALEInterface
# ale = ALEInterface()
gym.register_envs(ale_py)
# Ensure ale-py is installed for Atari environments
# from ale_py import ALEInterface # Not directly used but good to know it's the backend

# Helper function for multiprocessing pool.
# This must be a top-level function to be pickleable by multiprocessing.
def _simulate_rollout_task(env_name, initial_state_bytes, rollout_depth_limit, possible_actions):
    """
    Performs a random rollout from a given state.
    This function is executed by worker processes.
    """
    # Create a new environment instance for this worker process
    # Render mode is None as workers should not render.
    # env = gym.make(env_name, render_mode=None) 
    env = gym.make(env_name)
    env.reset()  # Initialize ALE
    
    # Restore the environment to the specific state for rollout
    # This is a critical step for MCTS with Atari environments.
    try:
        env.ale.restoreSystemState(initial_state_bytes)
    except Exception as e:
        # Fallback if restoreSystemState fails or not available (should be for Atari)
        # This indicates a problem with the environment setup or state handling.
        print(f"Error restoring system state in worker: {e}")
        env.close()
        return 0 # Or raise an error

    current_rollout_depth = 0
    is_terminal = False # Check if the restored state is already terminal
    
    # To check if the restored state is terminal, we need to get the current game over status.
    # This is typically available via env.ale.game_over() or by checking terminated/truncated flags
    # after a dummy step, but for simplicity, we assume the node passed for simulation is not terminal.
    # The main MCTS logic should handle nodes that are already terminal.

    total_rollout_reward = 0.0

    # Perform the random rollout
    while not is_terminal and current_rollout_depth < rollout_depth_limit:
        action = random.choice(possible_actions)
        
        # We don't need the observation in the rollout, just rewards and termination.
        _obs, reward, terminated, truncated, _info = env.step(action)
        is_terminal = terminated or truncated
        total_rollout_reward += reward
        current_rollout_depth += 1
    
    env.close()
    return total_rollout_reward

class Node:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    """
    def __init__(self, state_bytes, parent=None, action_that_led_here=None, 
                 possible_actions=None, is_terminal_state=False, terminal_reward=0.0,
                 ucb_exploration_const=1.41):
        self.state_bytes = state_bytes  # Serialized state from env.ale.cloneSystemState()
        self.parent = parent
        self.action_that_led_here = action_that_led_here # Action parent took to reach this node

        self.children = []
        # Shuffle untried actions to ensure random exploration order.
        self.untried_actions = possible_actions.copy() if possible_actions else []
        random.shuffle(self.untried_actions)

        self.wins = 0.0  # Sum of rewards from rollouts through this node
        self.visits = 0 # Number of times this node has been visited

        self.is_terminal = is_terminal_state
        # If this node represents a terminal state, this is the reward obtained for reaching it.
        self.terminal_reward_value = terminal_reward 
        
        self.C = ucb_exploration_const # Exploration constant for UCB1

    def is_fully_expanded(self):
        """Checks if all possible actions from this node have led to child nodes."""
        return not self.untried_actions

    def add_child(self, child_node):
        """Adds a child node to this node."""
        self.children.append(child_node)

    def select_best_child_ucb1(self):
        """
        Selects the child with the highest UCB1 score.
        UCB1 = (wins/visits) + C * sqrt(log(parent_visits) / visits)
        """
        if not self.children:
            return None
            
        best_child = None
        best_score = -float('inf')

        for child in self.children:
            if child.visits == 0: # Avoid division by zero; prioritize unvisited children
                # Assign a very high score to unvisited children to ensure they are picked.
                # This is a common strategy, or one could return such a child immediately.
                score = float('inf') 
            else:
                exploitation_term = child.wins / child.visits
                exploration_term = self.C * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation_term + exploration_term
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class MCTS_Parallel_Simulations:
    """
    Monte Carlo Tree Search with parallelized simulation phase.
    """
    def __init__(self, env_name, num_workers=None, ucb_exploration_const=1.41, rollout_depth=100):
        self.env_name = env_name
        self.ucb_exploration_const = ucb_exploration_const
        self.rollout_depth = rollout_depth

        if num_workers is None:
            self.num_workers = cpu_count() # Default to all available CPUs
        else:
            self.num_workers = num_workers
            
        if self.num_workers > 0:
            # Initialize the multiprocessing pool.
            self.pool = Pool(processes=self.num_workers)
        else:
            self.pool = None # Serial execution

        # Get action space info from a temporary environment instance.
        _temp_env = gym.make(self.env_name)
        self.action_space_n = _temp_env.action_space.n
        self.possible_actions = list(range(self.action_space_n))
        _temp_env.close()

    def _select_promising_node(self, root_node):
        """
        Traverses the tree from the root, selecting nodes using UCB1,
        until a leaf node (not fully expanded or terminal) is reached.
        """
        node = root_node
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node  # Ready for expansion
            else:
                if not node.children: 
                    # This case should ideally not be reached if logic is correct,
                    # as a fully expanded non-terminal node should have children.
                    # If it occurs, it's a leaf that cannot be expanded further.
                    return node 
                node = node.select_best_child_ucb1()
                if node is None: # Should not happen if children exist
                    return root_node # Fallback, though indicates an issue
        return node  # Terminal node

    def _expand_node(self, parent_node, parent_env_state_bytes):
        """
        Expands a node by trying an untried action, creating a new child node.
        Returns the new child node.
        """
        if not parent_node.untried_actions:
            # Should not happen if called on a node that is not fully expanded.
            return None 

        action_to_try = parent_node.untried_actions.pop()

        # Create a temporary environment to take the action and get the child state.
        temp_env = gym.make(self.env_name, render_mode=None)
        temp_env.reset()
        try:
            temp_env.ale.restoreSystemState(parent_env_state_bytes)
        except Exception as e:
            print(f"Error restoring system state in _expand_node: {e}")
            temp_env.close()
            return None # Indicate failure

        _obs, step_reward, terminated, truncated, _info = temp_env.step(action_to_try)
        child_state_bytes = temp_env.ale.cloneSystemState()
        is_child_terminal = terminated or truncated
        
        child_node = Node(state_bytes=child_state_bytes, 
                          parent=parent_node,
                          action_that_led_here=action_to_try,
                          possible_actions=self.possible_actions if not is_child_terminal else [],
                          is_terminal_state=is_child_terminal,
                          terminal_reward=step_reward, # Store reward if this child is terminal
                          ucb_exploration_const=self.ucb_exploration_const)
        parent_node.add_child(child_node)
        temp_env.close()
        return child_node

    def _backpropagate(self, node, reward):
        """
        Updates the win/visit counts from the given node back up to the root.
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.wins += reward
            current_node = current_node.parent

    def search(self, initial_env_state_bytes, num_total_simulations_for_this_move):
        """
        Performs the MCTS search for a given number of simulations to decide the best next action.
        """
        # Check if the initial state itself is terminal.
        # This requires a temporary environment.
        _temp_check_env = gym.make(self.env_name)
        _temp_check_env.reset()
        try:
            _temp_check_env.ale.restoreSystemState(initial_env_state_bytes)
            # A bit of a hack: step with a no-op action (0) if possible or check game_over
            # For robust check, one might need to know if the game has ended from ALE directly.
            # obs, rew, term, trunc, inf = _temp_check_env.step(0) # Assuming 0 is often a safe action
            # initial_is_terminal = term or trunc
            initial_is_terminal = _temp_check_env.ale.game_over() # More direct
        except Exception:
            initial_is_terminal = False # Assume not terminal if check fails
        finally:
            _temp_check_env.close()

        root_node = Node(state_bytes=initial_env_state_bytes, 
                         possible_actions=self.possible_actions if not initial_is_terminal else [],
                         is_terminal_state=initial_is_terminal,
                         terminal_reward=0.0, # Root node's terminal reward is typically 0 unless game starts ended
                         ucb_exploration_const=self.ucb_exploration_const)

        if root_node.is_terminal: # If the game has already ended at the root.
            return random.choice(self.possible_actions) if self.possible_actions else 0 # No valid move or default


        simulations_done = 0
        while simulations_done < num_total_simulations_for_this_move:
            # Determine batch size for this iteration
            # If using pool, batch up to num_workers. If serial, batch size is 1.
            current_batch_workers = self.num_workers if self.pool else 1
            actual_batch_size = min(current_batch_workers, num_total_simulations_for_this_move - simulations_done)
            if actual_batch_size <= 0: # Ensure we always try to do at least one if budget remains
                 actual_batch_size = 1 if num_total_simulations_for_this_move > simulations_done else 0

            if actual_batch_size == 0: break 

            nodes_for_simulation_tasks = [] # Stores (node_object_in_main_tree)
            
            # Phase 1 & 2: Selection & Expansion for each path in the batch
            for _ in range(actual_batch_size):
                promising_node = self._select_promising_node(root_node)

                node_for_sim_or_terminal_eval = None
                if promising_node.is_terminal:
                    node_for_sim_or_terminal_eval = promising_node
                elif not promising_node.is_fully_expanded():
                    # state_bytes for expansion is promising_node.state_bytes
                    node_for_sim_or_terminal_eval = self._expand_node(promising_node, promising_node.state_bytes)
                    if node_for_sim_or_terminal_eval is None: # Expansion failed
                        continue 
                else:
                    # This case means _select_promising_node returned a fully expanded, non-terminal node.
                    # This shouldn't happen if select_best_child_ucb1 always returns a child.
                    # If it does, it implies an issue or that we are at a deep leaf.
                    # For robustness, we can treat it as a node to simulate from.
                    node_for_sim_or_terminal_eval = promising_node 

                if node_for_sim_or_terminal_eval:
                    if node_for_sim_or_terminal_eval.is_terminal:
                        # If expansion led to a terminal node, backpropagate its stored terminal reward.
                        self._backpropagate(node_for_sim_or_terminal_eval, node_for_sim_or_terminal_eval.terminal_reward_value)
                        simulations_done += 1 
                    else:
                        # Add to batch for actual simulation via _simulate_rollout_task
                        nodes_for_simulation_tasks.append(node_for_sim_or_terminal_eval)
                else: # Path did not yield a node (e.g. expansion failed)
                    simulations_done +=1 # Count as a completed path attempt


            # Phase 3: Simulation for collected non-terminal leaf nodes
            simulation_states_for_pool = [node.state_bytes for node in nodes_for_simulation_tasks]
            rewards_from_rollouts = []

            if simulation_states_for_pool:
                if self.pool:
                    # Distribute simulation tasks to the worker pool.
                    # Each task is (env_name, state_bytes, rollout_depth, possible_actions)
                    tasks_for_starmap = [
                        (self.env_name, state_bytes, self.rollout_depth, self.possible_actions) 
                        for state_bytes in simulation_states_for_pool
                    ]
                    rewards_from_rollouts = self.pool.starmap(_simulate_rollout_task, tasks_for_starmap)
                else: # Serial simulation
                    rewards_from_rollouts = [
                        _simulate_rollout_task(self.env_name, state_bytes, self.rollout_depth, self.possible_actions) 
                        for state_bytes in simulation_states_for_pool
                    ]
            
            # Phase 4: Backpropagation for simulated rollouts
            for i, node_simulated_from in enumerate(nodes_for_simulation_tasks):
                self._backpropagate(node_simulated_from, rewards_from_rollouts[i])
            
            simulations_done += len(nodes_for_simulation_tasks)

        # After all simulations, choose the best action from the root node's children.
        # Typically, the child with the most visits is chosen for robustness.
        if not root_node.children:
             # This can happen if num_total_simulations_for_this_move is very small (e.g., 0 or 1)
             # or if the root was terminal.
            return random.choice(self.possible_actions) if self.possible_actions else 0


        best_child = max(root_node.children, key=lambda child: child.visits, default=None)
        return best_child.action_that_led_here if best_child else random.choice(self.possible_actions)


    def shutdown_pool(self):
        """Closes the multiprocessing pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

def evaluate_mcts_on_atari(env_name, num_episodes=10, num_simulations_per_move=100, 
                           num_workers=None, rollout_depth=50, render=False):
    """
    Evaluates the MCTS agent on a given Atari environment.
    """
    print(f"Initializing MCTS agent for {env_name}...")
    mcts_agent = MCTS_Parallel_Simulations(env_name, 
                                           num_workers=num_workers, 
                                           rollout_depth=rollout_depth)

    # render_mode = 'human' if render else None
    # env = gym.make(env_name, render_mode=render_mode)
    env = gym.make(env_name)
    
    total_rewards_all_episodes = []

    print(f"Starting evaluation for {num_episodes} episodes...")
    for episode_num in range(num_episodes):
        obs, info = env.reset()
        current_env_state_bytes = env.ale.cloneSystemState()
        
        terminated = False
        truncated = False
        episode_reward = 0
        step_count = 0

        while not (terminated or truncated):
            print(f"\nEpisode {episode_num + 1}, Step {step_count + 1}: Thinking...")
            action = mcts_agent.search(current_env_state_bytes, num_simulations_per_move)
            
            if action is None: # Fallback if search returns None (should be rare)
                print("MCTS search returned None, taking random action.")
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            current_env_state_bytes = env.ale.cloneSystemState() # Update state bytes
            
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
            
            print(f"  Action: {action}, Reward: {reward}, Total Ep Reward: {episode_reward}")
            if terminated or truncated:
                print(f"Episode {episode_num + 1} finished after {step_count} steps.")
                break
        
        total_rewards_all_episodes.append(episode_reward)
        print(f"Episode {episode_num + 1}: Total Reward = {episode_reward}")

    mcts_agent.shutdown_pool()
    env.close()

    avg_reward = sum(total_rewards_all_episodes) / num_episodes if num_episodes > 0 else 0
    print(f"\nEvaluation Complete for {env_name}.")
    print(f"Number of Episodes: {num_episodes}")
    print(f"Simulations per Move: {num_simulations_per_move}")
    print(f"Number of Parallel Workers: {mcts_agent.num_workers}")
    print(f"Rollout Depth: {rollout_depth}")
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    print(f"Individual Episode Rewards: {total_rewards_all_episodes}")
    return total_rewards_all_episodes


if __name__ == '__main__':
    # --- Configuration for Evaluation ---
    ATARI_ENV_NAME = 'ALE/Breakout-v5'  # Example: Pong. Try 'ALE/Breakout-v5', 'ALE/MsPacman-v5' etc.
    NUM_EPISODES_TO_RUN = 3       # Keep low for quick testing; increase for proper evaluation.
    SIMULATIONS_PER_MOVE = 50     # Higher is better but slower. Atari might need 100s-1000s for good play.
    NUM_MCTS_WORKERS = cpu_count()  # Use all available CPU cores, or set to e.g., 4. Set to 0 for serial.
    ROLLOUT_POLICY_DEPTH = 30       # Max depth for random rollouts.
    RENDER_GAME = False             # Set to True to watch the agent play (slower).

    print(f"Using {NUM_MCTS_WORKERS} worker(s) for MCTS simulations.")

    # It's crucial that the main script execution is protected by `if __name__ == '__main__':`
    # when using multiprocessing on platforms like Windows.
    
    # Ensure you have the ROMs by installing ale-py and accepting the license:
    # pip install gymnasium[atari] ale-py
    # gymnasium.utils.rom_checker.accept_rom_license() # or pip install gymnasium[accept-rom-license]

    try:
        evaluate_mcts_on_atari(
            env_name=ATARI_ENV_NAME,
            num_episodes=NUM_EPISODES_TO_RUN,
            num_simulations_per_move=SIMULATIONS_PER_MOVE,
            num_workers=NUM_MCTS_WORKERS,
            rollout_depth=ROLLOUT_POLICY_DEPTH,
            render=RENDER_GAME
        )
    except Exception as e:
        print(f"An error occurred during MCTS evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have Atari ROMs installed and licenses accepted.")
        print("Try: pip install gymnasium[atari] ale-py gymnasium[accept-rom-license]")

