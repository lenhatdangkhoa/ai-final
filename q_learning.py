import numpy as np
import random
import pandas as pd

class Q_learning:
    """
    Q-learning agent for a grid world with red light violations and obstacles.
    The agent learns to navigate a grid while respecting traffic lights and avoiding obstacles.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.goal = {(0 , i) for i in range(width)}
        self.actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stay
        self.obstacles = set()

        # Randomly place obstacles
        for _ in range(max(width, 25)):
            i = random.randint(0, width - 1)
            j = random.randint(0, height - 1)
            pos = (i, j)
            if pos not in self.goal:
                self.obstacles.add(pos)
   
        # Initialize Q-Table
        self.Q = {}
        for x in range(width):
            for y in range(height):
                for light in [0, 1]:
                    self.Q[(x, y, light)] = [0, 0, 0, 0, 0]

        # Hyperparameters
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 1.0
        self.decay_rate = 0.999
        self.episodes = 10000

    def get_next_state(self, state, action, light):
        """
        Get the next state based on the current state and action.
        The state is represented as (x, y, light).

        Args:
            state (tuple): Current state (x, y, light).
            action (int): Action taken by the agent.
            light (int): Current traffic light state (0 for red, 1 for green).
        Returns:
            tuple: Next state (x, y, light).
        """

        x, y, _ = state        
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.width-1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.height-1, y + 1)
        if (x, y) in self.obstacles:
            return (x, y, light) 
        return (x, y, light)

    def get_reward(self, state, action, next_state):
        """
        Calculate the reward based on the current state, action, and next state.
        Args:
            state (tuple): Current state (x, y, light).
            action (int): Action taken by the agent.
            next_state (tuple): Next state (x, y, light).
        Returns:
            tuple: Reward and terminal flag.
        """

        x, y, light = state
        next_x, next_y, _ = next_state
        if (x, y) == (next_x, next_y) and action != 4:
            return -2, False  # Hit wall
        if light == 0:  # Red light
            if action != 4:
                return -50, False  # Tried to move
            else:
                return 2, False  # Stayed on red – neutral reward
        if (next_x, next_y) in self.goal:
            return 50, True  # Goal reached
        if (next_x, next_y) in self.obstacles:
            return -2, False # Hit obstacle
        if action == 0 and light == 1:
            return 2,False  # Move up on green
        return -2 , False  # Time penalty
    
    def run(self, episodes=10000):
        """
        Run the Q-learning algorithm
        Args:
            episodes (int): Number of episodes to run.
        Returns:
            dict: Q-table with learned values.
        """

        light = 0  # Start with red light
        episode_rewards = []
        steps_to_goal = []
        red_light_violations = []
        obstacle_hits = []

        #Q-Learning
        for ep in range(episodes):
            steps_since_switch = 0 # Switch light every few steps
            light_duration = 4
            state = (self.width-1, random.randint(0, self.height - 1), light)
            done = False
            total_reward = 0
            steps = 0
            red_violations = 0
            collisions = 0
            while not done:
                steps += 1

                # Balance exploration and exploitation
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(self.actions)
                else:
                    action = np.argmax(self.Q[state])
                
                next_state = self.get_next_state(state, action, light)
                reward, terminal = self.get_reward(state, action, next_state)
                if (next_state[0], next_state[1]) in self.obstacles:
                    collisions += 1
                if light == 0 and action != 4:
                    red_violations += 1
                total_reward += reward

                # Stop value propagation from goal
                target = reward
                if not terminal:
                    target += self.gamma * max(self.Q[next_state])
                self.Q[state][action] += self.alpha * (target - self.Q[state][action]) # Bellman Equation
                
                # Switch light every few steps
                steps_since_switch += 1
                if steps_since_switch >= light_duration:
                    light = 1 - light
                    steps_since_switch = 0
                if terminal:
                    done = True
                state = next_state

            steps_to_goal.append(steps)
            episode_rewards.append(total_reward)
            red_light_violations.append(red_violations)
            obstacle_hits.append(collisions)
            self.epsilon = max(0.05, self.epsilon * self.decay_rate) # Decay epsilon
            
        metrics = pd.DataFrame({
        "Episode": list(range(episodes)),
        "Reward": episode_rewards,
        "Steps": steps_to_goal,
        "RedLightViolations": red_light_violations,
        "ObstacleCollisions": obstacle_hits
    })
        metrics.to_csv("metrics.csv", index=False)
        return self.Q
