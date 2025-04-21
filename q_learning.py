import numpy as np
import random
import copy


width, height = 75,75
goal = {(0 , i) for i in range(width)}
actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stay
obstacles = set()
for _ in range(max(width, 25)):
    i = random.randint(0, width - 1)
    j = random.randint(0, height - 1)
    pos = (i, j)
    if pos not in goal:
        obstacles.add(pos)

Q = {}
Q_10 = {}
Q_100 = {}
Q_1000 = {}
# Initialize Q-Table
for x in range(width):
    for y in range(height):
        for light in [0, 1]:
            Q[(x, y, light)] = [0, 0, 0, 0, 0]

alpha = 0.2
gamma = 0.9
epsilon = 1.0
decay_rate = 0.999
episodes = 10000

def get_next_state(state, action, light):
    x, y, _ = state
    # next_light = random.choice([0, 1])
    
    if action == 0:  # Up
        x = max(0, x - 1)
    elif action == 1:  # Down
        x = min(width-1, x + 1)
    elif action == 2:  # Left
        y = max(0, y - 1)
    elif action == 3:  # Right
        y = min(height-1, y + 1)
    
    if (x, y) in obstacles:
        return (x, y, light) 
    # Stay does not change position
    return (x, y, light)

def get_reward(state, action, next_state):
    x, y, light = state
    next_x, next_y, _ = next_state

    if (x, y) == (next_x, next_y) and action != 4:
        return -2, False  # Hit wall
    
    if light == 0:  # Red light
        if action != 4:
            return -50, False  # Tried to move
        else:
            return 2, False  # Stayed on red – neutral reward
    
    if (next_x, next_y) in goal:
        return 50, True  # Goal reached

    if (next_x, next_y) in obstacles:
        return -2, False # Hit obstacle
    
    if action == 0 and light == 1:
        return 2,False  # Move up on green
    
    return -2 , False  # Time penalty



light = 0  # Start with red light
episode_rewards = []
steps_to_goal = []
red_light_violations = []
obstacle_hits = []
#Q-Learning
for ep in range(episodes):
    if ep == 10:
        Q_10 = copy.deepcopy(Q)
    elif ep == 100:
        Q_100 = copy.deepcopy(Q)
    elif ep == 1000:
        Q_1000 = copy.deepcopy(Q)
    steps_since_switch = 0
    light_duration = 4
    state = (width-1, random.randint(0, height - 1), light)
    done = False
    total_reward = 0
    steps = 0
    red_violations = 0
    collisions = 0
    while not done:
        steps += 1
        # Balance exploration and exploitation
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])
        
        next_state = get_next_state(state, action, light)
        reward, terminal = get_reward(state, action, next_state)
        if (next_state[0], next_state[1]) in obstacles:
            collisions += 1
        if light == 0 and action != 4:
            red_violations += 1
        total_reward += reward

        # Stop value propagation from goal
        target = reward
        if not terminal:
            target += gamma * max(Q[next_state])

        Q[state][action] += alpha * (target - Q[state][action])
        
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
    epsilon = max(0.05, epsilon * decay_rate) # Decay epsilon

import pandas as pd

metrics = pd.DataFrame({
    "Episode": list(range(episodes)),
    "Reward": episode_rewards,
    "Steps": steps_to_goal,
    "RedLightViolations": red_light_violations,
    "ObstacleCollisions": obstacle_hits
})
metrics.to_csv("metrics.csv", index=False)