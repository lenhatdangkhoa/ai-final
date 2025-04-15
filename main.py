import numpy as np
import random

width, height = 5, 5  # 5x5 grid
goal = (4, 4)
actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stay

Q = {}

# Initialize Q-Table
for x in range(width):
    for y in range(height):
        for light in [0, 1]:
            Q[(x, y, light)] = [0, 0, 0, 0, 0]

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 10000

def get_next_state(state, action):
    x, y, light = state
    next_light = random.choice([0, 1])
    
    if action == 0:  # Up
        x = max(0, x - 1)
    elif action == 1:  # Down
        x = min(width-1, x + 1)
    elif action == 2:  # Left
        y = max(0, y - 1)
    elif action == 3:  # Right
        y = min(height-1, y + 1)
    # Stay does not change position
    return (x, y, next_light)

def get_reward(state, action, next_state):
    x, y, light = state
    next_x, next_y, _ = next_state

    if (x, y) == (next_x, next_y) and action != 4:
        return -1  # Hit wall
    
    if light == 0 and action != 4:
        return -10  # Moved on Red
    
    if (next_x, next_y) == goal:
        return 50  # Goal reached
    
    if action != 4 and light == 1:
        return 0.1  # Move on green
    
    return -0.1  # Time penalty

# Q-Learning
for ep in range(episodes):
    state = (0, 0, random.choice([0, 1]))
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])
        
        next_state = get_next_state(state, action)
        reward = get_reward(state, action, next_state)
        
        Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
        
        if (next_state[0], next_state[1]) == goal:
            done = True
        
        state = next_state

print("Training Complete!")
print("Q-Table:")
for key, value in Q.items():
    print(f"State: {key}, Q-Values: {value}")