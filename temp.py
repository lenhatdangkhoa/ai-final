import numpy as np
import random
import os, time
import pygame

# pygame.mixer.init()
# pygame.mixer.music.load("rlgl.mp3")

width, height = 30,30  
goal = {(0 , i) for i in range(width)}
actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stay

Q = {}
# Initialize Q-Table
for x in range(width):
    for y in range(height):
        for light in [0, 1]:
            Q[(x, y, light)] = [0, 0, 0, 0, 0]

alpha = 0.2
gamma = 0.9
epsilon = 1.0
decay_rate = 0.999
episodes = 30000

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
    # Stay does not change position
    return (x, y, light)

def get_reward(state, action, next_state):
    x, y, light = state
    next_x, next_y, _ = next_state

    if (x, y) == (next_x, next_y) and action != 4:
        return -2, False  # Hit wall
    
    if light == 0:  # Red light
        if action != 4:
            return -50, True  # Tried to move
        else:
            return 2, False  # Stayed on red – neutral reward
    
    if (next_x, next_y) in goal:
        return 50,True  # Goal reached
    
    if action == 0 and light == 1:
        return 2,False  # Move up on green
    
    return -2 , False  # Time penalty

def print_grid(x, y):
    for i in range(width):
        row = ""
        for j in range(height):
            if (i, j) == (x, y):
                row += "🟩"  # Agent
            elif (i, j) in goal:
                row += "🟥"  # Goal
            else:
                row += "◻️ "  # Empty
        print(row)
    print("\n")

episode_rewards = []
#Q-Learning

light = 0

for ep in range(episodes):
    steps_since_switch = 0
    light_duration = 4
    state = (width-1, random.randint(0, height - 1), light) # Start at (0, 0) with random light
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])
        
        next_state = get_next_state(state, action, light)
        reward, terminal = get_reward(state, action, next_state)
        total_reward += reward

        # Stop value propagation from illegal red moves
        target = reward
        if not terminal:
            target += gamma * max(Q[next_state])

        Q[state][action] += alpha * (target - Q[state][action])
        #Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
        
        steps_since_switch += 1
        if steps_since_switch >= light_duration:
            light = 1 - light
            steps_since_switch = 0
        #if (next_state[0], next_state[1]) in goal:
        if terminal:
            done = True
        
        state = next_state
    
    epsilon = max(0.05, epsilon * decay_rate)


        
    episode_rewards.append(total_reward)
print("Training Complete!")
print("Q-Table:")
for key, value in Q.items():
    print(f"State: {key}, Q-Values: {value}")

def simulate_agent(Q, start=(width - 1,height - 1), max_steps=100, light_duration=4):
    x, y = start
    light = 0  # Start on green
    steps_since_switch = 0

    print("Starting agent simulation...\n")
    # if not light:
    #     pygame.mixer.music.play()  # Loop the music
    for step in range(max_steps):
        # Print grid
        print_grid(x, y)
        print(f"Step {step+1} | Light: {'🟢 Green' if light else '🔴 Red'}")
        
        time.sleep(0.3)
        print(f"Current Position: ({x}, {y})")
        print(f"Current Light: {'Green' if light else 'Red'}")
        print(f"Current Q-Values: {Q[(x, y, light)]}")
        print(f"Current Action: {actions[np.argmax(Q[(x, y, light)])]}")
        state = (x, y, light)
        action = np.argmax(Q[state])

        # Apply action
        next_x, next_y = x, y
        if action == 0: next_x = max(0, x - 1)       # Up
        elif action == 1: next_x = min(width - 1, x + 1)  # Down
        elif action == 2: next_y = max(0, y - 1)       # Left
        elif action == 3: next_y = min(height - 1, y + 1) # Right
        # action == 4 → stay

        # Prevent illegal move during red light
        # if light == 0 and action != 4:
        #     next_x, next_y = x, y  # Cancel move

        x, y = next_x, next_y

        # Check for goal
        if (x, y) in goal:
            print_grid(x, y)
            print(f"🎉 Agent reached goal in {step+1} steps!")
            return

        # Switch light every few steps
        steps_since_switch += 1
        if steps_since_switch >= light_duration:
            light = 1 - light
            steps_since_switch = 0

    print("❌ Agent did not reach the goal within max steps.")

# Run the agent after training
simulate_agent(Q)

import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.show()