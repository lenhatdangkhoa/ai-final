import numpy as np
import random
import os, time
import pygame
import matplotlib.pyplot as plt

# pygame.mixer.init()
# pygame.mixer.music.load("rlgl.mp3")
# pygame.mixer.music.play(-1)

width, height = 30, 30  # Grid dimensions
goal = {(0, i) for i in range(width)}  # Top row as goal

# 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
# 5: Top-left, 6: Top-right, 7: Bottom-left, 8: Bottom-right
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

Q = {}
for x in range(width):
    for y in range(height):
        for light in [0, 1]:
            Q[(x, y, light)] = [0 for _ in actions]

alpha = 0.2
gamma = 0.9
epsilon = 0.1
episodes = 50000

def get_next_state(state, action):
    x, y, light = state
    next_light = random.choice([0, 1])  # Red (0) or Green (1)

    # Movement logic
    dx, dy = 0, 0
    if action == 0: dx = -1
    elif action == 1: dx = 1
    elif action == 2: dy = -1
    elif action == 3: dy = 1
    elif action == 5: dx, dy = -1, -1
    elif action == 6: dx, dy = -1, 1
    elif action == 7: dx, dy = 1, -1
    elif action == 8: dx, dy = 1, 1
    # 4 is stay

    nx, ny = max(0, min(width - 1, x + dx)), max(0, min(height - 1, y + dy))
    return (nx, ny, next_light)

def get_reward(state, action, next_state):
    x, y, light = state
    nx, ny, _ = next_state

    if (x, y) == (nx, ny) and action != 4:
        return -1, False

    if light == 0 and action != 4:
        return -20, True  # Moved on red

    if (nx, ny) in goal:
        return 100, True

    return -0.1, False  # Time penalty

def print_grid(x, y):
    for i in range(width):
        row = ""
        for j in range(height):
            if (i, j) == (x, y):
                row += "🚶"
            elif (i, j) in goal:
                row += "🏁"
            else:
                row += "◻️ "  # Empty
        print(row)
    print("\n")

episode_rewards = []
for ep in range(episodes):
    state = (width - 1, random.randint(0, height - 1), random.choice([0, 1]))
    total_reward = 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        next_state = get_next_state(state, action)
        reward, terminal = get_reward(state, action, next_state)
        total_reward += reward

        target = reward
        if not terminal:
            target += gamma * max(Q[next_state])

        Q[state][action] += alpha * (target - Q[state][action])
        state = next_state
        done = terminal

    episode_rewards.append(total_reward)

print("Training Complete!")

def simulate_agent(Q, start=(width - 1, height - 1), max_steps=100, light_duration=3):
    x, y = start
    light = 1
    steps_since_switch = 0

    print("Simulation begins...\n")
    for step in range(max_steps):
        print_grid(x, y)
        print(f"Step {step+1} | Light: {'🟢 Green' if light else '🔴 Red'}")
        time.sleep(0.3)
        print(f"Current Position: ({x}, {y})")
        print(f"Current Light: {'Green' if light else 'Red'}")
        print(f"Current Q-Values: {Q[(x, y, light)]}")
        print(f"Current Action: {actions[np.argmax(Q[(x, y, light)])]}")
        state = (x, y, light)
        action = np.argmax(Q[state])

        dx, dy = 0, 0
        if action == 0: dx = -1
        elif action == 1: dx = 1
        elif action == 2: dy = -1
        elif action == 3: dy = 1
        elif action == 5: dx, dy = -1, -1
        elif action == 6: dx, dy = -1, 1
        elif action == 7: dx, dy = 1, -1
        elif action == 8: dx, dy = 1, 1

        nx = max(0, min(width - 1, x + dx))
        ny = max(0, min(height - 1, y + dy))
        if light == 1 or action == 4:
            x, y = nx, ny

        if (x, y) in goal:
            print_grid(x, y)
            print(f"🎉 Reached goal in {step+1} steps!")
            return

        steps_since_switch += 1
        if steps_since_switch >= light_duration:
            light = 1 - light
            steps_since_switch = 0

    print("❌ Goal not reached in max steps.")
    
simulate_agent(Q)

# plt.plot(episode_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Q-learning Progress")
# plt.grid(True)
# plt.show()