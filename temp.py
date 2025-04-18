import numpy as np
import random
import os, time, copy
import pygame

width, height = 50,50
goal = {(0 , i) for i in range(width)}
actions = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stay
obstacles = set()
for _ in range(max(width, 25)):
    i = random.randint(0, width - 1)
    j = random.randint(0, height - 1)
    pos = (i, j)
    if pos not in goal:
        obstacles.add(pos)
print("Goal positions:", goal)
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
        return -20, False # Hit obstacle
    
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

light = 0  # Start with red light
for ep in range(episodes):
    if ep == 10:
        Q_10 = copy.deepcopy(Q)
    elif ep == 100:
        Q_100 = copy.deepcopy(Q)
    elif ep == 1000:
        Q_1000 = copy.deepcopy(Q)
    elif ep == 50000:
        Q_50000 = copy.deepcopy(Q)
    steps_since_switch = 0
    light_duration = 4
    state = (width-1, random.randint(0, height - 1), light)
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

        # Stop value propagation from goal
        target = reward
        if not terminal:
            target += gamma * max(Q[next_state])

        Q[state][action] += alpha * (target - Q[state][action])
        #Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
        
        steps_since_switch += 1
        if steps_since_switch >= light_duration:
            light = 1 - light
            steps_since_switch = 0
        if terminal:
            done = True
        state = next_state
    epsilon = max(0.05, epsilon * decay_rate) # Decay epsilon

    episode_rewards.append(total_reward)
# print("Training Complete!")
# print("Q-Table:")
# for key, value in Q.items():
#      print(f"State: {key}, Q-Values: {value}")

def simulate_agent(Q, start=(width - 1,height // 2), max_steps=100, light_duration=4):
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
# simulate_agent(Q)

# import matplotlib.pyplot as plt

# plt.plot(episode_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Total Reward per Episode")
# plt.grid(True)
# plt.show()
def simulate_with_gui(Q_table, cell_size, light_duration, max_steps):
    pygame.init()
    screen = pygame.display.set_mode((width * cell_size, height * cell_size + 50))
    pygame.display.set_caption("Red Light Green Light Agent")

    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    x, y = width - 1, height // 2
    game_light = 0  # 0 = red, 1 = green
    steps_since_switch = 0
    robot_img = pygame.image.load("robot.png").convert_alpha()
    robot_img = pygame.transform.scale(robot_img, (cell_size, cell_size))
    doll_front = pygame.image.load("doll_look.png").convert_alpha()
    doll_back = pygame.image.load("doll_away.gif").convert_alpha()

    doll_front = pygame.transform.scale(doll_front, (cell_size * 2, cell_size * 2))
    doll_back = pygame.transform.scale(doll_back, (cell_size * 2, cell_size * 2))

    running = True
    step = 0
    while running and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Update Logic ---
        state = (x, y, game_light)
        if np.array(Q_table[state]).std() == 0:
            action = random.choice(Q_table[state])
        else:
            action = np.argmax(Q_table[state])
        next_x, next_y = x, y
        if action == 0: next_x = max(0, x - 1)       # Up
        elif action == 1: next_x = min(width - 1, x + 1)  # Down
        elif action == 2: next_y = max(0, y - 1)       # Left
        elif action == 3: next_y = min(height - 1, y + 1) # Right
        if game_light == 0 and action != 4:
            print("Moved during red light!")

        x, y = next_x, next_y
        step += 1

        if (x, y) in goal:
            print(f"🎉 Agent reached the goal in {step}!")
            running = False

        # Update traffic light
        steps_since_switch += 1
        if steps_since_switch >= light_duration:
            game_light = 1 - game_light
            steps_since_switch = 0

        # --- Drawing ---
        screen.fill((30, 30, 30))

        top_offset = cell_size * 2
        
        # Draw grid and goal
        for i in range(width):
            for j in range(height):
                rect = pygame.Rect(j * cell_size, i * cell_size + top_offset, cell_size, cell_size)
                if (i, j) in obstacles:
                    color = (0, 0, 255)
                elif (i, j) in goal:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

        # Draw agent
        screen.blit(robot_img, (y * cell_size, x * cell_size + top_offset))

        # Draw light indicator
        light_text = font.render("RED" if game_light == 0 else "GREEN", True, (255, 255, 255))
        screen.blit(light_text, (10, height * cell_size + 10))
        doll_image = doll_front if game_light == 0 else doll_back
        doll_x = (width * cell_size // 2) - (cell_size)  # Centered horizontally
        doll_y = cell_size // 2 # Just above the top row
        screen.blit(doll_image, (doll_x, doll_y))
        pygame.display.flip()
        clock.tick(15)  # Slower for visibility

    pygame.quit()


simulate_with_gui(Q_10, cell_size=25, light_duration=4, max_steps=10000)
simulate_with_gui(Q_100, cell_size=25, light_duration=4, max_steps=10000)
simulate_with_gui(Q_1000, cell_size=25, light_duration=4, max_steps=10000)
simulate_with_gui(Q, cell_size=25, light_duration=4, max_steps=10000)
