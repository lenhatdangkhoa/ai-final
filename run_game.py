import pygame
import numpy as np
import random
import q_learning

def simulate_with_gui(Q_table, ql, cell_size, light_duration, max_steps):
    pygame.init()
    screen = pygame.display.set_mode((ql.width* cell_size, ql.height * cell_size + 50))
    pygame.display.set_caption("Red Light Green Light Agent")

    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    x, y = ql.width - 1, ql.height // 2
    game_light = 0  # 0 = red, 1 = green
    steps_since_switch = 0
    robot_img = pygame.image.load("assets/robot.png").convert_alpha()
    robot_img = pygame.transform.scale(robot_img, (cell_size, cell_size))
    doll_front = pygame.image.load("assets/doll_look.png").convert_alpha()
    doll_back = pygame.image.load("assets/doll_away.gif").convert_alpha()

    doll_front = pygame.transform.scale(doll_front, (cell_size * 2, cell_size * 2 + 10))
    doll_back = pygame.transform.scale(doll_back, (cell_size * 2, cell_size * 2))

    running = True
    step = 0
    total_reward = 0
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
        reward = 0
        if (x, y) in ql.goal:
            reward = 50
        elif (x, y) in ql.obstacles:
            reward = -20
        elif game_light == 0 and action != 4:
            reward = -50
        elif action == 0 and game_light == 1:
            reward = 2
        else:
            reward = -2
        total_reward += reward
        if action == 0: next_x = max(0, x - 1)       # Up
        elif action == 1: next_x = min(ql.width - 1, x + 1)  # Down
        elif action == 2: next_y = max(0, y - 1)       # Left
        elif action == 3: next_y = min(ql.height - 1, y + 1) # Right

        if game_light == 0 and action != 4:
            print("Moved during red light!")

        if (next_x, next_y) in ql.obstacles and action == 0:
            print("Hit an obstacle!")
            

        x, y = next_x, next_y
        step += 1

        if (x, y) in ql.goal:
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
        for i in range(ql.width):
            for j in range(ql.height):
                rect = pygame.Rect(j * cell_size, i * cell_size + top_offset, cell_size, cell_size)
                if (i, j) in ql.obstacles:
                    color = (0, 0, 255)
                elif (i, j) in ql.goal:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(screen, color, rect)
                #pygame.draw.rect(screen, (0, 0, 0), rect, 1)

        # Draw agent
        screen.blit(robot_img, (y * cell_size, x * cell_size + top_offset))

        # Draw light indicator
        text_color = (0, 255, 0) if game_light == 1 else (255, 0, 0)
        light_text = font.render("RED" if game_light == 0 else "GREEN", True, text_color)
        screen.blit(light_text, (10, ql.height * cell_size + 10))
        reward_text = font.render(f"Reward: {total_reward}", True, (255, 255, 255))
        screen.blit(reward_text, (10, 0))
        doll_image = doll_front if game_light == 0 else doll_back
        doll_x = (ql.width * cell_size // 2) - (cell_size)  # Centered horizontally
        doll_y = cell_size // 2 # Just above the top row
        screen.blit(doll_image, (doll_x, doll_y))
        pygame.display.flip()

        clock.tick(20)  # Slower for visibility

    pygame.quit()

ql = q_learning.Q_learning(75, 75)
Q = ql.run(10000)
simulate_with_gui(Q, ql,cell_size=25, light_duration=4, max_steps=200)
# # print("Episode 10: " + str(episode_rewards[10]))
# simulate_with_gui(Q_100, cell_size=25, light_duration=4, max_steps=200)
# # print("Episode 100: " + str(episode_rewards[100]))
# simulate_with_gui(Q_1000, cell_size=25, light_duration=4, max_steps=200)
# # print("Episode 1000: " + str(episode_rewards[100]))
# simulate_with_gui(Q, cell_size=25, light_duration=4, max_steps=10000)




