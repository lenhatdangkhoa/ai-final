import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("metrics.csv")
plt.figure(figsize=(10, 8))

# 1. Total Reward
plt.subplot(2, 2, 1)
plt.plot(df["Episode"], df["Reward"], label="Total Reward", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.legend()

# 2. Steps Taken
plt.subplot(2, 2, 2)
plt.plot(df["Episode"], df["Steps"], label="Steps Taken", color='orange', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps Taken per Episode")
plt.grid(True)
plt.legend()

# 3. Red Light Violations
plt.subplot(2, 2, 3)
plt.plot(df["Episode"], df["RedLightViolations"], label="Red Light Violations", color='red', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Violations")
plt.title("Red Light Violations per Episode")
plt.grid(True)
plt.legend()

# 4. Obstacle Collisions
plt.subplot(2, 2, 4)
plt.plot(df["Episode"], df["ObstacleCollisions"], label="Obstacle Collisions", color='blue', alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Collisions")
plt.title("Obstacle Collisions per Episode")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
