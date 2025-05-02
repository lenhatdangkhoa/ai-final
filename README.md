

#  Red Light, Green Light Q-Learning

The game allows an agent to navigate though the Red Light, Green Game using Q-Learning

----
## Run the game


Clone the repistory
```bash
git clone https://github.com/lenhatdangkhoa/ai-final.git
```
Install the required packaged
```bash
cd ai-final
pip install -r requirements.txt
```
Run the application.
```bash
python run_game.py
```
**Game Configuration** \
You can change the following settings of the game for experiments: 
- Game Speed
- Environment grid size
- The GUI grid size
- Light duration
- Number of training episodes
On line 115, decrease this number increase the clock tick value will speed up the simulation and vice versa
```basg
clock_tick(number) 
```
On line 119, change the environment's size
```bash
ql = q_learning.Q_learning(25, 25) # 25 x 25 grid
```
On line 120, change the number of training episodes
```bash
Q = ql.run(10000) # 10,000 episodes
```
Lastly, on line 121, you can change the pixel size of each cell and light duration
```bash
simulate_with_gui(Q, ql,cell_size=20, light_duration=4, max_steps=200)
# This means each cell is 20 x 20 pixels
# Each red light is 4 seconds long
# The maximum iteration is 200 to prevent infinite loop
```

