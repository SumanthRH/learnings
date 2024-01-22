"""Code from chapter 3 of Learning Ray"""

import random
import os
import numpy as np

class Discrete:
    def __init__(self, num_actions: int):
        """ Discrete action space for num_actions.
            Discrete(4) can be used as encoding moving in
            one of the cardinal directions.
        """
        self.n = num_actions 
    
    def sample(self):
        return random.randint(0, self.n - 1)
    

class Environment:
    def __init__(self, *args, **kwargs):
        self.seeker, self.goal = (0, 0), (4, 4)
        self.info = {'seeker': self.seeker, 'goal': self.goal}
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5*5)

    def reset(self):
        self.seeker = (0,0)
        return self.get_observation()

    def get_observation(self):
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        return 1 if self.seeker == self.goal else 0
    
    def is_done(self):
        return self.seeker == self.goal
    
    def step(self, action):
        row, col = self.seeker
        if action == 0:
            self.seeker = (min(row+1, 4), col)
        elif action == 1:
            self.seeker = (row, max(col-1, 0))
        elif action == 2:
            self.seeker = (max(row-1, 0), col)
        elif action == 3:
            self.seeker = (row, min(col+1, 4))
        else:
            raise ValueError("INvalid action")

        obs = self.get_observation()
        rew = self.get_reward()
        is_done = self.is_done()
        return obs, rew, is_done, self.info    

    def render(self, *args, **kwargs):
        os.system('cls' if os.name == "nt" else "clear")

        grid = [["|" for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] += "G"
        grid[self.seeker[0]][self.seeker[1]] += "S"
        print("".join(["".join(grid_row) for grid_row in grid]))

class Policy:
    def __init__(self, env):
        """A Policy suggests actions based on the current state.
        We do this by tracking the value of each state-action pair. """
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n) ]
        self.action_space = env.action_space
        
    def get_action(self, state, explore=True, epsilon=0.1):
        """Explore randomly or exploit the best value currently available.""" 
        if explore and random.uniform(0, 1) < epsilon:
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])

if __name__ == "__main__":
    import time
    environment = Environment()
    random_pos = environment.observation_space.sample()
    environment.seeker = (random_pos // 5, random_pos % 5)
    while not environment.is_done():
        random_action = environment.action_space.sample()
        environment.step(random_action)
        time.sleep(0.1)
        environment.render()
        
