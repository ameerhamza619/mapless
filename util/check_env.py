from stable_baselines3 import A2C, PPO, DQN
from carenv import CarEnv


env = CarEnv()
env.reset()

for i in range(10):
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        print("Action: ", action)
        print("State: ", n_state)