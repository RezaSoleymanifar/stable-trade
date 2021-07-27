import gym
from stable_baselines3 import A2C
from time import time
import logging

logging.basicConfig(filename='logging.log', level=logging.INFO)

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1, device="cuda")

t = time()
model.learn(total_timesteps=50000)
logging.debug(f'elapsed time: {time() - t}')

obs = env.reset()

t = time()
for i in range(1000):
    logging.info(i)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()