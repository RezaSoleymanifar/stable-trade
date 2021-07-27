from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from envs.single_stock.env import StockTradingEnv

import pandas as pd

df = pd.read_csv('../data/raw/AAPL.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
