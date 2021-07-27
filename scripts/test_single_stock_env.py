from envs.single_stock.env import StockTradingEnv
import pandas as pd
import logging

df = pd.read_csv('../data/raw/AAPL.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = StockTradingEnv(df)

obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done , info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
      print('*'*20)
      logging.warning('agent net worth = 0')