import random
import gym
from gym import spaces
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from .config import *
from matplotlib.animation import FuncAnimation

METRICS = ['profit', 'shares_held', 'time', 'current_price', 'transaction']
NO_PLOT = ['time']

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, hold_radius =1 / 6, visualize = True):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.hold_pad = hold_radius
        self.visualize = visualize

        # good practice to use [-1, 1] for action space.
        self.action_space = spaces.Box(low=-1, high=1, shape = (1,),  dtype=np.float16)


        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

        self.metrics_history = defaultdict(list)
        self.action = None
        self.current_price = None
        self.balance = None
        self.shares_held = None
        self.total_shares_sold = None
        self.total_sales_value = None
        self.net_worth = None
        self.profit = None
        self.quantity = None
        self.max_net_worth = None
        self.transaction = None

    def update_metrics_history(self):
        for metric in METRICS:
            self.metrics_history[metric].append(getattr(self, metric))
        return None

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def read_action(self, action):
        # we hold if action is in [-hold_span, hold_span]
        value = float(action)
        # unsqueezing the [hold_span, 1] interval to (0, 1) interval.
        percentage = max((abs(value) - self.hold_pad) / (1 - self.hold_pad), 0)

        if value  > self.hold_pad:
            action_type = 'Buy'
        elif value < -self.hold_pad:
            action_type = 'Sell'
        else:
            action_type = 'Hold'

        return action_type, percentage

    def _take_action(self, action):
        self.action = action

        # Set the current price to a random price within the time step
        self.current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        self.time = self.df.loc[self.current_step + 5, 'Date']

        action_type, percentage = self.read_action(action)
        if action_type == 'Buy':
            # Buy amount % of balance in shares
            total_possible = int(self.balance / self.current_price)
            shares_bought = int(total_possible * percentage)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            self.quantity = shares_bought

        elif action_type == 'Sell':
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * percentage)
            self.balance += shares_sold * self.current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * self.current_price
            self.quantity = shares_sold

        else:
            self.quantity = 0

        self.net_worth = self.balance + self.shares_held * self.current_price
        self.profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        self.transaction = action_type, self.time

        self.update_metrics_history()

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        # Sell amount % of shares held
        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        reward = self.profit

        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.profit = 0
        self.metrics_history.clear()

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        self.metrics_history.clear()

        return self._next_observation()

    def visualize_metrics(self):
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        colors = cm.rainbow(np.linspace(0, 1, len(self.metrics_history)))

        for metric, color in zip(METRICS, colors):
            metric_history = self.metrics_history[metric]

            if metric == 'transaction':
                price_history = self.metrics_history['current_price']
                x_buy = [time for action_type, time in metric_history if action_type =='Buy']
                y_buy = [price for (action_type, time),
                                   price in zip(metric_history, price_history) if action_type == 'Buy']

                x_sell = [time for action_type, time in metric_history if action_type =='Sell']
                y_sell = [price for (action_type, time), price
                          in zip(metric_history, price_history) if action_type == 'Sell']

                s = list(40 * np.linspace(0, 1, len(x_buy)))
                plt.scatter(x_buy, y_buy, color = 'green', label = 'Buy', s = s)

                s = list(20 * np.linspace(0, 1, len(x_sell)))
                plt.scatter(x_sell, y_sell, color = 'red', label = 'Sell', s = s)

            elif not metric in NO_PLOT:
                x = self.metrics_history['time']
                y = metric_history
                plt.plot(x, y, color=color, label = metric)
                plt.text(x[-1], y[-1], f'{y[-1]:.0f}', fontdict=None)

        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.ion()
        plt.draw()
        plt.pause(1e-5)
        plt.clf()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if self.visualize:
            self.visualize_metrics()

        print(60*'-')
        print(f'Step: {self.current_step}, Time: {self.time}')
        print(60*'-')

        action_type, percentage = self.read_action(self.action)
        print(f'Action: {action_type},'
              f' Quantity: {self.quantity},'
              f' Ratio: {percentage * 100:.2f}%,'
              f' Price: ${self.current_price:.2f}')
        print(f'Balance: ${self.balance:.2f}')
        print(
            f'Shares held: {self.shares_held}'
            f' (Total sold: {self.total_shares_sold})')
        print(
            f'Net worth: ${self.net_worth:.2f}'
            f' (Peak: ${self.max_net_worth:.2f})')
        print(f'Total profit: ${self.profit:.2f}')
