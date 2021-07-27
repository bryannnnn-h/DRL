import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from config import config

from math import exp
import os

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = 10
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
#TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4

# observation dimension
OBS_DIM = 61

class StockEnvValidation(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day = 0,vix=0, turbulence_threshold=140, iteration=''):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
      
        
        self.day = day
        self.df = df
        
            #add vix for use
        
        self.vix=vix
        
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1, shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (OBS_DIM,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        #self.reset()
        self._seed()
        
        self.iteration=iteration


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:   # within thres
            if self.state[index+STOCK_DIM+1] > 0:   # if share for single stock[stock index] is greater than 0
                self._sell_process(index, action, clear=False)
            else:
                pass
        else:   # over threshold, clear out all positions 
            if self.state[index+STOCK_DIM+1] > 0:
                self._sell_process(index, action, clear=True)
            else:
                pass
    
    def _sell_process(self, index, action, clear=False):   # conduct the selling action with clearing positions or not
        shares = 0   # shares to be sold
        if clear:   # clear all the positions
            shares = self.state[index+STOCK_DIM+1]   # shares right on hand (in positions)
        else:
            shares = min(abs(action), self.state[index+STOCK_DIM+1])   # min (shares specified by action, shares in positions)
        
        profit = self.state[index+1] * shares   # sell with the shares specified
        self.state[0] += profit * (1- TRANSACTION_FEE_PERCENT)   # update balance
        self.state[index+STOCK_DIM+1] -= shares   # reduce the sold shares; if clear, the result is 0
        self.cost += profit * TRANSACTION_FEE_PERCENT   # add the transaction cost   
        self.trades += 1
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]
            shares = min(available_amount, action)
            expense = self.state[index+1] * shares
            
            self.state[0] -= expense * (1 + TRANSACTION_FEE_PERCENT)
            self.state[index+STOCK_DIM+1] += shares
            self.cost += expense * TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:   # over threshold, just stop buying
            pass
        
    def get_vix_adj_value(self,vix):
        total_trade=self.trades
        vix_mean=config.vix_mean
        vix_max=config.vix_max
        trade_limit=config.trade_limit
        vix_fcator=exp((vix-vix_mean)/vix_max)
        adj_factor=2-(total_trade/trade_limit)*vix_fcator
        return adj_factor 
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        
        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/{}/account_value_validation_{}.png'.format(config.RESULT_DIR, self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/{}/account_value_validation_{}.csv'.format(config.RESULT_DIR, self.iteration), index=False)
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            return self.state, self.reward, self.terminal, {}
        else:
            '''
                Why to do this
            '''
            actions = actions * HMAX_NORMALIZE
            if self.turbulence >= self.turbulence_threshold:
                actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                        
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])
    
            # generate the next observation
            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'].values[0]
            self.state =  [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
            
            # <-- remove vix -->
            current_vix=self.vix.loc[self.day,:]  
            adj_factor = self.get_vix_adj_value(current_vix['Open'])
            adj_total_asset=end_total_asset*adj_factor
            adj_begin_total_asset=begin_total_asset*adj_factor
            self.reward =    adj_total_asset - adj_begin_total_asset  
            
            
            #self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist()  + \
                      self.data.cci.values.tolist()  + \
                      self.data.adx.values.tolist() 
            
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
