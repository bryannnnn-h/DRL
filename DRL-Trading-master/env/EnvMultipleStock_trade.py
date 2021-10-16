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

class StockEnvTrade(gym.Env):
    """
        A stock trading environment for OpenAI gym
        Inherited from gym.Env (defined in openai gym core.py) 
    """
     # set render mode
    '''
        Render to the current display or terminal and return nothing. 
        Usually for human consumption.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day = 0, turbulence_threshold=140
                 ,initial=True, previous_state=[], model_name='', iteration='',vix=0, iter_total=''):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        
        self.vix=vix
        
        self.initial = initial
        self.previous_state = previous_state
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (OBS_DIM,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.data.loc[~self.data["tic"].isin(config.tic_set), ["adjcp", "open", \
        "high", "low", "volume", "ajexdi", "macd", "rsi", "cci", "adx", "turbulence"]] = 0
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
        '''====================='''
        # record the action taken in each step (shares), and also the fill price (adj close here) -> for pyfolio "transaction" tear sheet
        self.action_for_one_step = []   # actual action taken in one step (take clearing or limitation into consideration)
        self.action_records = []   # action taken for each stock in each step
        self.fill_prices = []   # fill (executive) price for each stock in each step
        
        # record the allocation (each stock value + balance) in each step -> for pyfolio "position" tear sheet
        self.allocations = []   # initial allocation 
        '''====================='''
        #self.reset()
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration
        self.iter_total=iter_total


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
        
        '''==========='''
        # record the shares ordered for one stock
        self.action_for_one_step[index] = shares   
        '''==========='''
        profit = self.state[index+1] * shares   # sell with the shares specified
        self.state[0] += profit * (1- TRANSACTION_FEE_PERCENT)   # update balance
        self.state[index+STOCK_DIM+1] -= shares   # reduce the sold shares; if clear, the result is 0
        self.cost += profit * TRANSACTION_FEE_PERCENT   # add the transaction cost   
        self.trades += 1
            
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]     
            shares = min(available_amount, action)   # min (shares available constrained by balance, shares specified by action)
            
            '''==========='''
            # record the shares ordered for one stock
            self.action_for_one_step[index] = shares   
            '''==========='''
            expense = self.state[index+1] * shares    # buy with the shares of min (shares available constrained by balance, shares specified by action)
            self.state[0] -= expense * (1 + TRANSACTION_FEE_PERCENT)   # update balance
            self.state[index+STOCK_DIM+1] += shares
            self.cost += expense * TRANSACTION_FEE_PERCENT   # add the transaction cost
            self.trades+=1
        else:   # over threshold, just stop buying
            pass
        #get the vix adj factor
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
        
        if self.terminal:   # eps ends
            '''
                The last action will not be executed in the last step.
                But, the terminal state has a
                lready been recorded for the next eps.
            '''
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/{}/account_value_trade_{}_{}.png'.format(config.RESULT_DIR, self.model_name, self.iteration))
            plt.close()
            
            '''
                Fill epsisode gap
                    1. Record empty action with fill price
                    2. Record allocation (stock position will change due to adj close but balance not)
            '''
            '''======================'''
            self.action_for_one_step = [0 for _ in range(STOCK_DIM)]
            self._record_indicators()
            
            df_action_record = pd.DataFrame(self.action_records)
            df_action_record.to_csv("results/{}/action_record_trade_{}_{}.csv".format(config.RESULT_DIR, self.model_name, self.iteration), index=False)
            df_fill_prices = pd.DataFrame(self.fill_prices)
            df_fill_prices.to_csv("results/{}/fill_prices_trade_{}_{}.csv".format(config.RESULT_DIR, self.model_name, self.iteration), index=False)
            df_allocations = pd.DataFrame(self.allocations)#, coolumns = [])
            df_allocations.to_csv("results/{}/allocations_trade_{}_{}.csv".format(config.RESULT_DIR, self.model_name, self.iteration), index=False)
            '''======================'''
            
            
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/{}/account_value_trade_{}_{}.csv'.format(config.RESULT_DIR, self.model_name, self.iteration), index=False)
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))   # balance + stock values
            
            total_reward=self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))- self.asset_memory[0]
            
            print("previous_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))- self.asset_memory[0] ))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            df_total_value_std = df_total_value['daily_return'].std()
            tic_set = config.tic_set
            tmp = []
            val_tmp = self.df.groupby("tic")
            for tic in tic_set:
                val = val_tmp.get_group(tic)
                open_list = list(val["open"])
                close_list = list(val["adjcp"])
                tmp.append((close_list[-1] - open_list[0])/open_list[0])
            tmp_mean = np.mean(tmp)
            #sharpe = (4**0.5)*df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            sharpe = (252 ** 0.5) * (df_total_value['daily_return'].mean() - 0.05/365) / df_total_value_std
            print("Sharpe: ",sharpe)
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/{}/account_rewards_trade_{}_{}.csv'.format(config.RESULT_DIR, self.model_name, self.iteration) , index=False)
            df_total = pd.DataFrame()
            df_total['previous_asset'] = [self.asset_memory[0]]
            df_total['end_total_asset'] = [end_total_asset]
            df_total['total_reward'] = [self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))- self.asset_memory[0]]
            df_total['total_cost'] = [self.cost]
            df_total['total_trades'] = [self.trades]
            df_total['sharpe'] = [sharpe]
            #print("Df_total: ",df_total)
            df_total.to_csv('result_record/result_{}.csv'.format(self.iter_total), mode='a', header=False, index=False)
            return self.state, self.reward, self.terminal, {'total_reward':total_reward}

        else:
            '''
                Why to do this?
                A:
                    Action space belongs to [-1, 1]
                    Multiply action with 100 will yield the actual shares 
                    The max is 100 shares and the min is -100
            '''
            actions = actions * HMAX_NORMALIZE   # action norm.
            #actions = (actions.astype(int))
            
            '''=====================necessary?======================='''
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
            '''============================================'''
                            
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))  
            
            '''============================================'''
            self.action_for_one_step = [0 for _ in range(STOCK_DIM)]
            '''============================================'''
            if config.operate_mode == 1:
                if actions[0] * actions[1] > 0:
                    actions[1] = actions[1]*(-1)
            argsort_actions = np.argsort(actions)   # sort from small to large
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]    # reverse the np arrray first

            for index in sell_index:
                if index not in config.tic_set_index:
                    continue
                self._sell_stock(index, actions[index])

            for index in buy_index:
                if index not in config.tic_set_index:
                    continue
                self._buy_stock(index, actions[index])
                
            '''============================================'''
            self._record_indicators()
            '''============================================'''
            
            # gen next obs for agent
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.data.loc[~self.data["tic"].isin(config.tic_set), ["adjcp", "open", \
            "high", "low", "volume", "ajexdi", "macd", "rsi", "cci", "adx", "turbulence"]] = 0        
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
            
            
            #<-- remove vix -->
            current_vix=self.vix.loc[self.day,:]
            adj_factor = self.get_vix_adj_value(current_vix['Open'])
            adj_total_asset=end_total_asset*adj_factor
            adj_begin_total_asset=begin_total_asset*adj_factor
            self.reward =    adj_total_asset - adj_begin_total_asset  
            
#             self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*REWARD_SCALING

        return self.state, self.reward, self.terminal, {'total_reward':0}
    
    def _record_indicators(self):   # record some indicators used to generate tear sheets
        self.action_records.append(self.action_for_one_step)
        self.fill_prices.append(self.data.adjcp.values.tolist())
        stock_positions = (np.multiply(np.array(self.state[1:(STOCK_DIM+1)]), self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])).tolist()
        self.allocations.append(stock_positions + [self.state[0]])   # record allocation (stock position + balance)
        
    def reset(self):  
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        if self.initial:   # if it's the first eps
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]   # init. to pre-defined account balance value
            # initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()  + \
                          self.data.cci.values.tolist()  + \
                          self.data.adx.values.tolist() 
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.previous_state[1:(STOCK_DIM+1)])*np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory = [previous_total_asset]   # init. to previous account balance
            # the shares dimension will be initialized to previous shares in position
            self.state = [ self.previous_state[0]] + \
                          self.data.adjcp.values.tolist() + \
                          self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]+ \
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
