import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces



import os
from math import exp

from gym.utils import closer
env_closer = closer.Closer()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from config import config

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100   # share upper bound for single stock
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# total number of stocks in our portfolio
STOCK_DIM = 10
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4
# observation dimension
OBS_DIM = 61

# class "StockEnvTrain" definition
class StockEnvTrain(gym.Env):
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

    def __init__(self, df,vix=0,day=0):   # initializer
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        #add vix for use
        
        self.vix=vix
        
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (OBS_DIM,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False             
        # initalize state
        '''
            state = balance (1) + (adj close + share per stock + technical indicator * 4) (all with 30)
        '''
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]   # init. with holding initial asset (balance)
        self.rewards_memory = []
        self.trades = 0
        #self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+STOCK_DIM+1] > 0:   # if share for single stock[stock index] is greater than 0
            profit = self.state[index+1] * min(abs(action), self.state[index+STOCK_DIM+1])   # sell with the shares of min (shares specified by action, shares in positions)
            self.state[0] += profit * (1- TRANSACTION_FEE_PERCENT)   # update balance
            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])   # reduce the sold shares
            self.cost += profit * TRANSACTION_FEE_PERCENT   # add the transaction cost   
            self.trades+=1
        else:
            '''
                It doesn't support short now. It can be modified here to add in the short functionality.
            '''
            pass

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        # print('available_amount:{}'.format(available_amount))
        
        expense = self.state[index+1] * min(available_amount, action)    # buy with the shares of min (shares available constrained by balance, shares specified by action)
        self.state[0] -= expense * (1 + TRANSACTION_FEE_PERCENT)   # update balance
        self.state[index+STOCK_DIM+1] += min(available_amount, action)
        self.cost += expense * TRANSACTION_FEE_PERCENT   # add the transaction cost
        self.trades+=1
    
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
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1   # termination if reaching the end of training set
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/{}/account_value_train.png'.format(config.RESULT_DIR))
            plt.close()
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))   # balance + final stock position values
            
            #print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/{}/account_value_train.csv'.format(config.RESULT_DIR), index=False)
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            #print("total_cost: ", self.cost)
            #print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)   # percentage change of total asset values
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            #print("Sharpe: ",sharpe)
            #print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)   # reward records
            #df_rewards.to_csv('results/account_rewards_train.csv')
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal, {}   # return (s', r, T, info)

        else:
            # print(np.array(self.state[1:29]))

            '''
                Why to do this?
            '''
            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            
            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
#             print("begin_total_asset:{}".format(begin_total_asset))            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            '''
                Increase the day forward and work on the next state
            '''
            
            self.day += 1
            self.data = self.df.loc[self.day,:]
            
           
          
            
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
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
            #print("end_total_asset:{}".format(end_total_asset))
            
             #Check current vix value
                
             
            # <-- remove vix -->
            current_vix=self.vix.loc[self.day,:]
            adj_factor = self.get_vix_adj_value(current_vix['Open'])
            adj_total_asset=end_total_asset*adj_factor
            adj_begin_total_asset=begin_total_asset*adj_factor
            
#             self.reward = end_total_asset - begin_total_asset   # reward is defined as the change of total asset before and after the action taken 
            self.reward =    adj_total_asset - adj_begin_total_asset  
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward * REWARD_SCALING   # scale the reward to the proper size
            
        return self.state, self.reward, self.terminal, {}   # return (s', r, T, info)

    def reset(self):   # reset the env. and return the init.state
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() 
        # iteration += 1 
        return self.state
    
    def render(self, mode='human'):   # render the env. (support different mode)
        return self.state

    def _seed(self, seed=None):   # set the seed for this env's random number generator(s)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
