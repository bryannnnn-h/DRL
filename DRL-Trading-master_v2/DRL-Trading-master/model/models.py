# common library
import pandas as pd
import numpy as np
from math import exp
import time
import gym

# debugging
import sys
import os

# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
'''
    Specify total_timesteps when model is learning
    total_timesteps: int, the total number of samples to train on
'''
def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_ACER(env_train, model_name, timesteps=25000):
    start = time.time()
    model = ACER('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_GAIL(env_train, model_name, timesteps=1000):
    """GAIL Model"""
    #from stable_baselines.gail import ExportDataset, generate_expert_traj
    start = time.time()
    # generate expert trajectories
    model = SAC('MLpPolicy', env_train, verbose=1)
    generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

    # Load dataset
    dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
    model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

    model.learn(total_timesteps=1000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

'''==================================================================================================================='''
def DRL_validation(model, val_data, val_env, val_obs) -> None:
    for i in range(len(val_data.index.unique())):
        action, _states = model.predict(val_obs)
        val_obs, rewards, dones, info = val_env.step(action)

def get_validation_sharpe(iteration):
    # 'results/account_value_validation_{}.csv' stored when terminal in function "step" of EnvMultiStock_validation.py 
    df_total_value = pd.read_csv('results/{}/account_value_validation_{}.csv'.format(config.RESULT_DIR, iteration))#, index_col=0)   
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    '''
        Seasonal? 63**0.5?
    '''
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe

def DRL_prediction(df,
                   model,
                   action_weights,
                   weight_base,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial,
                   vix):   # actual trading process
    
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,   # last state will be the first one in the next eps 
                                                   model_name=name,
                                                   iteration=iter_num,
                                                   vix=vix)])
    '''
        If not the very initial state, the init. state will be set to the previous state (the last state of the previous eps)
    '''
    obs_trade = env_trade.reset()   

    total_reward=0
    for i in range(len(trade_data.index.unique())):
        
#         actions = []   # actions taken by different agents in agent pool
#         action_weighted_sum = np.array([0 for _ in range(10)])
#         for agent, w in zip(model, action_weights):
#             action, _states = agent.predict(obs_trade,deterministic=True) ##set deterministic to true to test
#             weighted_action = action * w
#             action_weighted_sum = np.add(action_weighted_sum, weighted_action)
#         action = action_weighted_sum / weight_base]
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        
        total_reward=info[0]['total_reward']
        
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()   # already updated to ST (the init. state for the next eps)
    
   
    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/{}/last_state_{}_{}.csv'.format(config.RESULT_DIR, name, i), index=False)
    return last_state,total_reward  

def train_model(model_use,env_train,unique_trade_date,rebalance_window,validation_window,i,validation,env_val,obs_val):
    if model_use=='a2c': 
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_30k_0050_{}".format(i), timesteps=30000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ", unique_trade_date[i - rebalance_window], "=====")
        DRL_validation(model=model_a2c, val_data=validation, val_env=env_val, val_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        
        print("A2C Sharpe Ratio: ", sharpe_a2c)
        return  model_a2c,sharpe_a2c
        
    elif model_use=='ppo':
        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_0050_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ", unique_trade_date[i - rebalance_window], "=====")
        DRL_validation(model=model_ppo, val_data=validation, val_env=env_val, val_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)
        return  model_ppo,sharpe_ppo
    elif model_use=='ddpg':
        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_0050_{}".format(i), timesteps=10000)
        #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ", unique_trade_date[i - rebalance_window], "=====")
        DRL_validation(model=model_ddpg, val_data=validation, val_env=env_val, val_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)
        print("DDPG Sharpe Ratio: ", sharpe_ddpg)
        return  model_ddpg,sharpe_ddpg

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window,model_use,iter_number) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    '''
        df: pd.DataFrame, preprocessed dataframe
        unique_trade_date: array_like, the dates starting from the first date for validation, including val and actual trading
        rebalance_window: int, number of months to retrain the model
        validation_window: int, number of months to validation the model and select for trading
    '''
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    
    ##VIX use
    vix=pd.read_csv('data/VIX.csv')
    print('vix read')
    print(vix.head())
    
    last_state_ensemble = []

    #ppo_sharpe_list = []
    #ddpg_sharpe_list = []
    #a2c_sharpe_list = []
    sharpe_list=[]
    
    total_reward_list=[]
    
   
    
    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<"2016-01-00") & (df.datadate>="2009-00-00")]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    # use quantile as thres (e.g. 0.5 -> median)
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .75)#.90)
    

    start = time.time()
    
    '''
        self-defined episode counter to help observing
    '''
    eps_count = 0
     
    # i means training dates

    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        eps_count += 1 
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:   # the first eps
            # inital state
            initial = True
        else:   # other eps, should notice the problem of the previous state
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        
        
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        '''
            Retrain all the data evey eps?
            Why not jusyt tained on the new one with the nearest three months
        '''
        
        train = data_split(df, start="2009-01-01", end=unique_trade_date[i - rebalance_window - validation_window])
      
        vix_train=vix[(vix.datadate >= "2009-01-01") & (vix.datadate < unique_trade_date[i - rebalance_window - validation_window])]   
        vix_train.reset_index(inplace=True)
    
            
              
        env_train = DummyVecEnv([lambda: StockEnvTrain(train,vix=vix_train)])

        ## validation env
        
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        
        
        vix_valid=vix[(vix.datadate >= unique_trade_date[i - rebalance_window - validation_window]) & (vix.datadate <unique_trade_date[i - rebalance_window])]   
        vix_valid.reset_index(inplace=True)
        
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i,vix=vix_valid)])
        obs_val = env_val.reset()
        
        # Training and validation
        print("======Model training from: ", "2009-01-01", "to ",
              unique_trade_date[i - rebalance_window - validation_window], "=====")
        
        print("======Traing episode: " + str(eps_count) + "=======")
       
        model_ensemble,sharpe=train_model(model_use,env_train,unique_trade_date,rebalance_window,validation_window,i,validation,env_val,obs_val)
        sharpe_list.append(sharpe)
        
        
        
        # Real trading
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        vix_trade=vix[(vix.datadate >= unique_trade_date[i - rebalance_window]) & (vix.datadate < unique_trade_date[i])]   
        vix_trade.reset_index(inplace=True)
        
        last_state_ensemble,total_reward  = DRL_prediction(df=df, model=model_ensemble, action_weights=0, weight_base=0, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial,
                                             vix=vix_trade)
        total_reward_list.append(total_reward)
    end = time.time()
    df=pd.DataFrame()
    df['sharpe']=sharpe_list
    df['reward']= total_reward_list
    df.to_csv('./exp_data/'+str(iter_number)+'_'+model_use+'_sharpe_reward.csv',index=False)
    
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
