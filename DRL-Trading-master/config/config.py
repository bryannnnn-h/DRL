import pathlib
import pandas as pd
import datetime
import os


now = datetime.datetime.now()
date_time = now.strftime("%Y_%m_%d_%H-%M")
TRAINED_MODEL_DIR = f"trained_models/{date_time}"

RESULT_DIR = "results"
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"

# Env
VALID_START = "2016-01-01"   # date starting validation 
VALID_END = "2020-12-01"   # date ending validation
OBS_DIM = 61   # dimension of state representation (select 10 of 0050 constituent stocks)

tic_set = [1301,1303,2303,2308,2317,2330,2454,2882,2891,3008]

#VIX
vix_mean=18.1
vix_max=82.7
trade_limit=1000