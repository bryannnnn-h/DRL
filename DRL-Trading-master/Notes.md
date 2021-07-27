# Notes
## Environment
### Turbulence
* The limitation added when validation and trading, not for training 
### Reward Function
1. When at state s, the asset value before action will be calculated
2. Take action a at state s
    * The shares for each stock in position and the balance will change
3. Calculate the asset value after action is taken
    * Use close price tomorrow to calculate the value for the new stock share allocations
    * Plus the balance to be the new asset value 

### Sharpe Ratio
* The time interval shouldn't be short, should include bearish and bullish trends
* The good strategy shouldn't yield sharpe ratios with too large fluctuation

### Recordings
* The iteration, iter_num and i are used to calculate the start index of validation and trading 
    * For example, if the first validation starts from "2016-01", then the i will be 63+63; hence, we know the training will end when date reaches the first date of validation. Namely, unique_trade_date[i-63-63] 

## Problems
### Action Normalization
### Agent Pool Ensemble Method
### Retrain the Model with Seen Data
* Adjust to loading old model and training on new data
### Annually 
* 252**2?



