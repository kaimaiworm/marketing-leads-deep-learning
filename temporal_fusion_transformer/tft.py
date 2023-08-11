import pathlib
import numpy as np
import pandas as pd
import optuna 
from tft_functions import NNTuner, final_prediction
from darts import TimeSeries
import pickle


# current working directory
path = pathlib.Path().absolute()


################
## Folders ##
################

data_input_folder         = "{}\\dataset\\output".format(path)
output_folder        = "{}\\temporal_fusion_transformer\\output".format(path)

####################
### Load datasets ###
#####################

### Data 
df_train = pd.read_excel("{}\\data.xlsx".format(data_input_folder), sheet_name = 0)
df_test = pd.read_excel("{}\\data.xlsx".format(data_input_folder), sheet_name = 1)

df_train = df_train.set_index("Date")
df_test = df_test.set_index("Date")

df_train_diff = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 0)
df_test_diff  = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 1)

df_train_diff  = df_train_diff.set_index("Date")
df_test_diff  = df_test_diff.set_index("Date")
      

#### Create dedicated target and feature sets
X_train = df_train.drop(["website", "manual"], axis = 1)
y_train = df_train[["website", "manual"]]

X_test = df_test.drop(["website", "manual"], axis = 1)
y_test = df_test[["website", "manual"]]

X_train_diff = df_train_diff.drop(["website", "manual"], axis = 1)
y_train_diff = df_train_diff[["website", "manual"]]

X_test_diff = df_test_diff.drop(["website", "manual"], axis = 1)
y_test_diff = df_test_diff[["website", "manual"]] 

## Create lagged targets from data for first difference re-transformation
y_train_lag= y_train.shift(1).drop(y_train.index[0])

y_test_lag = pd.concat([y_train.tail(1), y_test], axis = 0).shift(1)
y_test_lag.drop(y_test_lag.index[0], inplace = True)

y_test_diff_lag = pd.concat([y_train_diff.tail(1), y_test_diff], axis = 0).shift(1)
y_test_diff_lag.drop(y_test_diff_lag.index[0], inplace = True)



#######################################
##### Temporal Fusion Transformer #####
#######################################

###### Create Time Series Dataframe for DARTS
y_train_ts = {}
y_test_ts = {}

for var in ["manual", "website"]:
    y_train_ts[var] = TimeSeries.from_series(y_train_diff[var])
    y_test_ts[var] = TimeSeries.from_series(y_test_diff[var])
    

X_train_ts = X_train_diff.copy()
X_test_ts = X_test_diff.copy()


for df in [X_train_ts, X_test_ts]:
    #df.drop(columns = ["weekday_Monday", "weekday_Tuesday","weekday_Wednesday","weekday_Thursday","weekday_Friday","weekday_Saturday", "weekday_Sunday"], inplace = True) #drop day of week dummies since model can handle categories 
    df["weekday"] = pd.Series(df.index, name = "weekday", index = df.index).dt.day #create day of week variable
    df["time_idx"] = np.arange(0, len(df))    #create time index
    
X_train_ts = TimeSeries.from_dataframe(X_train_ts)
X_test_ts = TimeSeries.from_dataframe(X_test_ts)

######################
###### NN Tuning #####
######################

"""
IMPORTANT: Hyperparameter tuning of neural networks is computationally expensive and was too much for our old laptops to handle when the number of
           optuna trials and NN epochs was greater than 2 respectively. Therefore, the results in the submission paper are achieved without parameter tuning
           but fixed hyperparameters instead, based on proposed defaults in the original paper or values that are based on the dataset.
           When uncommenting the next part, the tuning routine starts for each lead_type and is expected to enhance the results of the baseline TFT model
           but this is not recommended since it takes ages
"""


"""
###init study and params dict
best_params_tft = {}
study_tft = {}
n_trials = 100  #number of trials in optuna; the higher, the longer tuning takes but performance of model increases (should be 50 at least)

for var in ["website", "manual"]:
    # optimize hyperparameters by minimizing the RMSE on the validation set
    np.random.seed(42) #reproducability
    tuner = NNTuner(X_train_ts, y_train_ts[var]) # init Tuning class
    study_tft[var] = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=np.random.seed(42)))                                 
    study_tft[var].optimize(tuner.objective, n_trials=n_trials) # start optimization
    best_params_tft[var] = study_tft[var].best_params #store results

##Safe best params
with open("{}preds_tft.pkl".format(output_folder), "wb") as fp:
   pickle.dump(best_params_tft, fp)

"""


################################
#### Prediction on test set ####
################################

##### Make final predictions with optimized hyperparameters    
#init dicts
tft = {}
preds_tft = {}    
rmse_tft = {}
n_epochs = 300 # number of epochs for the NN model, high # of epochs used since early stopping is implemented 

for var in ["website", "manual"]:
    tft[var] = final_prediction(X_train_ts, X_test_ts, y_train_ts[var], y_test_ts[var]) # init final_prediction class for data transformation
    preds_tft[var], rmse_tft[var] = tft[var].predict(n_epochs = n_epochs) # make predictions #comment this out to use optimized parameters
    
    # uncomment line below and comment the above line out in order to use optimized parameters from hyperparameter training
    #preds_tft[var], rmse_tft[var] = tft.predict(n_epochs = n_epochs, best_params = best_params_tft[var]) 
                
##Safe predictions and rmse
with open("{}preds_tft.pkl".format(output_folder), "wb") as fp:
   pickle.dump(preds_tft, fp)   
 
with open("{}rmse_tft.pkl".format(output_folder), "wb") as fp:
   pickle.dump(rmse_tft, fp)     