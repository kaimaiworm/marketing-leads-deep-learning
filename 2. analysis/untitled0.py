import pathlib
import numpy as np
import pandas as pd
import optuna 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from hyperparameter_ml import objective, tuning, stack_tuning
from ml_models import crossval, train_oof_predictions, model_selector, create_meta_dataset, stack_prediction
from neural_network import NNTuner, final_prediction
from darts import TimeSeries
import pickle
import copy

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
pio.renderers.default="browser"
import plotly.express as px


# current working directory
path = pathlib.Path().absolute()


################
## Folders ##
################

input_folder         = "{}\\input".format(path)
output_folder        = "{}\\output".format(path)


#####################
### Load datasets ###
#####################

### Data 
df_train = pd.read_excel("{}\\data.xlsx".format(input_folder), sheet_name = 0)
df_test = pd.read_excel("{}\\data.xlsx".format(input_folder), sheet_name = 1)

df_train = df_train.set_index("Date")
df_test = df_test.set_index("Date")

df_train_diff = pd.read_excel("{}\\data_diff.xlsx".format(input_folder), sheet_name = 0)
df_test_diff  = pd.read_excel("{}\\data_diff.xlsx".format(input_folder), sheet_name = 1)

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


##### Standardize data ####

cols = np.array(X_train_diff.columns).tolist() # scaling only on int/floats
remove = ["Climate",
 "State",
 "Expectations",
 "weekday_Friday",
 "weekday_Monday",
 "weekday_Saturday",
 "weekday_Sunday",
 "weekday_Thursday",
 "weekday_Tuesday",
 "weekday_Wednesday"]

for col in remove:
    cols.remove(col)
    
y_train_scaled = copy.deepcopy(y_train_diff)
y_test_scaled = copy.deepcopy(y_test_diff)    
y_scaler = {}

X_scaler = StandardScaler()

X_train_scaled = copy.deepcopy(X_train_diff)
X_test_scaled = copy.deepcopy(X_test_diff)

for var in ["website", "manual"]:
    store_train = []
    store_test = []
    y_scaler[var] = StandardScaler()
    temp = y_scaler[var].fit_transform(y_train_diff[var].values.reshape(-1,1))
    for i in range(0, len(temp)): #loop to create array, no other method found to change dimension of array-tuple
        store_train.append(temp[i][0])
    y_train_scaled[var] = pd.Series(store_train, index = y_train_diff.index)    
    temp =  y_scaler[var].transform(y_test_diff[var].values.reshape(-1,1))
    for i in range(0, len(temp)): #loop to create array, no other method found to change dimension of array-tuple
        store_test.append(temp[i][0])
    y_test_scaled[var] = pd.Series(store_test, index = y_test_diff.index)    

X_train_scaled[cols] = X_scaler.fit_transform(X_train_scaled[cols].copy())
X_test_scaled[cols] = X_scaler.transform(X_test_scaled[cols].copy())

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


### NOT USED BECAUSE IT DOES NOT INCREASE ACCURACY
### Use only feature from RF feature importance list
with open("features.pkl", "rb") as fp:
    features = pickle.load(fp)

X_test_diff = X_test_diff.loc[:, features.tolist()]
X_train_diff = X_train_diff.loc[:, features.tolist()]


######################
###### NN Tuning #####
######################

"""
IMPORTANT: Hyperparameter tuning of neural networks is computationally expensive and was too much for my old laptop to handle when the number of
           optuna trials and NN epochs was greater than 2 respectively. Therefore, the results in my submission paper are achieved without parameter tuning. 
           When uncommenting the next part, the tuning routine starts for each lead_type and is expected to enhance the results of the baseline TFT model
"""


"""
#init study and params dict
best_params_tft = {}
study_tft = {}
n_trials = 100  #number of trials in optuna, the higher the longer tuning takes but performance of model increases (should be 50 at least)

for var in ["website", "manual"]:
    # optimize hyperparameters by minimizing the RMSE on the validation set
    np.random.seed(42) #reproducability
    tuner = NNTuner(X_train_ts, y_train_ts[var]) # init Tuning class
    study_tft[var] = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=np.random.seed(42)))                                 
    study_tft[var].optimize(tuner.objective, n_trials=n_trials) # start optimization
    best_params_tft[var] = study_tft[var].best_params #store results

##Safe best params
with open("best_params_tft.pkl", "wb") as fp:
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
n_epochs = 300 # number of epochs for the NN model, 

for var in ["website", "manual"]:
    tft[var] = final_prediction(X_train_ts, X_test_ts, y_train_ts[var], y_test_ts[var]) # init final_prediction class for data transformation
    preds_tft[var], rmse_tft[var] = tft[var].predict(n_epochs = n_epochs) # make predictions
    
    # uncomment line below and comment the above line out in order to use optimized parameters
    #preds_tft[var], rmse_tft[var] = tft.predict(n_epochs = n_epochs, best_params = best_params_tft[var]) 
                
##Safe predictions and rmse
with open("{}\\output\\preds_tft.pkl".format(path), "wb") as fp:
   pickle.dump(preds_tft, fp)   
 
with open("{}\\output\\rmse_tft.pkl".format(path), "wb") as fp:
   pickle.dump(rmse_tft, fp)    