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
#from neural_network import NNTuner, final_prediction
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


         
         
"""
### NOT USED BECAUSE IT DOES NOT INCREASE ACCURACY
### Use only feature from RF feature importance list
with open("features.pkl", "rb") as fp:
    features = pickle.load(fp)

X_test_diff = X_test_diff.loc[:, features.tolist()]
X_train_diff = X_train_diff.loc[:, features.tolist()]
"""


### How strong is correlation between types of leads (possible simultaneous forecasting?)
web = np.array(y_train["website"])
man = np.array(y_train["manual"])
print(np.corrcoef(web, man))
# correlation is weak, no simultaneous forecasting needed


############################
###### Tune ML Models ######
############################

# Define list of models used for prediction
models = ["elastic", "knn", "svm", "dt", "rf", "gbt", "gblm"] 
#models = ["dt"] 

# Init dict to store tuning parameters
best_params = {}

# Tune models for each lead-type
for var in ["website", "manual"]:
    tuner = objective(X_train_scaled, y_train_scaled[var], var) #define objective for each type
    best_params[var] = tuning(tuner, models) # store


## Save best parameters
with open("{}\\output\\best_params_ml.pkl".format(path), "wb") as fp:
    pickle.dump(best_params, fp)

       
#################################
#### Prediction on test data ####
#################################

# init dictionary of models  
models_dict_test = {}
# Needed later for model stacking, but easier to setup here
elastic_yhat, knn_yhat, svm_yhat, dt_yhat, rf_yhat, gbt_yhat, gblm_yhat= [], [], [], [], [], [], []

for var in ["website", "manual"]:  
    # initiate models and store in dict, use optuna parameters
    gbt = XGBRegressor(objective="reg:squarederror", booster = "gbtree", **best_params[var]["gbt"], eval_metric="rmse", random_state = 123, n_jobs = 4)
    gblm = XGBRegressor(objective="reg:squarederror", booster = "gblinear", **best_params[var]["gblm"], eval_metric="rmse", random_state = 123, n_jobs = 4)
    rf = RandomForestRegressor(**best_params[var]["rf"], random_state = 123, n_jobs = -1)
    elastic = ElasticNet(**best_params[var]["elastic"], max_iter = 10000)        
    dt = DecisionTreeRegressor(**best_params[var]["dt"], random_state = 123)
    knn = KNeighborsRegressor(**best_params[var]["knn"])
    svm = SVR(**best_params[var]["svm"])
    
    models_dict_test[var] = {"elastic": [elastic, elastic_yhat],
                        "knn": [knn, knn_yhat],
                        "svm": [svm, svm_yhat],
                        "dt": [dt, dt_yhat],
                        "rf" : [rf, rf_yhat], 
                        "gbt" : [gbt, gbt_yhat],  
                        "gblm" : [gblm, gblm_yhat]
                        }
    
# make deepcopy before fitted to data     
models_dict_stack = copy.deepcopy(models_dict_test)

# init dicts for storage
test_rmse = {}
test_preds_scaled = {}
test_preds = {}


### Make prediction on scaled test data
for var in ["manual", "website"]:
    # init dict inside dicts to store for each model
    test_rmse[var] = {}
    test_preds_scaled[var]= {}
    test_preds[var] = {}
    
    for mod in models_dict_test[var].keys():
        store = []
        models_dict_test[var][mod][0].fit(X_train_scaled, y_train_scaled[var]) #fit model to scaled test data
        test_preds_scaled[var][mod] = models_dict_test[var][mod][0].predict(X_test_scaled)  # predict on scaled test data 
        test_preds[var][mod] = y_scaler[var].inverse_transform(test_preds_scaled[var][mod].reshape(-1, 1)) #undo scale transformation 
        
        for i in range(0, len(test_preds[var][mod])): #loop to create array, no other method found to change dimension of array-tuple
            store.append(test_preds[var][mod][i][0])
        test_preds[var][mod] = np.array(store)    
        test_rmse[var][mod] = mean_squared_error(y_test_diff[var], test_preds[var][mod], squared=False) #compute RMSE for preds vs. original FD
 

## Save model_dict, predictions and RMSE
with open("{}\\output\\models_dict_test.pkl".format(path), "wb") as fp:
    pickle.dump(models_dict_test, fp)
with open("{}\\output\\test_preds_ml.pkl".format(path), "wb") as fp:
    pickle.dump(test_preds, fp)
with open("{}\\output\\test_rmse_ml.pkl".format(path), "wb") as fp:
    pickle.dump(test_rmse, fp)    


var = "website"
all(y_test[var] == y_test_diff[var] + y_test_lag[var]) == True


trans = test_preds[var]["gbt"] + y_test_lag[var]
print(mean_squared_error(y_test[var], test_preds[var]["gbt"], squared=False))

px.line(x = y_test[var].index, y = [y_test[var], trans])

"""
#### FEATURE SELECTION VIA FEATURE IMPORTANCE OF RANDOM FOREST REGRESSOR
#### NOT USED BECAUSE BETTER ACCURACY WITH ALL FEATURES

n = 30  # number of best features to extract for each lead-type
liste = []
importance = {}

for var in ["website", "manual"]:
    mod = RandomForestRegressor(**best_params[var]["rf"], random_state = 123)
    mod.fit(X_train_diff, y_train_diff[var]) #fit model to train data
    importance[var] = pd.DataFrame(mod.feature_importances_, columns = ["imp"]).sort_values("imp", ascending=False)
    importance[var]["sum"] = np.cumsum(importance[var]["imp"])
    
    for i in importance[var].index[:n]: 
        liste.append(X_train_diff.columns[i])                                                        

features = np.unique(liste)
"""

########################
#### Model Stacking ####
########################

##### OOF predictions
# init dicts for data and models
data_y = {}
trained_models = {} #stores models and oof predictions

# loop over lead type
for var in ["website", "manual"]:
    data_x, data_y[var], trained_models[var] = train_oof_predictions(X_train_scaled, y_train_scaled[var],  models_dict_stack[var])


##### Model selection
# Set up a scoring dictionary to hold the model stack selector results
scores = {}

# loop over lead-type and models
for var in ["website", "manual"]:
    scores[var] = {}
    scores[var]["Model"] = []
    scores[var]["RMSE"] = []
    scores[var]["Included"] = []
    
    # Run the model stack selector for each model in trained_models
    for model in trained_models[var]:    
        meta_model = trained_models[var][model][0]
        label = model   
        resulting_models, best_acc = model_selector(data_x, data_y[var],  meta_model, trained_models[var], label, verbose=True)
        scores[var]["Model"].append(model)
        scores[var]["RMSE"].append(best_acc)
        scores[var]["Included"].append(resulting_models)


# Transform scoreboard from dataframe to dictionary
best_models = {}
for var in ["website", "manual"]:
    best_models[var] = pd.DataFrame(scores[var])
    best_models[var] = best_models[var].sort_values("RMSE", ascending=True).reset_index(drop=True)

### Save best models
with open("{}\\output\\best_models_ml.pkl".format(path), "wb") as fp:
    pickle.dump(best_models, fp)   


##### Init new training data frames  
X_train_stack = pd.DataFrame(data_x, columns = X_train_scaled.columns)
y_train_stack = {}
for var in ["website", "manual"]:
    y_train_stack[var] = pd.Series(data_y[var])

# Init dict for final oof predics
yhat_predics = {}
meta_X_train = {}
meta_X_test = {}
final_models = {}

### Create meta training and test data
for var in ["website", "manual"]: 

    #Create list of predictions    
    yhat_predics[var] = {}
    meta_X_train[var] = {}
    meta_X_test[var] = {}
    final_models[var] = {}
    
    for idx in best_models[var].index:
        meta = best_models[var]["Model"][idx] # idx: 0 = best models, 1 = second best etc.
        yhat_predics[var][meta] = []
        for model in best_models[var]["Included"][idx]: 
            trained_models[var][model][0].fit(X_train_stack, y_train_stack[var]) # fit base models to stack data
            yhat_predics[var][meta].append(trained_models[var][model][1]) # collect oof-predictions from before as new variables
        
        # create the meta training data set using the oof predictions, call create_meta_dataset
        meta_X_train[var][meta] = create_meta_dataset(data_x, yhat_predics[var][meta])
        
        #create list of final base models
        final_models[var][meta] = []
        for model in best_models[var]["Included"][idx]: # idx: 0 = best models, 1 = second best etc.
            final_models[var][meta].append(trained_models[var][model][0]) # append fitted base models to final_models dict
        
        #create the meta test data set using the oof predictions, call stack_prediction
        meta_X_test[var][meta] = stack_prediction(X_test_diff, final_models[var][meta])



##############################
#### Tuning Stacked Models ###
##############################

# init meta model and parameter dict
meta_models = {}
stack_params = {} 

models = ["elastic", "knn", "svm", "dt", "rf", "gbt", "gblm"] # list of model names

for var in ["website", "manual"]:
        stack_params[var] = stack_tuning(meta_X_train[var], y_train_stack[var], var, models) #hyperparameter tuning

## Save best parameters
with open("{}\\output\\stack_params.pkl".format(path), "wb") as fp:
    pickle.dump(stack_params, fp)


################################
### Stacked Model Prediction ###
################################


#### Calculate RMSE for test period
# init dictionary 
meta_models = {}

#init models with best stack params
for var in ["website", "manual"]:  
    # initiate models and store in dict, use parameters from optuna
    gbt = XGBRegressor(objective="reg:squarederror", booster = "gbtree", **stack_params[var]["gbt"], eval_metric="rmse", random_state = 123, n_jobs = 4)
    gblm = XGBRegressor(objective="reg:squarederror", booster = "gblinear", **stack_params[var]["gblm"], eval_metric="rmse", random_state = 123, n_jobs = 4)
    rf = RandomForestRegressor(**stack_params[var]["rf"], random_state = 123, n_jobs = -1)
    elastic = ElasticNet(**stack_params[var]["elastic"], max_iter = 10000)        
    dt = DecisionTreeRegressor(**stack_params[var]["dt"], random_state = 123)
    knn = KNeighborsRegressor(**stack_params[var]["knn"])
    svm = SVR(**stack_params[var]["svm"])
    
    meta_models[var] = {"elastic": [elastic],
                        "knn": [knn],
                        "svm": [svm],
                        "dt": [dt],
                        "rf" : [rf], 
                        "gbt" : [gbt],  
                        "gblm" : [gblm]
                        }

stack_rmse = {} # init dict
stack_preds = {}
stack_preds_scaled = {}

for var in ["manual", "website"]:
    # init dict inside dict
    stack_rmse[var] = {} 
    stack_preds[var] = {} 
    stack_preds_scaled[var] = {}
    
    for mod in meta_models[var].keys():
        store = []
        meta_models[var][mod][0].fit(meta_X_train[var][mod], y_train_stack[var]) #fit model to scaled meta train data
        stack_preds_scaled[var][mod] = meta_models[var][mod][0].predict(meta_X_test[var][mod])  # predict on scaled meta test data 
        stack_preds[var][mod] = y_scaler[var].inverse_transform(stack_preds_scaled[var][mod].reshape(-1, 1)) #undo scale transformation 
        
        for i in range(0, len(stack_preds[var][mod])): #loop to create array, no other method found to change dimension of array-tuple
            store.append(stack_preds[var][mod][i][0])
        stack_preds[var][mod] = np.array(store)    
        stack_rmse[var][mod] = mean_squared_error(y_test_diff[var], stack_preds[var][mod], squared=False) #compute RMSE for preds vs. original FD
  

## Save models, predictions and rmse
with open("{}\\output\\meta_models.pkl".format(path), "wb") as fp:
    pickle.dump(meta_models, fp)
with open("{}\\output\\stack_rmse.pkl".format(path), "wb") as fp:
    pickle.dump(stack_rmse, fp)
with open("{}\\output\\stack_preds.pkl".format(path), "wb") as fp:
    pickle.dump(stack_preds, fp)  
    
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
with open("{}\\output\\best_params_tft.pkl", "wb") as fp:
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
    tft[var] = final_prediction(X_train_ts, X_test_ts, y_train_ts[var], y_test_ts[var], var) # init final_prediction class for data transformation
    preds_tft[var], rmse_tft[var] = tft[var].predict(n_epochs = n_epochs) # make predictions
    
    # uncomment line below and comment the above line out in order to use optimized parameters
    #preds_tft[var], rmse_tft[var] = tft.predict(n_epochs = n_epochs, best_params = best_params_tft[var]) 
                
##Safe predictions and rmse
with open("{}\\output\\preds_tft.pkl".format(path), "wb") as fp:
   pickle.dump(preds_tft, fp)   
 
with open("{}\\output\\rmse_tft.pkl".format(path), "wb") as fp:
   pickle.dump(rmse_tft, fp)     
   
    
 
   