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
from hyperparameter_ml import objective, tuning
import pickle
import copy


# current working directory
path = pathlib.Path().absolute()


################
## Folders ##
################

data_input_folder         = "{}\\dataset\\output".format(path)
output_folder        = "{}\\ml_models\\output".format(path)


#####################
### Load datasets ###
#####################

### Use differenced Data 
df_train_diff = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 0)
df_test_diff  = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 1) 

df_train_diff  = df_train_diff.set_index("Date")
df_test_diff  = df_test_diff.set_index("Date")
      

#### Create dedicated target and feature sets
X_train_diff = df_train_diff.drop(["website", "manual"], axis = 1)
y_train_diff = df_train_diff[["website", "manual"]]

X_test_diff = df_test_diff.drop(["website", "manual"], axis = 1)
y_test_diff = df_test_diff[["website", "manual"]] 

##### Standardize data ####
#Important only fit scaler to training data, not to test data
cols = np.array(X_train_diff.columns).tolist() # scaling only on continuous variables
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
  

X_train_scaled[cols] = X_scaler.fit_transform(X_train_scaled[cols].copy())
X_test_scaled[cols] = X_scaler.transform(X_test_scaled[cols].copy())

         
"""
### NOT USED BECAUSE IT DOES NOT INCREASE ACCURACY ###

### Use only feature from RF feature importance list
with open("features.pkl", "rb") as fp:
    features = pickle.load(fp)

X_test_diff = X_test_diff.loc[:, features.tolist()]
X_train_diff = X_train_diff.loc[:, features.tolist()]
"""


############################
###### Tune ML Models ######
############################

# Define list of models used for prediction
models = ["elastic", "knn", "svm", "dt", "rf", "gbt", "gblm"] 

# Init dict to store tuning parameters
best_params_ml = {}

# Tune models for each lead-type
for var in ["website", "manual"]:
    tuner = objective(X_train_scaled, y_train_scaled[var], var) #define objective for each type
    best_params_ml[var] = tuning(tuner, models) # store


## Save best parameters
with open("{}\\best_params_ml.pkl".format(output_folder), "wb") as fp:
    pickle.dump(best_params_ml, fp)
  
       
#################################
#### Prediction on test data ####
#################################

# init dictionary of models  
models_dict_test = {}

# Needed later for model stacking, but easier to setup here
elastic_yhat, knn_yhat, svm_yhat, dt_yhat, rf_yhat, gbt_yhat, gblm_yhat= [], [], [], [], [], [], []

for var in ["website", "manual"]:  
    #initiate models and store in dict, use optuna parameters
    gbt = XGBRegressor(objective="reg:squarederror", booster = "gbtree", **best_params_ml[var]["gbt"], eval_metric="rmse", random_state = 123, n_jobs = 4)
    gblm = XGBRegressor(objective="reg:squarederror", booster = "gblinear", **best_params_ml[var]["gblm"], eval_metric="rmse", random_state = 123, n_jobs = 4)
    rf = RandomForestRegressor(**best_params_ml[var]["rf"], random_state = 123, n_jobs = -1)
    elastic = ElasticNet(**best_params_ml[var]["elastic"], max_iter = 10000)        
    dt = DecisionTreeRegressor(**best_params_ml[var]["dt"], random_state = 123)
    knn = KNeighborsRegressor(**best_params_ml[var]["knn"])
    svm = SVR(**best_params_ml[var]["svm"])
    
    models_dict_test[var] = {"elastic": [elastic, elastic_yhat],
                        "knn": [knn, knn_yhat],
                        "svm": [svm, svm_yhat],
                        "dt": [dt, dt_yhat],
                        "rf" : [rf, rf_yhat], 
                        "gbt" : [gbt, gbt_yhat],  
                        "gblm" : [gblm, gblm_yhat]
                        }
    
# make deepcopy for later, before fitted to data     
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
 

"""
#### FEATURE SELECTION VIA FEATURE IMPORTANCE OF RANDOM FOREST REGRESSOR
#### NOT USED BECAUSE BETTER ACCURACY WITH ALL FEATURES

n = 30  # number of best features to extract for each lead-type
liste = []
importance = {}

for var in ["website", "manual"]:
    mod = RandomForestRegressor(**best_params_ml[var]["rf"], random_state = 123)
    mod.fit(X_train_diff, y_train_diff[var]) #fit model to train data
    importance[var] = pd.DataFrame(mod.feature_importances_, columns = ["imp"]).sort_values("imp", ascending=False)
    importance[var]["sum"] = np.cumsum(importance[var]["imp"])
    
    for i in importance[var].index[:n]: 
        liste.append(X_train_diff.columns[i])                                                        

features = np.unique(liste)

with open("{}\\rf_features.pkl".format(output_folder), "wb") as fp:
    pickle.dump(features, fp)
"""  

## Save model_dict, predictions and RMSE
with open("{}\\models_dict_test.pkl".format(output_folder), "wb") as fp:
    pickle.dump(models_dict_test, fp)
with open("{}\\models_dict_stack.pkl".format(output_folder), "wb") as fp:
    pickle.dump(models_dict_stack, fp)    
with open("{}\\test_preds_ml.pkl".format(output_folder), "wb") as fp:
    pickle.dump(test_preds, fp)
with open("{}\\test_rmse_ml.pkl".format(output_folder), "wb") as fp:
    pickle.dump(test_rmse, fp)    
