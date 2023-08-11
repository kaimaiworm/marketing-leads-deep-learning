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
from hyperparameter_ml import stack_tuning
from stacking_functions import train_oof_predictions, model_selector, create_meta_dataset, stack_prediction
import pickle
import copy



# current working directory
path = pathlib.Path().absolute()


################
## Folders ##
################

data_input_folder         = "{}\\dataset\\output".format(path)
ml_input_folder         = "{}\\ml_models\\output".format(path)
output_folder        = "{}\\stacking\\output".format(path)


####################
### Load datasets ###
#####################

### Use Differenced Data 
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
### NOT USED BECAUSE IT DOES NOT INCREASE ACCURACY
### Use only feature from RF feature importance list
with open("features.pkl", "rb") as fp:
    features = pickle.load(fp)

X_test_diff = X_test_diff.loc[:, features.tolist()]
X_train_diff = X_train_diff.loc[:, features.tolist()]
"""


######################## 
#### Model Stacking ####
########################

##### Load needed model dictionary
with open("{}\\models_dict_stack.pkl".format(ml_input_folder), "rb") as fp:
        models_dict_stack = pickle.load(fp) 


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
        resulting_models, best_acc = model_selector(data_x, data_y[var],  meta_model, trained_models[var], label, var)
        scores[var]["Model"].append(model)
        scores[var]["RMSE"].append(best_acc)
        scores[var]["Included"].append(resulting_models)


# Transform scoreboard from dataframe to dictionary
best_models = {}
for var in ["website", "manual"]:
    best_models[var] = pd.DataFrame(scores[var])
    best_models[var] = best_models[var].sort_values("RMSE", ascending=True).reset_index(drop=True)

### Save best models
with open("{}\\best_stacks_ml.pkl".format(output_folder), "wb") as fp:
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
        meta_X_test[var][meta] = stack_prediction(X_test_scaled, final_models[var][meta])



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
with open("{}\\stack_params.pkl".format(output_folder), "wb") as fp:
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
with open("{}\\stack_models.pkl".format(output_folder), "wb") as fp:
    pickle.dump(meta_models, fp)
with open("{}\\stack_rmse.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_rmse, fp)
with open("{}\\stack_preds.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_preds, fp)  



