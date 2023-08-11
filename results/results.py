import pathlib
import numpy as np
import pandas as pd
import optuna 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from stacking_functions import train_oof_predictions, create_meta_dataset, stack_prediction
import pickle
import copy
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.io as pio
import kaleido
import plotly.express as px
pio.renderers.default="browser"


# current working directory
path = pathlib.Path().absolute()


################
## Folders ##
################
data_input_folder         = "{}\\dataset\\output".format(path)
ml_input_folder         = "{}\\ml_models\\output".format(path)
stack_input_folder         = "{}\\stacking\\output".format(path)
tft_input_folder         = "{}\\temporal_fusion_transformer\\output".format(path)

output_folder        = "{}\\results\\output".format(path)

#####################
### Load datasets ###
#####################

### Data 
df_train = pd.read_excel("{}\\data.xlsx".format(data_input_folder), sheet_name = 0)
df_val = pd.read_excel("{}\\data.xlsx".format(data_input_folder), sheet_name = 1)
df_test = pd.read_excel("{}\\data.xlsx".format(data_input_folder), sheet_name = 2)

df_train = df_train.set_index("Date")
df_val = df_val.set_index("Date")
df_test = df_test.set_index("Date")

### Use Differenced Data 
df_train_diff = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 0)
df_val_diff  = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 1)
df_test_diff  = pd.read_excel("{}\\data_diff.xlsx".format(data_input_folder), sheet_name = 2)

df_train_diff  = df_train_diff.set_index("Date")
df_val_diff  = df_val_diff.set_index("Date")
df_test_diff  = df_test_diff.set_index("Date")


#### Create dedicated target and feature sets 
X_train = df_train.drop(["website", "manual"], axis = 1)
y_train = df_train[["website", "manual"]]

X_val = df_val.drop(["website", "manual"], axis = 1)
y_val = df_val[["website", "manual"]]

X_test = df_test.drop(["website", "manual"], axis = 1)
y_test = df_test[["website", "manual"]]

X_train_diff = df_train_diff.drop(["website", "manual"], axis = 1)
y_train_diff = df_train_diff[["website", "manual"]]

X_val_diff = df_val_diff.drop(["website", "manual"], axis = 1)
y_val_diff = df_val_diff[["website", "manual"]] 

X_test_diff = df_test_diff.drop(["website", "manual"], axis = 1)
y_test_diff = df_test_diff[["website", "manual"]] 

#### Creat complete training dataset for final prediction (train+val combined)
X_combined = pd.concat([X_train, X_val], axis = 0 )
y_combined = pd.concat([y_train, y_val], axis = 0 )

X_combined_diff = pd.concat([X_train_diff, X_val_diff], axis = 0 )
y_combined_diff = pd.concat([y_train_diff, y_val_diff], axis = 0 )


##### Standardize data ####
#Important only fit scaler to training data, not to test data to prevent data leakage

cols = np.array(X_combined_diff.columns).tolist() # scaling only on continuous variables
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
    
y_combined_scaled = copy.deepcopy(y_combined_diff)
  
y_scaler = {}
X_scaler = StandardScaler()

X_combined_scaled = copy.deepcopy(X_combined_diff)
X_test_scaled = copy.deepcopy(X_test_diff)

for var in ["website", "manual"]:
    store_combined = []
    store_test = []
    y_scaler[var] = StandardScaler()
    temp = y_scaler[var].fit_transform(y_combined_diff[var].values.reshape(-1,1)) #fit_transform to combined data
    for i in range(0, len(temp)): #loop to create array, no other method found to change dimension of array-tuple
        store_combined.append(temp[i][0])
    y_combined_scaled[var] = pd.Series(store_combined, index = y_combined_diff.index)    


X_combined_scaled[cols] = X_scaler.fit_transform(X_combined_scaled[cols].copy()) #fit_transform to train
X_test_scaled[cols] = X_scaler.transform(X_test_scaled[cols].copy()) #transform test


## Create lagged targets from data for first difference re-transformation
y_train_lag= y_train.shift(1).drop(y_train.index[0])

y_val_lag = pd.concat([y_train.tail(1), y_val], axis = 0).shift(1)
y_val_lag.drop(y_val_lag.index[0], inplace = True)

y_test_lag = pd.concat([y_val.tail(1), y_test], axis = 0).shift(1)
y_test_lag.drop(y_test_lag.index[0], inplace = True)


##################################
### Load results from analysis ###
##################################

## ML base model results
with open("{}\\test_rmse_ml.pkl".format(ml_input_folder), "rb") as fp:
    test_rmse = pickle.load(fp)    
with open("{}\\test_preds_ml.pkl".format(ml_input_folder), "rb") as fp:
    test_preds = pickle.load(fp) 
with open("{}\\models_dict_test.pkl".format(ml_input_folder), "rb") as fp:
    models_dict_ml = pickle.load(fp)
with open("{}\\models_dict_stack.pkl".format(ml_input_folder), "rb") as fp:
    models_dict_stack = pickle.load(fp)    

## Stacked model results
with open("{}\\best_stacks_ml.pkl".format(stack_input_folder), "rb") as fp:
    best_stacks = pickle.load(fp)    
with open("{}\\stack_preds.pkl".format(stack_input_folder), "rb") as fp:
    stack_preds = pickle.load(fp)
with open("{}\\stack_rmse.pkl".format(stack_input_folder), "rb") as fp:
    stack_rmse = pickle.load(fp)     
with open("{}\\stack_models.pkl".format(stack_input_folder), "rb") as fp:
    stack_models = pickle.load(fp)
with open("{}\\stack_params.pkl".format(stack_input_folder), "rb") as fp:
    stack_params = pickle.load(fp)
    
## TFT Model results
with open("{}\\preds_tft.pkl".format(tft_input_folder), "rb") as fp:
     preds_tft = pickle.load(fp)
with open("{}\\rmse_tft.pkl".format(tft_input_folder), "rb") as fp:
     rmse_tft = pickle.load(fp)        
    
##################################    
### Naive Forecast as Baseline ###
##################################

naive_rmse = {}

for var in ["website", "manual"]:
    naive_rmse[var] = mean_squared_error(y_test[var], y_test_lag[var], squared = False)

#######################   
### Compare Results ###
#######################
comparison = {} #init comparison dict
for var in ["website", "manual"]:
    comparison[var] = pd.DataFrame(stack_rmse[var].items(), columns=["model", "rmse"]) #add stacking results to df
    comparison[var]["model"] = "stack_"+comparison[var]["model"] #change name to distinguish between base models
    comparison[var] = pd.concat([comparison[var], pd.DataFrame(test_rmse[var].items(), columns=["model", "rmse"])], axis = 0) #add base model results
    comparison[var] = pd.concat([comparison[var], pd.DataFrame([["tft", rmse_tft[var]]], columns=["model", "rmse"])], axis = 0) #tft results
    comparison[var] = pd.concat([comparison[var], pd.DataFrame([["naive", naive_rmse[var]]], columns=["model", "rmse"])], axis = 0) #naive result
    comparison[var] = comparison[var].sort_values("rmse", ascending=True).reset_index(drop=True)


#######################
### Model averaging ###
#######################

#Use two models for averaging for each lead type
averaging = {}

for var in ["website", "manual"]:
    if var == "website":
        models = ["rf", "knn"] ##played around with a few combinations, this proved to be the most accurate for "website"
    if var == "manual":
        models = ["svm", "elastic"] 
    averaging[var] = pd.DataFrame(columns=["alpha", "rmse"])
    counter = 0    
    for a in np.linspace(0, 1, 101):
        avg = a*stack_preds[var][models[0]] + (1-a)*stack_preds[var][models[1]]
        rmse = mean_squared_error(y_val_diff[var], avg, squared=False)
        averaging[var].loc[counter] = [a, rmse]
        counter +=1
    averaging[var] = averaging[var].sort_values("rmse", ascending=True).reset_index(drop=True)     

    
#######################################################
### Final prediction on validation set (March 2023) ###
#######################################################    

final_preds_val = {}
final_preds_val_trans = {}
final_rmse_val = {}

for var in ["website", "manual"]:
    if var == "website":
        models = ["rf", "knn"] 
    if var == "manual":
        models = ["svm", "elastic"] 
    
    a = averaging[var]["alpha"][0]    
    final_preds_val[var] = a*stack_preds[var][models[0]] + (1-a)*stack_preds[var][models[1]] #use best averaging
    final_preds_val_trans[var] =  final_preds_val[var] + y_val_lag[var] #retransformation   
    final_rmse_val[var] = mean_squared_error(y_val[var], final_preds_val_trans[var] , squared=False)

# calculate RMSE for combined lead-types
final_rmse_val["combined"] = mean_squared_error(pd.concat([y_val["website"], y_val["manual"]], axis = 0), 
                                                pd.concat([final_preds_val_trans["website"], final_preds_val_trans["manual"]], axis = 0),    
                                                squared=False)





#############################################################
### Create Meta Datasets for Final Prediction (April 2023 ###
#############################################################

##### Since our best performing models are stacks, we have to create a new meta_dataset for the combined and test data

##### OOF predictions
# init dicts for data and models
data_y = {}
trained_models = {} #stores models and oof predictions

# loop over lead type
for var in ["website", "manual"]:
    data_x, data_y[var], trained_models[var] = train_oof_predictions(X_combined_scaled, y_combined_scaled[var],  models_dict_stack[var])

##### Init new training data frames  
X_combined_stack = pd.DataFrame(data_x, columns = X_combined_scaled.columns)
y_combined_stack = {}
for var in ["website", "manual"]:
    y_combined_stack[var] = pd.Series(data_y[var])
    
# Init dict for final oof predics
yhat_predics = {}
meta_X_combined = {}
meta_X_test = {}
final_models = {}

### Create meta training and test data
for var in ["website", "manual"]: 

    #Create list of predictions    
    yhat_predics[var] = {}
    meta_X_combined[var] = {}
    meta_X_test[var] = {}
    final_models[var] = {}
    
    for idx in best_stacks[var].index:
        meta = best_stacks[var]["Model"][idx] # idx: 0 = best models, 1 = second best etc.
        yhat_predics[var][meta] = []
        for model in best_stacks[var]["Included"][idx]: 
            trained_models[var][model][0].fit(X_combined_stack, y_combined_stack[var]) # fit base models to stack data
            yhat_predics[var][meta].append(trained_models[var][model][1]) # collect oof-predictions from before as new variables
        
        # create the meta training data set using the oof predictions, call create_meta_dataset
        meta_X_combined[var][meta] = create_meta_dataset(data_x, yhat_predics[var][meta])
        
        #create list of final base models
        final_models[var][meta] = []
        for model in best_stacks[var]["Included"][idx]: # idx: 0 = best models, 1 = second best etc.
            final_models[var][meta].append(trained_models[var][model][0]) # append fitted base models to final_models dict
        
        #create the meta test data set using the oof predictions, call stack_prediction
        meta_X_test[var][meta] = stack_prediction(X_test_scaled, final_models[var][meta])


#################################################
### Final prediction on test set (April 2023) ###
#################################################

### Collect predictions
final_stack_models = {}
final_diff_preds_test = {}
final_preds_test = {}
final_preds_test_trans = {}
final_rmse_test = {}

for var in ["website", "manual"]: 
    if var == "website":
        models = ["rf", "knn"]
    if var == "manual":
        models = ["svm", "elastic"] 
    final_stack_models[var] = {}    
    final_diff_preds_test[var] = {}
    
    for mod in models: #fit best models to complete dataset and predict test data for april 2023
        store = []
        final_stack_models[var][mod] = copy.deepcopy(stack_models[var][mod][0]) #use hyperparams from stacked models
        
        final_stack_models[var][mod].fit(meta_X_combined[var][mod], y_combined_stack[var]) #fit model to final meta data
        temp_preds = final_stack_models[var][mod].predict(meta_X_test[var][mod]) #make prediction on meta data
        final_diff_preds_test[var][mod] = y_scaler[var].inverse_transform(temp_preds.reshape(-1, 1))  # undo scale transformation
        
        for i in range(0, len(final_diff_preds_test[var][mod])): #loop to create array, no other method found to change dimension of array-tuple
            store.append(final_diff_preds_test[var][mod][i][0])
        final_diff_preds_test[var][mod] = np.array(store)    

    
#### Averaging models        
final_preds_test = {}
final_preds_test_trans = {}
final_rmse_test = {}

for var in ["website", "manual"]:
    if var == "website":
        models = ["rf", "knn"] 
    if var == "manual":
        models = ["svm", "elastic"] 
    
    a = averaging[var]["alpha"][0]     #use averaging alphas from before
    
    final_preds_test[var] = a* final_diff_preds_test[var][models[0]] + (1-a)* final_diff_preds_test[var][models[1]] #use best averaging
    final_preds_test_trans[var] =  final_preds_test[var] + y_test_lag[var] #retransformation   
    final_rmse_test[var] = mean_squared_error(y_test[var], final_preds_test_trans[var], squared=False)

# calculate RMSE for combined lead-types
final_rmse_test["combined"] = mean_squared_error(pd.concat([y_test["website"], y_test["manual"]], axis = 0), 
                                                 pd.concat([final_preds_test_trans["website"], final_preds_test_trans["manual"]], axis = 0),    
                                                 squared=False)        

  
### Safe final RMSE results

with open("{}\\final_rmse_val.pkl".format(output_folder), "wb") as fp:
    pickle.dump(final_rmse_val ,fp)
with open("{}\\final_rmse_test.pkl".format(output_folder), "wb") as fp:
    pickle.dump(final_rmse_test ,fp)   
    
    
################################    
### Visualizations For Paper ###
################################

### Autocorrelation plot for original lead series ###
for var in ["website", "manual"]:
    plt.rcParams.update({"font.size": 15})
    acf = plot_acf(y_combined[var], title = "{} ACF Plot".format(var.capitalize()),auto_ylims = True)
    acf.savefig("{}\\{}_acf_plot.png".format(output_folder, var), bbox_inches="tight", pad_inches=0.3)

###### Plot TimeSpent and DAX Variable  ######
# Create figure with secondary y-axis
two_line = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
x_axis = X_combined.loc[X_combined.index > "2021-08-15"].index # only plot data from 15-08.2021 onwards, because no data for TimeSpent before

two_line.add_trace(
    go.Scatter(x=x_axis, y=X_combined.loc[X_combined.index > "2021-08-15", "Visits"], name="Visits"),
    secondary_y=False,
)

two_line.add_trace(
    go.Scatter(x=x_axis, y=X_combined.loc[X_combined.index > "2021-08-15", "DAX"], name="DAX"),
    secondary_y=True,
)

# Set x-axis title
two_line.update_xaxes(title_text="Date")

# Set y-axes titles
two_line.update_yaxes(title_text="Vistis", secondary_y=False)
two_line.update_yaxes(title_text="DAX", secondary_y=True)

two_line.update_layout(
    font=dict(size=20),
    margin=dict(l=20, r=20, t=20, b=20)
    )
pio.write_image(two_line, "{}\\two_line.png".format(output_folder),scale=3, width=850, height=500)


### ##Line Plot of final validation predictions vs validation observations for "manual" leads (March 2023) ####
pred_line = go.Figure()

x_axis = y_val["manual"].index

pred_line.add_trace(
    go.Scatter(x=x_axis, y=y_val["manual"], name="Obs"))

pred_line.add_trace(
    go.Scatter(x=x_axis, y=final_preds_val_trans["manual"], name="Preds"))

# Set x-axis title
pred_line.update_xaxes(title_text="Date")

pred_line.update_layout(
    font=dict(size=20),
    margin=dict(l=20, r=20, t=20, b=20)
    )

pio.write_image(pred_line, "{}\\pred_line.png".format(output_folder),scale=3, width=850, height=500)

#######################################################
### Stack Feature Importance plot for Website Leads ###
#######################################################

#RF Feature Importance
feature_names = list(X_train.columns)
for x in ["GBT_preds", "SVM_preds", "DT_preds", "GBLM_preds"]: feature_names.append(x)
rf_stack_gain = np.round(stack_models["website"]["rf"][0].feature_importances_ * 100, 2) #multiply with 100 for better visualization
rf_stack_imp = pd.DataFrame({"Information Gain": rf_stack_gain, "Feature": feature_names} )    
rf_stack_imp.sort_values("Information Gain", ascending = False, inplace = True)
rf_stack_imp.reset_index(drop = True, inplace=True)

# Elastic Feature Importance
feature_names = list(X_train.columns)
for x in ["DT_preds"]: feature_names.append(x)

elastic_stack_imp = pd.DataFrame({"Importance": np.round(np.abs(stack_models["website"]["elastic"][0].coef_), 2), "Feature": feature_names})
elastic_stack_imp.sort_values("Importance", ascending = False, inplace = True)

# Build traces
rf_trace = go.Scatter(
        x = rf_stack_imp["Information Gain"].values[0:10][::-1], # Only first ten features 
        y = rf_stack_imp["Feature"].values[0:10][::-1],
        mode="markers+text",
        marker=dict(
            sizemode = "diameter",
            sizeref = 1,
            size = 16,
            color = np.arange(0, 10),
            colorscale="Portland",
            showscale=False
        ),
        text = rf_stack_imp["Information Gain"].values[0:10][::-1]
    )

elastic_trace = go.Scatter(
        x = elastic_stack_imp["Importance"].values[0:10][::-1], # Only first ten features 
        y = elastic_stack_imp["Feature"].values[0:10][::-1],
        mode="markers+text",
        marker=dict(
            sizemode = "diameter",
            sizeref = 1,
            size = 16,
            color = np.linspace(0, 0.3, 10),
            colorscale="Portland",
            showscale=False
        ),
        text = elastic_stack_imp["Importance"].values[0:10][::-1]
    )

#Make plot
imp_plot = make_subplots(rows=1, cols=2, subplot_titles=("Randomn Forest",  "Elastic Net"))

imp_plot.add_trace(
    rf_trace,
    row = 1, col = 1
    ) 

imp_plot.add_trace(
    elastic_trace,
    row = 1, col = 2
    ) 

imp_plot.update_traces(showlegend=False, textposition="top center")
imp_plot.update_layout(
    font=dict(size=16),
    margin=dict(l=20, r=20, t=30, b=20),
    autosize= True,
    title= None,
    hovermode= "closest"
    )

# Ad X Axis titles


#Save
pio.write_image(imp_plot, "{}\\imp_website.png".format(output_folder),scale=3, width=1300, height=500)


##########################################
#### Random Forest Partial Dependence ####
##########################################
feature_names = list(X_train.columns)
for x in ["GBT_preds", "SVM_preds", "DT_preds", "GBLM_preds"]: feature_names.append(x)

data = meta_X_combined["website"]["rf"]
data.columns = feature_names #assign feature names to DataFrame

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("RandomForest")
rf_partial = PartialDependenceDisplay.from_estimator(stack_models["website"]["rf"][0], data, ["SVM_preds", "DT_preds", "website_lag1"], ax=ax)
fig.savefig("{}\\rf_partial.png".format(output_folder), bbox_inches="tight", pad_inches=0.3)
