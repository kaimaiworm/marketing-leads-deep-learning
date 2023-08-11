import numpy as np
import pandas as pd
import optuna 
from sklearn.metrics import mean_squared_error
from hyperparameter_ml import objective
from sklearn.model_selection import TimeSeriesSplit
import copy



########################
### Cross Validation ###
########################

def crossval(model, X, y, var = None):
    """
    Custom function for cross validation, using sklearn TimeSeriesSplit as base and RMSE as evaluation 

    """
    #set different validation sizes for variables, increases prediction accuracy
    if var == "website":
        size = 30 
    if var == "manual":
        size = 60
        
    cv = TimeSeriesSplit(n_splits=5) 
    cv_scores = np.empty(5)
    
    X = pd.DataFrame(X)
    y = pd.Series(y)
    mod = model

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X1, X2 = X.iloc[train_idx], X.iloc[test_idx]
        y1, y2 = y[train_idx], y[test_idx]
        
        mod.fit(X1, y1)
        preds = mod.predict(X2)
        cv_scores[idx] = mean_squared_error(y2, preds, squared=False)      
    return np.mean(cv_scores)



######################
### Model Stacking ###
######################


def train_oof_predictions(X, y, models, verbose=True):
    """
    Function to perform oof predictions on train data
    returns re-ordered predictors x, re-ordered target y, and model dictionary with filled predictors
    
    Important: Since we have a time series, we cannot make oof predictions for first fold, 
               i.e. place zero for first fold
    """
    
    # instantiate a KFold with 5 splits
    # we do not need TimeSeriesSplit here to perform CV but use it to create ordered oof-predictions
    kfold = TimeSeriesSplit(n_splits=5)
    
    # prepare lists to hold the x and y values and keep order
    data_x, data_y = [], []
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    data_x.extend(X_arr)
    data_y.extend(y_arr)
    
    mod = copy.deepcopy(models) #deepcopy do avoid any problems
    
    # run the following block for each of the kfold splits
    for idx, (train_idx, test_idx) in enumerate(kfold.split(X_arr, y_arr)):
    
        if verbose: print("\nStarting a new fold\n")
        print("\n {} \n".format(idx))
        if verbose: print("Creating splits")
        #create this fold"s training and test sets
        train_X, test_X = X_arr[train_idx], X_arr[test_idx] 
        train_y, test_y = y_arr[train_idx], y_arr[test_idx]
        
        if verbose: print("Adding x,y and ylag to lists\n")
        # add the data that is used in this fold to lists
        #data_x.extend(test_X)
        #data_y.extend(test_y)
    
        # run each model on this fold and add the predictors to the model"s running predictors list
        for item in mod:
            if idx == 0:
                mod[item][1].extend(np.zeros(len(train_y))) #fill predictions of first fold with zeroes, since no oof predictions available
            label = item # get label for verbose purposes
            model = mod[item][0] # get the model to use on the fold
                            
            # fit and make predictions 
            if verbose: print("Running",label,"on this fold")   
            model.fit(train_X, train_y) # fit to the train set for the kfold
            preds = model.predict(test_X) # fit on the sequential out-of-fold set
            mod[item][1].extend(preds) # add predictions to the model"s running predictors list
            
    return data_x, data_y, mod


def model_selector(X, y, meta_model, models_dict, model_label, var, verbose=True):
    """
    Function to select the best base models for each meta-model
    Basic function in steps:
        1. Choose a meta-model
        2. For current meta_model, perform CV on original data and obtain baseline accuracy
        3. Add oof predictions for one of the base models to training data, re-fit meta_model and obtain updated accuracy of meta_model 
            -> do this for all base-models
        4. Compare updated accuracies to baseline accuracy, add base-model whose updated accuracy was best to model stack
        5. In next round, again add oof predictions of base-models and see if accuracy of meta-model improves, add best model to stack
        6. Repeat 5. until accuracy no longer improves, then choose next meta_model
        
    """
    print("\n\nRunning model selector for ", model_label, "as meta-model")
    included_models = []
     
    while True:
        changed=False
        
        # forward step
        
        if verbose: print("\nNEW ROUND - Setting up score charts")
        excluded_models = list(set(models_dict.keys())-set(included_models)) # make a list of the current excluded_models
        if verbose: print("Included models: {}".format(included_models))
        if verbose: print("Exluded models: {}".format(excluded_models))
        new_acc = pd.Series(index=excluded_models, dtype=float) # make a series where the index is the current excluded_models
        
        current_meta_x = np.array(X)
        
        if len(included_models) > 0:
            for included in included_models:
                included = np.array(models_dict[included][1]).reshape((len(models_dict[included][1]), 1)) # gatheroof predictions of included models
                current_meta_x = np.hstack((current_meta_x, included)) # #add oof predictions of already included models to data stack
        #scores = cross_validate(meta_model, current_meta_x, y, cv=5, n_jobs=-1, scoring=("f1_weighted"))
        #starting_acc = round(scores["test_score"].mean(),3)
        starting_acc = round(crossval(meta_model, current_meta_x, y, var), 6)
        if verbose: print("Starting RMSE: {}\n".format(starting_acc))
       
        for excluded in excluded_models:  # for each item in the excluded_models list:
            
            new_yhat = np.array(models_dict[excluded][1]).reshape(-1, 1) # get the current models oof predictions
            meta_x = np.hstack((current_meta_x, new_yhat)) # add the predictions to the meta set
            
            # score the current item
            #scores = cross_validate(meta_model, meta_x, y, cv=5, n_jobs=-1, scoring=("f1_weighted"))
            #acc = round(scores["test_score"].mean(),3)
            acc = round(crossval(meta_model, meta_x, y, var), 6)
            if verbose: print("{} score: {}".format(excluded, acc))
            
            new_acc[excluded] = acc # append the rmse to the series field
        
        best_acc = new_acc.min() # evaluate best rmse of the excluded_models in this round
        if verbose: print("Best RMSE: {}\n".format(best_acc))
        
        if best_acc < starting_acc:  # if the best acc is better than the initial acc
            best_feature = new_acc.idxmin()  # define best oof predictions as new best feature
            included_models.append(str(best_feature)) # append this model name to the included list
            changed=True # flag that change happend
            if verbose: print("Add  {} with accuracy {}\n".format(best_feature, best_acc))
        else: changed = False
        
        if not changed:
            break  # stacking no longer increases performance
            
    print(model_label, "model optimized")
    print("resulting models:", included_models)
    print("Accuracy:", starting_acc)
    
    return included_models, starting_acc


def create_meta_dataset(data_x, items):
    """
    Function that takes in a data set and list of predictions, and forges into one dataset
    """
    # Deepcopies to avoid changes in original data
    meta_x = copy.deepcopy(data_x)
    yhat_preds = copy.deepcopy(items)
    
    # combine prediction and data for each model
    for z in yhat_preds:
        z = np.array(z).reshape((len(z), 1))
        meta_x = np.hstack((meta_x, z))
    meta_x = pd.DataFrame(meta_x)
    meta_x.columns = np.arange(0, len(meta_x.columns))    
    return meta_x


def stack_prediction(x_test, final_models): 
    """
    Takes in a test set and a list of fitted models.
    Fits each model in the list on the test set and stores it in a predictions list. 
    Then uses create_meta_dataset to combine test and predictions to be combined
    """
    predictions = []
    
    models = copy.deepcopy(final_models)
    X = copy.deepcopy(x_test)
    
    for model in models:
        preds = model.predict(X).reshape(-1,1) # make base models prediction for test set
        predictions.append(preds) 
    
    meta_X = create_meta_dataset(X, predictions)
        
    return meta_X

