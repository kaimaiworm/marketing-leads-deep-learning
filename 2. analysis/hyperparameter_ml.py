import optuna 
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit

########################
### Cross Validation ###
########################

def hyper_crossval(model, X, y, var = None):
    """
    Custom function for cross validation, using sklearn TimeSeriesSplit as base and RMSE as evaluation 

    """
    #set different validation sizes for variables, increases optimization accuracy
    if var == "website":
        size = 30 
    if var == "manual":
        size = 60
        
    cv = TimeSeriesSplit(n_splits=5, test_size=size)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X1, X2 = X.iloc[train_idx], X.iloc[test_idx]
        y1, y2 = y[train_idx], y[test_idx]
        
        
        model.fit(X1, y1)
        preds = model.predict(X2)
        
        cv_scores[idx] = mean_squared_error(y2, preds, squared=False)
    return np.mean(cv_scores)


###########################
### Objective Functions ###
###########################

"""
Defines the objective functions for each of the ML models for hyperparameter tuning using optuna

Consists of:
    - space = search space of possible parameter values for optuna
    - reg = build each model
    - cv = rolling window CV using sklearn TimeSeriesSplit, calculate RMSE for each split
    - evalu = mean RMSE for all folds, returned to optuna for minimization
"""

"""
## If crossval does not work
        cv = TimeSeriesSplit(n_splits=5, test_size = 30)

        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
            X1, X2 = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y1, y2 = self.y[train_idx], self.y[test_idx]
            
            
            reg.fit(X1, y1)
            preds = reg.predict(X2)
            
            cv_scores[idx] = mean_squared_error(y2, preds, squared=False)
        evalu = np.mean(cv_scores)
        return evalu
        
"""

class objective:
    def __init__(self, X, y, var):
        self.X = X
        self.y = y
        self.var = var
        

    def gbt(self, trial):
        space={"max_depth": trial.suggest_int("max_depth", 3, 18, 1),
                "gamma": trial.suggest_float("gamma", 1,9),
                "alpha" : trial.suggest_int("alpha", 10,120, 1),
                "lambda" : trial.suggest_float("lambda", 1e-2, 1),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1, log = True),
                "min_child_weight" : trial.suggest_int("min_child_weight", 0, 10, 1),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1, log = True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000, 100),
                "seed": 0
            }
        
        reg=XGBRegressor(objective="reg:squarederror", booster = "gbtree", **space, eval_metric="rmse", random_state = 123, n_jobs = 4)
        evalu = hyper_crossval(reg, self.X, self.y, self.var)
        
        return evalu

    
    def gblm(self, trial):
        space={ "alpha" : trial.suggest_int("alpha", 20,180,1),
                "lambda" : trial.suggest_float("lambda", 1e-2, 1),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1, log = True),
                "feature_selector": trial.suggest_categorical("feature_selector", ["cyclic", "shuffle"]),
                "seed": 0
            }
        
        
        reg=XGBRegressor(objective="reg:squarederror", booster = "gblinear", **space, eval_metric="rmse", random_state = 123, n_jobs = 4)
        evalu = hyper_crossval(reg, self.X, self.y, self.var)

        return evalu
    
    def elastic(self, trial):
        space={ "alpha" : trial.suggest_float("alpha", 1e-2, 1, log = True),
                "l1_ratio" : trial.suggest_float("l1_ratio", 1e-2, 1, log = True)
            }
        
        reg = ElasticNet(**space, max_iter = 10000)        
        evalu = hyper_crossval(reg, self.X, self.y, self.var)
  
        return evalu
    
    def rf(self, trial): 
        space={"max_depth": trial.suggest_int("max_depth", 2, 12, 1),
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"]),
                "max_features" : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 4),
                "min_samples_split" : trial.suggest_int("min_samples_split", 2, 6),
                "n_estimators": trial.suggest_int("n_estimators", 10, 300, 10)

            }
    
        reg=RandomForestRegressor(**space, bootstrap = False, random_state = 123, n_jobs = -1)
        evalu = hyper_crossval(reg, self.X, self.y, self.var)

        return evalu
    
    
    def dt(self, trial):
        space={"max_depth": trial.suggest_int("max_depth", 1, 12, 1),
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
                "max_features" : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 4),
                "min_samples_split" : trial.suggest_int("min_samples_split", 2, 6),
                "splitter": trial.suggest_categorical("splitter", ["best", "random"])

            }
    
        reg=DecisionTreeRegressor(**space, random_state = 123)
        evalu = hyper_crossval(reg, self.X, self.y, self.var)
 
        return evalu

    def knn(self, trial):
        space={"n_neighbors": trial.suggest_int("n_neighbors", 2, 15, 1),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm" : trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"]),
                "leaf_size" : trial.suggest_int("leaf_size", 10, 50, 1),
                "p" : trial.suggest_int("p", 1, 3)
            }
    
        reg=KNeighborsRegressor(**space)
        evalu = hyper_crossval(reg, self.X, self.y, self.var)
        return evalu

    
    def svm(self, trial):
        space={"kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma" : trial.suggest_categorical("gamma", ["scale", "auto"]),
                "C" : trial.suggest_float("C", 0, 1)
            }
    
        reg=SVR(**space)
        evalu = hyper_crossval(reg, self.X, self.y, self.var)

        return evalu




def tuning(objective_function, models):
    """
    Helper function for hyperparameter tuning using optuna, calls hyperparameter_ml.objective()
    """
    # init dicts
    study_dict = {}
    best_params = {}
    
    #iterate through lidt of models
    for mod in models:
        np.random.seed(42) #seed for reproducability
        #create study
        study_dict[mod] = optuna.create_study(direction="minimize",
                                              study_name = "study_{}".format(mod),
                                              sampler=optuna.samplers.TPESampler(seed=42),
                                              pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        study_dict[mod].optimize(getattr(objective_function, str(mod)), n_trials=100) # start optimization for current model
        best_params[mod] = study_dict[mod].best_params #save best pararms for current model
    return best_params


def stack_tuning(X, y, variable, models):
    """
    Function used for hyperparameter tuning the stacked models through hyperparameter_ml.py
    """
    # init dicts
    study_dict = {}
    best_params = {}
    for mod in models:
        stack_tuner = objective(X[mod], y, variable) #define objective for each var and model
        np.random.seed(42)
        #create study for each model stack
        study_dict[mod] = optuna.create_study(direction="minimize",
                                              study_name = "study_{}".format(mod),
                                              sampler=optuna.samplers.TPESampler(seed=42),
                                              pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        study_dict[mod].optimize(getattr(stack_tuner, str(mod)), n_trials=100) #start optimization
        best_params[mod] = study_dict[mod].best_params #safe best parameters
    return best_params


    

