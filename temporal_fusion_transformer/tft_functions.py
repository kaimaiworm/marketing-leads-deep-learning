import numpy as np
import pandas as pd
from darts import concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import rmse
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.nn import MSELoss
import optuna 



###############################
#### Hyperparameter Tuning ####
###############################

# Define class for hyperparameter tuning
class NNTuner():
    """
    Class used to hyperparameter tuning neural network (Temporal Fusion Transformer in this case)
    """
    def __init__(self, X, y):
        """
        init takes in the features/covariates as well as the target (leads), creates training and validation splits and scales the variables
        """
        self.covariates = X
        self.series = y
        
        # Create training and validation sets:
        VAL_LEN = 185 # needs to be >= input+output chunk
        self.train, self.val = self.series[:-VAL_LEN], self.series[-VAL_LEN:]

        # Normalize the time series, only use train for fitting of scaler to avoid data leakage 
        self.transformer = Scaler()
        self.train_transformed = self.transformer.fit_transform(self.train)
        self.val_transformed = self.transformer.transform(self.val)
        self.series_transformed = self.transformer.transform(self.series)

        # transform covariates, only use train for fitting scaler to avoid data leakage 
        self.scaler_covs = Scaler()
        training_cutoff = self.train._time_index.max() #define cutoff date for train/validation split
        
        self.cov_train, self.cov_val = self.covariates.split_after(training_cutoff)
        self.cov_train_transformed = self.scaler_covs.fit_transform(self.cov_train)
        self.cov_val_transformed = self.scaler_covs.transform(self.cov_val)
        self.covariates_transformed = self.scaler_covs.transform(self.covariates)

    def objective(self, trial):
        """
        Objective function of TFT for optuna tuning:
            Each hyperparameter is assigned a custom search space
        """
        # hyperparams optimized by optuna
        out_len = trial.suggest_int("out_len", 5, 35, 5)
        in_len = trial.suggest_int("in_len", 30, 150, 15)
        hidden_size = trial.suggest_int("hidden_size", 8, 128, 12)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3) 
        num_attention_heads = trial.suggest_int("num_attention_heads", 1, 5) 
        dropout = trial.suggest_float("dropout", 1e-3, 0.4, log = True) 
        batch_size = trial.suggest_int("batch_size", 8, 128, 12) 
        lr = trial.suggest_float("lr", 0.0001, 0.01 , log=True) 

        # monitor validation progress and stop when no improvement 
        pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stopper = EarlyStopping("val_loss", min_delta=0.01, patience=3, verbose=True)
        callbacks = [pruner, early_stopper]
        
        
        # reproducibility
        torch.manual_seed(42)
    
        # build the TFT model
        model = TFTModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            batch_size=batch_size,
            num_attention_heads = num_attention_heads,
            dropout=dropout,
            n_epochs=50,
            add_encoders={"cyclic": {"future": ["month"]}},
            add_relative_index=False,
            optimizer_kwargs={"lr": lr},
            pl_trainer_kwargs = {"callbacks": callbacks},
            random_state=42,
            likelihood = None, # used to create point estimates instead of probabilities
            loss_fn = MSELoss()
        )
    
        # train the model
        model.fit(
            series = self.train_transformed, 
            val_series = self.val_transformed,
            
            past_covariates= self.covariates_transformed,
            val_past_covariates = self.cov_val_transformed
        )
    
        # Evaluate how good it is on the validation set, using RMSE
        n = 30 # determines how long forecast should be, set to 30 for comparison with prediction on test data
        num_samples = 1 # number of samples drawn, set to 1 since point estimates required
        preds = model.predict(series=self.train_transformed, n=n, num_samples = num_samples) #make prediction
        rmses = rmse(self.val_transformed, preds, n_jobs=-1, verbose=True) #calculate RMSEs
        rmse_val = np.mean(rmses)
    
        return rmse_val


##################
### Prediction ###
##################

##### Make final predictions on test dataset
class final_prediction():
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        init takes in training/test data, splits full training into additional training and validation data,
        and transforms the respective series
        """
        self.cov_train_full, self.cov_test = X_train, X_test
        self.train_full, self.test = y_train, y_test
        
        # Split training into train and validation
        VAL_LEN = 80 # needs to be >= input+output chunk
        self.train, self.val = self.train_full[:-VAL_LEN], self.train_full[-VAL_LEN:]

        # Normalize the training and validation time series, scaler only fitted to training data to avoid data leakage 
        self.transformer = Scaler()
        self.train_transformed = self.transformer.fit_transform(self.train)
        self.val_transformed = self.transformer.transform(self.val)
        
        # Transform full training time series for prediction 
        self.train_full_transformed = self.transformer.transform(self.train_full)
        
        # transform covariates, scaler only fitted to training data to avoid data leakage  
        # training:
        training_cutoff = self.train._time_index.max() #define cutoff date for train/validation split
        self.cov_train, self.cov_val = self.cov_train_full.split_after(training_cutoff)
        self.scaler_covs = Scaler()
        self.cov_train_transformed = self.scaler_covs.fit_transform(self.cov_train)
        self.cov_val_transformed = self.scaler_covs.transform(self.cov_val)
        # prediction:
        self.scaler_full_covs = Scaler()
        self.cov_train_full_transformed = self.scaler_full_covs.fit_transform(self.cov_train_full)
        self.cov_test_transformed = self.scaler_full_covs.transform(self.cov_test)
        self.covariates = self.cov_train_full.concatenate(self.cov_test, axis = 0)
        self.covariates_transformed = self.scaler_full_covs.transform(self.covariates)
            
    def predict(self, n_epochs, best_params = None):
        """
        Function that builds the FTF model, fits it on the training data and makes final predictions for the test data
        If no params supplied, the model is build using fixed hyperparameters, that are either recommended in the PyTorch documentation 
        or proved to deliver good results during testing

        """
        
        
        # early stop callback
        early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=3,
            min_delta=0.01,
            verbose=True
            )
        
        
            
        # reproducibility
        torch.manual_seed(42)
        
        if best_params == None:
            # build the TFT model with fixed hyperparameters
            model = TFTModel(
                input_chunk_length=60,
                output_chunk_length=7,
                hidden_size=32,
                lstm_layers=1,
                batch_size=64,
                num_attention_heads = 4,
                dropout=0.1,
                n_epochs=n_epochs,
                add_encoders={"cyclic": {"future": ["month"]}},
                add_relative_index=False,
                optimizer_kwargs={"lr": 0.001},
                pl_trainer_kwargs = {"callbacks": [early_stopper]},
                random_state=42,
                likelihood = None, #point estimates
                loss_fn = MSELoss(),
                )
          
        else:
            # build the TFT model with optimized hyperparameters
            model = TFTModel(
                input_chunk_length=best_params["in_len"],
                output_chunk_length=best_params["out_len"],
                hidden_size=best_params["hidden_size"],
                lstm_layers=best_params["lstm_layers"],
                batch_size=best_params["batch_size"],
                num_attention_heads = best_params["num_attention_heads"],
                dropout=best_params["dropout"],
                n_epochs=n_epochs,
                add_encoders={"cyclic": {"future": ["month"]}},
                add_relative_index=False,
                optimizer_kwargs={"lr": best_params["lr"]},
                pl_trainer_kwargs = {"callbacks": [early_stopper]},
                random_state=42,
                likelihood = None, #point estimates
                loss_fn = MSELoss()
                )
         
        #fit model on training data and covariates
        model.fit(  series = self.train_transformed, 
                    val_series = self.val_transformed,
                    past_covariates= self.covariates_transformed,
                    val_past_covariates = self.cov_val_transformed
                  )
        
        n = len(self.test) # set prediction_length to test period
        num_samples = 1 # number of samples drawn, set to 1 since point estimates required
        
        # Predict leads for the test period
        preds_transformed = model.predict(series=self.train_full_transformed,
                              past_covariates= self.covariates_transformed,
                              n=n, 
                              num_samples = num_samples)
        
        preds = self.transformer.inverse_transform(preds_transformed) #inverse scaling for predictions
        rmses = rmse(self.test, preds, n_jobs=-1) #compute rmse 
        rmse_val = np.mean(rmses)
        return preds, rmse_val
    
                
    