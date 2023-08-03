import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from adtk.pipe import Pipeline
from adtk.visualization import plot
from adtk.detector import QuantileAD
from adtk.transformer import ClassicSeasonalDecomposition
from adtk.data import validate_series
from pandas.plotting import autocorrelation_plot
from darts.utils.statistics import check_seasonality
from darts.timeseries import TimeSeries
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



#################
### Functions ###
#################

#### Detect outliers, trends, seasonality and change points in order to see what engineering is necessary    
def remove_outlier(data, y):
    
    ts = TimeSeries.from_dataframe(data) #transform data frame to TS object for seasonality check via DARTS

    anomalies = {}

    cols = np.array(data.columns).tolist() #list of columns
    for var in y: 
        cols.remove(var) #no outlier detection for target variables
    
    ### Check seasonality of each time series, if True, detect outliers
    for var in cols:
        check = check_seasonality(ts[var])
        if check[0] == True:
            decomp = ClassicSeasonalDecomposition(freq = check[1]).fit_transform(data[var]) # seasonality decomposition of TS
            anomalies[var] = QuantileAD(high=0.995, low=0.005).fit_detect(decomp) # detect outliers by alpha = 5% - quantiles
                
    ### Detect outliers of variables without seasonality
    for key in anomalies.keys():
        cols.remove(key) # remove vars from list where outliers have already been detected
    for var in cols: 
        anomalies[var] = QuantileAD(high=0.995, low=0.005).fit_detect(data[var]) # detect outliers by quantile for leftover variables
        
    ## Remove outliers from data
    for key in anomalies.keys():
        data.loc[anomalies[key].index[anomalies[key] == True], key] = np.nan
      
    ## Linearly interpolate data    
    data = data.interpolate()    
    
    return data


#####################
### Load datasets ###
#####################


# Read in data 
leads_raw = pd.read_excel("{}\\TermPaperChallenge.xlsx".format(input_folder), sheet_name=0)
ads_raw = pd.read_excel("{}\\TermPaperChallenge.xlsx".format(input_folder), sheet_name=1)
website_raw = pd.read_excel("{}\\TermPaperChallenge.xlsx".format(input_folder), sheet_name=2)
macro_raw = pd.read_excel("{}\\TermPaperChallenge.xlsx".format(input_folder), sheet_name=3)
ifo_raw = pd.read_excel("{}\\ifo index.xlsx".format(input_folder), sheet_name = 0)

# Convert type "nan" in data to string
ads_raw.loc[ads_raw["Funnel"].isnull(), "Funnel"] = "nan"

########################
### Create Dataframe ###
########################

# Lead dictionary
leads_dict = {}
leads_dict["manual"] = leads_raw.loc[leads_raw["Type"] == "Manual", ["DateCreated", "NextDayLeads"]]
leads_dict["website"] = leads_raw.loc[leads_raw["Type"] == "Website", ["DateCreated", "NextDayLeads"]]

leads_dict["manual"] = leads_dict["manual"].rename(columns = {"DateCreated": "Date", "NextDayLeads": "manual"})
leads_dict["website"] = leads_dict["website"].rename(columns = {"DateCreated": "Date", "NextDayLeads": "website"})


# Macro dictionary
macro_dict = {}

for col in macro_raw.columns:
    macro_dict[col] = macro_raw.loc[(macro_raw["Date"] >= "2020-04-04") & (macro_raw["Date"] <= "2023-03-31"), ["Date", col]] 

macro_dict.pop("Date")

### Aggregate advertisement data in order to create longer time series
## What is the best solution? 
# 1. Aggregate all data by columns "Spend", "Impressions", "Click" for each day 
# 2. Aggregate via Funnel for each day
# 3. Aggregate via Platform for each day
# -> Try all and see which one gives best results

ads_all = ads_raw.groupby(["Date"], as_index = False)[["Spend", "Impressions", "Clicks"]].sum() # Aggregate all
ads_funnel = ads_raw.groupby(["Date", "Funnel"], as_index = False)[["Spend", "Impressions", "Clicks"]].sum() # Aggregation via funnel
ads_platform = ads_raw.groupby(["Date", "Platform"], as_index = False)[["Spend", "Impressions", "Clicks"]].sum() # Aggregation via platform


# Advert dictionary

ads_dict = {}

# 1. Aggregation of all data by columns
for col in ["Spend", "Impressions", "Clicks"]:
    ads_dict[col] = ads_all.loc[(ads_raw["Date"] >= "2020-04-04") & (ads_all["Date"] <= "2023-03-31"), ["Date", col]] 


"""
# 2. Aggreagation via Funnel
for col in ["Spend", "Impressions", "Clicks"]:
    for fun in ads_funnel["Funnel"].unique():
        name = "{}_{}".format(col, fun)
        ads_dict[name] = ads_funnel.loc[(ads_funnel["Date"] >= "2020-04-04") & 
                                             (ads_funnel["Date"] <= "2023-03-31") &
                                             (ads_funnel["Funnel"] == fun), ["Date", col]].rename(columns = {col: name})

# 3. Aggregation via Platform
for col in ["Spend", "Impressions", "Clicks"]:
    for plat in ads_funnel["Platform"].unique():
        name = "{}_{}".format(col, plat)
        ads_dict[name] = ads_platform.loc[(ads_platform["Date"] >= "2020-04-04") & 
                                             (ads_platform["Date"] <= "2023-03-31") &
                                             (ads_platform["Platform"] == plat), ["Date", col]].rename(columns = {col: name})

# Remove lists with zero entries (only for aggregation 2. and 3.)       
remove = []            
for key in ads_dict.keys(): 
    if len(ads_dict[key]) == 0:
        remove.append(key)  # create lists with keys that have no entries

for key in remove:
    ads_dict.pop(key)   # delete keys with no entries

"""
# Ifo Business Climate dictionary

ifo_dict = {}

for col in ifo_raw.columns:
    ifo_dict[col] = ifo_raw.loc[(website_raw["Date"] >= "2020-04-04") & (ifo_raw["Date"] <= "2023-03-31"), ["Date", col]] 
    
ifo_dict.pop("Date")

# Website Traffic dictionary

website_dict = {}

for col in ["Visits", "TimeSpent"]:
    website_dict[col] = website_raw.loc[(website_raw["Date"] >= "2020-04-04") & (website_raw["Date"] <= "2023-03-31"), ["Date", col]] 
  
    
# Create DataFrame with all data

df_all = pd.DataFrame({"Date": leads_dict["manual"]["Date"]})

for key in leads_dict.keys():
    df_all = pd.merge(df_all, leads_dict[key], on = "Date", how = "left")

for key in website_dict.keys():
    df_all = pd.merge(df_all, website_dict[key], on = "Date", how = "left")

for key in ads_dict.keys():
    df_all = pd.merge(df_all, ads_dict[key], on = "Date", how = "left")

for key in macro_dict.keys():
    df_all = pd.merge(df_all, macro_dict[key], on = "Date", how = "left")    

for key in ifo_dict.keys():
    df_all = pd.merge(df_all, ifo_dict[key], on = "Date", how = "left")        

    
###########################
##### Data Imputation #####
###########################

########
#Some imputations are done before train/test split to make coding easier, but these do not lead to data leakage
######## 

#### Macro Data ####
# Macro Data is only available on weekdays
# -> linearly impute value between Friday and Monday since values change over weekends as well
# This does not lead to data leakage as we do not bring info of test data into training data
for key in macro_dict.keys():
    #for the first two instances, use the data from the first monday
    df_all.loc[(df_all["Date"] <= "2020-04-05"), [key]] = np.array(df_all.loc[(df_all["Date"] == "2020-04-06"), [key]].copy())
    df_all[key] = df_all[key].interpolate() #linear interpolation for saturdays and sundays


#### Website Traffic Data #### 
# Zero traffic to website before 15-08-2021 
# -> set missing values in that time perido to zero
# Adding lots of zeros to data might harm predictions, try fitting without those variables or sliced dataframe
# No data leakage in this imputation step

for key in website_dict.keys():
    df_all.loc[(df_all["Date"] < "2021-08-15"), [key]] = 0
    
 
#### IFO Business Climate Data ####
# No data leakage in this imputation step
series = pd.date_range("2020-05-01", "2023-04-01", freq = "MS")

for key in ifo_dict.keys():
    # retrieve values from dict and set all values for april
    df_all.loc[(df_all["Date"] < "2020-05-01"), [key]] = np.array(ifo_dict[key].loc[(ifo_dict[key]["Date"] == "2020-04-01"), [key]]) 
    for i in range(0, len(series)):
        if i == 35: 
            break
        #set rest of to value of first of month
        df_all.loc[(df_all["Date"] > series[i]) & (df_all["Date"] < series[i+1]), [key]] = np.array(df_all.loc[(df_all["Date"] == series[i]), [key]])         
         
#### Advertisement data ####
# Ad data is only available from 14-06-2021 onwards, set data before that to zero
# Adding lots of zeros to data might harm predictions, try fitting without those variables or sliced dataframe
# No data leakage in this step
 
for key in ads_dict.keys():
    df_all.loc[(df_all["Date"] < "2021-06-14"), [key]] = 0

# Using 1. aggregation: Only two missing rows in ad data -> linear interpolation
# No data leakage in this step as interpolation is in training period, i.e. does not rely on test data
for key in ads_dict.keys():
    df_all[key] = df_all[key].interpolate() 

##########################
#### Train/Test Split ####
##########################

# Set Date as index in data frame
df_all = df_all.set_index("Date")

# Split dataset, use March 2023 as test data 
df_train = df_all.loc[df_all.index < "2023-03-01"].copy()
df_test = df_all.loc[df_all.index >= "2023-03-01" ].copy()


###############################
##### Feature Engineering #####
###############################

#### Plot all variables 
for col in df_all.columns:
    px.line(df_all, x=df_all.index, y = col)#.show()

    
#### Detect outliers in order to see what engineering is necessary    
df_train = remove_outlier(df_train, y = [])
df_test = remove_outlier(df_test, y = ["manual", "website"]) # no outlier detection for target variables in test data

# Cannot interpolate data that is at the beginning or end of data frame, fill with data of closest date 
for df in [df_train, df_test]:
    cols = df.columns[df.isna().any()].tolist() # determine columns with NaNs
    if not cols: continue # Go to next dataframe if no NaNs
    for col in cols:
        rows = df.loc[df[col].isna(), col] # determine rows with NaNs for column
        mid = df.index[int(len(df)/2)] # determine whether data is at beginning or end of data frame
        
        # if data is at the beginning of frame, use next possible value
        if all(rows.index < mid) == True:
            df.loc[(df.index <= rows.index.max()), [col]] = np.array(df.loc[(df.index== pd.date_range(rows.index.max(), periods=2)[1]), [col]].copy())
        # if data is at the endof frame, use next possible value
        elif all(rows.index < mid) == False:
            df.loc[(df.index >= rows.index.min()), [col]] = np.array(df.loc[(df.index== pd.date_range(end=rows.index.min(), periods=2)[0]), [col]].copy())
        # if data is on both sides of middle -> all NaNs, no interpolation possible, drop column
        else:
            df.drop(col, axis = 1, inplace = True)


#### Add lagged leads as features
# Since we forecast NextDayLeads, we naturally already have one period lags between y and X
# Calculate lagged variables 
df_lags = pd.DataFrame(index = df_all.index)
lags = [1, 7, 30] #include weekly and monthly lags 

cols = df_all.columns.tolist()
remove = ["Climate", "State", "Expectations"] #only 1 month lag for business climate variables
for col in remove:
    cols.remove(col)
    
for var in cols:
    for lag in lags:
        df_lags[str(var)+"_lag"+str(lag)] = df_all[var].shift(lag)

for var in remove:
    df_lags[str(var)+"_lag"+str(30)] = df_all[var].shift(30) #only 1 month lag for business climate variables
        
        
# Add to data frames
df_all = df_all.join(df_lags)
df_train = df_train.join(df_lags)
df_test = df_test.join(df_lags)    

#### Add rolling window statistics of leads
df_windows = pd.DataFrame(index = df_all.index)
window = [3, 7, 30]
for var in ["website", "manual"]:
    for win in window:
        df_windows[str(var)+"_roll"+str(win)] = df_all[var].rolling(window = win).mean()
    
 # Add to data frames
df_all = df_all.join(df_windows)
df_train = df_train.join(df_windows)
df_test = df_test.join(df_windows)    

##### Add day of week dummy variables 
for df in [df_all, df_train, df_test]:           
    df["weekday"] = pd.Series(df.index, name = "weekday", index = df.index).dt.day_name()

# Turn categorical variables into binary dummies
enc = OneHotEncoder()

all_cat_enc = pd.DataFrame(enc.fit_transform(df_all[["weekday"]]).toarray(), columns = enc.get_feature_names_out(), index = df_all.index)
train_cat_enc= pd.DataFrame(enc.fit_transform(df_train[["weekday"]]).toarray(), columns = enc.get_feature_names_out(), index = df_train.index)
test_cat_enc = pd.DataFrame(enc.fit_transform(df_test[["weekday"]]).toarray(), columns = enc.get_feature_names_out(), index = df_test.index)
weekdays = enc.get_feature_names_out().tolist()

df_all = pd.concat([df_all, all_cat_enc], axis = 1)
df_train = pd.concat([df_train, train_cat_enc], axis = 1)
df_test = pd.concat([df_test, test_cat_enc], axis = 1)
df_all.drop(columns = ["weekday"], inplace = True)
df_train.drop(columns = ["weekday"], inplace = True)
df_test.drop(columns = ["weekday"], inplace = True)


##### Autocorrelation plots
#manual leads
autocorrelation_plot(df_all["manual"]) #-> shows no significant autocorrelation

# website leads
autocorrelation_plot(df_all["website"]) #-> shows significant autocorrelation, possible non-stationary

###### Since website leads are non-stationary and predictors show trends, use differences for all variables
# Create first differences
cols = df_all.columns.tolist()
remove = ["Climate", "State", "Expectations"]+weekdays
for col in remove:
    cols.remove(col)
    
df_all_diff = df_all.copy()    
df_all_diff[cols] = df_all[cols] - df_all[cols].shift(1)

train_diff = df_all_diff.loc[df_all_diff.index.isin(df_train.index),]
test_diff = df_all_diff.loc[df_all_diff.index.isin(df_test.index),]

###### Drop NaNs due to added lags
for df in [df_train, train_diff]:
    cols = df.columns[df.isna().any()].tolist() # determine columns with NaNs
    rows = []
    if not cols: continue # Go to next dataframe if no NaNs
    for col in cols:
        rows.append(df.loc[df[col].isna(), col].index.max()) # determine highest rows with NaNs for column
    df.drop(index=df.index[df.index<=np.array(rows).max()], inplace = True)  


#### Detect outliers in order to see what engineering is necessary 
# No outlier detection for weekday dummies   
train_diff = remove_outlier(train_diff, y = remove)
test_diff = remove_outlier(test_diff, y = ["manual", "website"]+remove)

# Cannot interpolate data that is at the beginning or end of data frame, fill with data of closest date 
for df in [train_diff, test_diff]:
    cols = df.columns[df.isna().any()].tolist() # determine columns with NaNs
    if not cols: continue # Go to next dataframe if no NaNs
    for col in cols:
        rows = df.loc[df[col].isna(), col] # determine rows with NaNs for column
        mid = df.index[int(len(df)/2)] # determine whether data is at beginning or end of data frame
        
        # if data is at the beginning of frame, use next possible value
        if all(rows.index < mid) == True:
            df.loc[(df.index <= rows.index.max()), [col]] = np.array(df.loc[(df.index== pd.date_range(rows.index.max(), periods=2)[1]), [col]].copy())
        # if data is at the endof frame, use next possible value
        elif all(rows.index < mid) == False:
            df.loc[(df.index >= rows.index.min()), [col]] = np.array(df.loc[(df.index== pd.date_range(end=rows.index.min(), periods=2)[0]), [col]].copy())
        # if data is on both sides of middle -> all NaNs, no interpolation possible, drop column
        else:
            df.drop(col, axis = 1, inplace = True)

    


#######################
#### Save datasets ####  
#######################

with pd.ExcelWriter("{}\\data.xlsx".format(output_folder)) as writer:
    df_train.to_excel(writer, sheet_name="train")  
    df_test.to_excel(writer, sheet_name="test")
    
    
with pd.ExcelWriter("{}\\data_diff.xlsx".format(output_folder)) as writer:
    train_diff.to_excel(writer, sheet_name="train")  
    test_diff.to_excel(writer, sheet_name="test")    


    
      
