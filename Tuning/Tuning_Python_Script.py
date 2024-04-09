import os
import joblib
import pandas as pd
import numpy as np
import random
import itertools

import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import sys
sys.path.append('/data/Hydra_Work/Competition_Functions') 
from Processing_Functions import process_forecast_date, process_seasonal_forecasts
from Data_Transforming import read_nested_csvs, generate_daily_flow, use_USGS_flow_data, USGS_to_daily_df_yearly

sys.path.append('/data/Hydra_Work/Pipeline_Functions')
from Folder_Work import filter_rows_by_year, csv_dictionary, add_day_of_year_column

sys.path.append('/data/Hydra_Work/Post_Rodeo_Work/ML_Functions.py')
from Full_LSTM_ML_Functions import Specific_Heads, Google_Model_Block, SumPinballLoss, EarlyStopper, Model_Run, No_Body_Model_Run



from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim.lr_scheduler as lr_scheduler


# All the prep
monthly_basins = ['animas_r_at_durango', 'boise_r_nr_boise', 'boysen_reservoir_inflow', 'colville_r_at_kettle_falls', 'detroit_lake_inflow', 'dillon_reservoir_inflow',
    'fontenelle_reservoir_inflow', 'green_r_bl_howard_a_hanson_dam', 'hungry_horse_reservoir_inflow', 'libby_reservoir_inflow',
    'missouri_r_at_toston','owyhee_r_bl_owyhee_dam', 'pecos_r_nr_pecos', 'pueblo_reservoir_inflow',
    'ruedi_reservoir_inflow', 'skagit_ross_reservoir', 'snake_r_nr_heise', 'stehekin_r_at_stehekin', 'sweetwater_r_nr_alcova',
    'taylor_park_reservoir_inflow', 'virgin_r_at_virtin', 'weber_r_nr_oakley', 'yampa_r_nr_maybell',
]


USGS_basins = ['animas_r_at_durango', 'boise_r_nr_boise', 'boysen_reservoir_inflow', 'colville_r_at_kettle_falls', 'detroit_lake_inflow', 'dillon_reservoir_inflow',   
    'green_r_bl_howard_a_hanson_dam', 'hungry_horse_reservoir_inflow', 'libby_reservoir_inflow', 'merced_river_yosemite_at_pohono_bridge', 'missouri_r_at_toston',
    'owyhee_r_bl_owyhee_dam', 'pecos_r_nr_pecos', 'pueblo_reservoir_inflow',    'san_joaquin_river_millerton_reservoir', 'snake_r_nr_heise', 'stehekin_r_at_stehekin',
    'sweetwater_r_nr_alcova', 'taylor_park_reservoir_inflow', 'virgin_r_at_virtin', 'weber_r_nr_oakley', 'yampa_r_nr_maybell',
]

basins = list(set(monthly_basins + USGS_basins))


selected_years = range(2000,2024,2)

era5_folder = '/data/Hydra_Work/Rodeo_Data/era5'
era5 = csv_dictionary(era5_folder, basins, years=selected_years)
era5 = add_day_of_year_column(era5)

flow_folder = '/data/Hydra_Work/Rodeo_Data/train_monthly_naturalized_flow'
flow = csv_dictionary(flow_folder, monthly_basins)
flow = filter_rows_by_year(flow, 1998)

climatology_file_path = '/data/Hydra_Work/Rodeo_Data/climate_indices.csv'
climate_indices = pd.read_csv(climatology_file_path)
climate_indices['date'] = pd.to_datetime(climate_indices['date'])
climate_indices.set_index('date', inplace = True)
climate_indices.drop('Unnamed: 0', axis = 1, inplace = True)
climate_indices = climate_indices[~climate_indices.index.duplicated(keep='first')]

root_folder = '/data/Hydra_Work/Rodeo_Data/seasonal_forecasts'
seasonal_forecasts = read_nested_csvs(root_folder)

USGS_flow_folder = '/data/Hydra_Work/Rodeo_Data/USGS_streamflows'
USGS_flow = csv_dictionary(USGS_flow_folder, USGS_basins)

Static_variables = pd.read_csv('/data/Hydra_Work/Rodeo_Data/static_indices.csv', index_col= 'site_id')

# Convert monthly flow values to daily flow estimates
daily_flow = {}

# Iterate through the dictionary and apply generate_daily_flow to each DataFrame
for key, df in flow.items():
    daily_flow[key] = generate_daily_flow(df, persistence_factor=0.7)

# Replacing monhtly data for normalised USGS when available
daily_flow = use_USGS_flow_data(daily_flow, USGS_flow)


normalising_basins = ['san_joaquin_river_millerton_reservoir', 'merced_river_yosemite_at_pohono_bridge', 'detroit_lake_inflow']

for basin in normalising_basins:
    path = f'/data/Hydra_Work/Rodeo_Data/USGS_streamflows/{basin}.csv' 
    normalising_path = f'/data/Hydra_Work/Rodeo_Data/train_yearly/{basin}.csv'
    USGS_to_daily_df_yearly(daily_flow, path, basin, normalising_path)

climate_scaler_filename = '/data/Hydra_Work/Rodeo_Data/scalers/climate_normalization_scaler.save'
climate_scaler = joblib.load(climate_scaler_filename) 
climate_indices = pd.DataFrame(climate_scaler.transform(climate_indices), columns=climate_indices.columns, index=climate_indices.index)

era5_scaler_filename = '/data/Hydra_Work/Rodeo_Data/scalers/era5_scaler.save'
era5_scaler = joblib.load(era5_scaler_filename) 
era5 = {key: pd.DataFrame(era5_scaler.transform(df), columns=df.columns, index=df.index) for key, df in era5.items()}

for basin, df in daily_flow.items(): 
    flow_scaler_filename = f'/data/Hydra_Work/Rodeo_Data/scalers/flows/{basin}_flow_scaler.save'
    flow_scaler = joblib.load(flow_scaler_filename) 
    daily_flow[basin] = pd.DataFrame(flow_scaler.transform(df), columns=df.columns, index=df.index)

seasonal_scaler_filename = "/data/Hydra_Work/Rodeo_Data/scalers/seasonal_scaler.save"
seasonal_scaler = joblib.load(seasonal_scaler_filename)
seasonal_forecasts = {key: pd.DataFrame(seasonal_scaler.transform(df), columns=df.columns, index=df.index ) for key, df in seasonal_forecasts.items()}

static_scaler_filename = '/data/Hydra_Work/Rodeo_Data/scalers/static_scaler.save'
static_scaler = joblib.load(static_scaler_filename) 
Static_variables = pd.DataFrame(static_scaler.transform(Static_variables), columns=Static_variables.columns, index=Static_variables.index)

climatological_flows = {}

for basin, df in daily_flow.items():
    # Extract day of year and flow values
    df['day_of_year'] = df.index.dayofyear

    grouped = df.groupby('day_of_year')['daily_flow'].quantile([0.1, 0.5, 0.9]).unstack(level=1)

    climatological_flows[basin] = pd.DataFrame({
        'day_of_year': grouped.index,
        '10th_percentile_flow': grouped[0.1],
        '50th_percentile_flow': grouped[0.5],
        '90th_percentile_flow': grouped[0.9]
    })
    
    climatological_flows[basin].set_index('day_of_year', inplace=True)

    # Drop the temporary 'day_of_year' column from the original dataframe
    df.drop(columns='day_of_year', inplace=True)

criterion = SumPinballLoss(quantiles = [0.1, 0.5, 0.9])

basin = 'animas_r_at_durango' 
All_Dates = daily_flow[basin].index[
    ((daily_flow[basin].index.month < 6) | ((daily_flow[basin].index.month == 6) & (daily_flow[basin].index.day < 25))) &
    ((daily_flow[basin].index.year % 2 == 0) | ((daily_flow[basin].index.month > 10) | ((daily_flow[basin].index.month == 10) & (daily_flow[basin].index.day >= 1))))
]
All_Dates = All_Dates[All_Dates.year > 1998]


# Validation Year
Val_Dates = All_Dates[All_Dates.year == 2022]
All_Dates = All_Dates[All_Dates.year < 2022]


basin_to_remove = 'sweetwater_r_nr_alcova'

if basin_to_remove in basins:
    basins.remove(basin_to_remove)


seed = 42 ; torch.manual_seed(seed) ; random.seed(seed) ; np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

days  = 90
hindcast_input_size = 17

LR = 1e-3
static_size = np.shape(Static_variables)[1]
forecast_size = np.shape(seasonal_forecasts['american_river_folsom_lake_2000_apr'])[1]
History_Fourier_in_forcings = 0 #2*3*(6 - 1)
Climate_guess = 3
History_Statistics_in_forcings = 5*2

head_input_size = forecast_size + static_size + History_Fourier_in_forcings + History_Statistics_in_forcings  + Climate_guess + 3
head_output_size = 3


LR = 1e-3
static_size = np.shape(Static_variables)[1]
forecast_size = np.shape(seasonal_forecasts['american_river_folsom_lake_2000_apr'])[1]
History_Fourier_in_forcings = 0 #2*3*(6 - 1)
Climate_guess = 3
History_Statistics_in_forcings = 5*2

forecast_input_size = forecast_size + static_size + History_Fourier_in_forcings + History_Statistics_in_forcings  + Climate_guess + 3
output_size, head_hidden_size, head_num_layers =  3, 64, 3
hindcast_input_size = 17

# Do we want hindcast and forecast num-layers to be different?
def define_models(hindcast_input_size, forecast_input_size, hidden_size, num_layers, dropout, bidirectional, learning_rate, copies = 3, forecast_output_size = 3, device = device):
    models = {}
    params_to_optimize = {}
    optimizers = {}
    schedulers = {}
    
    hindcast_output_size = forecast_output_size
    for copy in range(copies):
        models[copy] = Google_Model_Block(hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout, bidirectional)
        
        models[copy].to(device)
        params_to_optimize[copy] = list(models[copy].parameters())

        optimizers[copy] = torch.optim.Adam(params_to_optimize[copy], lr= learning_rate, weight_decay = 1e-3)
        schedulers[copy] = lr_scheduler.CosineAnnealingLR(optimizers[copy], T_max=1e4)

    return models, params_to_optimize, optimizers, schedulers

def update_final_parameters(Final_Parameters, basin, min_val_loss_parameters, min_val_loss):
    Final_Parameters['basin'].append(basin)
    Final_Parameters['hidden_size'].append(min_val_loss_parameters[0])
    Final_Parameters['num_layers'].append(min_val_loss_parameters[1])
    Final_Parameters['dropout'].append(min_val_loss_parameters[2])
    Final_Parameters['bidirectional'].append(min_val_loss_parameters[3])
    Final_Parameters['learning_rate'].append(min_val_loss_parameters[4])
    Final_Parameters['val_loss'].append(min_val_loss)
    
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.search.optuna import OptunaSearch
import optuna

# Fixed parameters
total_epochs = 2
n_epochs = 1  # Epochs between tests
group_lengths = np.arange(180)
batch_size = 1
copies = 3

# parameters to tune
hidden_sizes = [16, 64, 128]
num_layers =  [1,3]
dropout = [0.1, 0.4]
bidirectional = [False, True]
learning_rate = [1e-3, 1e-5]

# Set up configuration space
config_space = {
    "hidden_size": tune.grid_search(hidden_sizes),
    "num_layers": tune.grid_search(num_layers),
    "dropout": tune.grid_search(dropout),
    "bidirectional": tune.grid_search(bidirectional),
    "learning_rate": tune.grid_search(learning_rate)
}

def train_model(config):

    All_Dates = ray.get(All_Dates_id)  
    Val_Dates = ray.get(Val_Dates_id)  
    era5 = ray.get(era5_id)  
    daily_flow = ray.get(daily_flow_id)  
    climatological_flows = ray.get(climatological_flows_id)
    climate_indices = ray.get(climate_indices_id)
    seasonal_forecasts = ray.get(seasonal_forecasts_id)
    Static_variables = ray.get(Static_variables_id)



    copies = 3
    
    device = torch.device('cuda' if torch.cuda.
                    is_available() else 'cpu')
    
    models, params_to_optimize, optimizers, schedulers = define_models(hindcast_input_size, forecast_input_size,
    config["hidden_size"], config["num_layers"], config["dropout"],
    config["bidirectional"], config["learning_rate"], copies=copies, device = device)


    losses, val_losses = [], []

    for epoch in range(total_epochs):

        train_losses = {}
        epoch_val_losses = {}

        for copy in range(copies):

             # Need to fix the outputs of No_Body_Model_Run
            train_losses[copy], Climate_Loss = No_Body_Model_Run(All_Dates, [basin], models[copy], era5, daily_flow, climatological_flows, climate_indices, seasonal_forecasts,
                Static_variables, optimizers[copy], schedulers[copy], criterion, early_stopper= None, n_epochs=n_epochs,
                batch_size=batch_size, group_lengths=group_lengths, Train_Mode=True, device=device, specialised=False)
            epoch_val_losses[copy], Climate_Loss = No_Body_Model_Run(Val_Dates, [basin], models[copy], era5, daily_flow, climatological_flows, climate_indices, seasonal_forecasts,
                Static_variables, optimizers[copy], schedulers[copy], criterion, early_stopper= None, n_epochs=n_epochs,
                batch_size=batch_size, group_lengths=group_lengths, Train_Mode=False, device=device, specialised=False)

        loss = np.mean(list(train_losses.values())) - Climate_Loss
        val_loss = np.mean(list(epoch_val_losses.values())).mean() - Climate_Loss

        ray.train.report({'val_loss' : val_loss})

        losses.append(loss)
        val_losses.append(val_loss)


    return val_loss

    

from ray import train, tune



ray.shutdown()
ray.init(runtime_env = { "env_vars":   {"PYTHONPATH": '/data/Hydra_Work/Competition_Functions/' } } )
         
All_Dates_id = ray.put(All_Dates)  
Val_Dates_id = ray.put(Val_Dates)  
era5_id = ray.put(era5)  
daily_flow_id = ray.put(daily_flow)  
climatological_flows_id = ray.put(climatological_flows)
climate_indices_id = ray.put(climate_indices)
seasonal_forecasts_id = ray.put(seasonal_forecasts)
Static_variables_id = ray.put(Static_variables)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='val_loss',
    mode='min',
    max_t=100,
    grace_period=5,
    reduction_factor=3,
    brackets=1,
)


optuna_search = OptunaSearch(
    define_optuna_search_space,
    metric="val_loss",
    mode="min")

plateau_stopper = TrialPlateauStopper(
    metric="val_loss",
    num_results = 4,
    grace_period=10,
    mode="min",
)


def objective(config):  
    device = torch.device('cuda' if torch.cuda.
                      is_available() else 'cpu')
    
    print('Device available is', device)
    

    score = train_model(config) # Have training loop in here that outputs loss of model
    return {"val_loss": score}

basin = 'stehekin_r_at_stehekin'

#, search_alg = optuna_search
optuna_tune_config = tune.TuneConfig(scheduler=asha_scheduler)
tune_config = tune.TuneConfig(scheduler=asha_scheduler)
run_config=train.RunConfig(stop= plateau_stopper)


tuner = tune.Tuner(tune.with_resources(tune.with_parameters(objective), resources={"cpu": 1, "gpu": 0}), param_space=config_space, tune_config = tune_config, run_config = run_config) 
best_config = results.get_best_result(metric="val_loss", mode="min").config


# Define the file path where you want to save the best configuration
file_path = f"/data/Hydra_Work/Tuning/Config_Text/{basin}_best_config.txt"

# Open the file in write mode and save the configuration
with open(file_path, "w") as f:
    f.write(str(best_config))

print("Best configuration saved to:", file_path)