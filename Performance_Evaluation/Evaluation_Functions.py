
import numpy as np
import pandas as pd
import torch

import calendar
import sys
sys.path.append('/data/gbmc/Rodeo_Submission/Competition_Functions') 
from Processing_Functions import process_forecast_date, process_seasonal_forecasts, fit_fourier_to_h0, Get_History_Statistics
from ML_Functions import Add_Static_To_Series, Process_History, Process_Seasonal_Forecast


def test_performance_for_basin_and_season(basin, Hydra_Body, General_Hydra_Head, model_heads, era5, seasonal_forecasts, daily_flow, climatological_flows, climate_indices, static_indices, device, end_season_date, start_season_date, flow_scaler,  furthest_distance=120, feed_forcing = True):
    Hydra_Body.eval()
    General_Hydra_Head.eval()
    model_heads[f'{basin}'].eval()

    in_season_list = []
    Basin_Guesses = []
    General_Guesses = []
    True_Flows = []

    Default_Predictions = []

    # Convert date strings to datetime objects
    end_season_date = pd.to_datetime(end_season_date)
    start_season_date = pd.to_datetime(start_season_date)
    
    climatological_basin_flow = climatological_flows[basin]
    static_basin_indices = pd.DataFrame(static_indices.loc[basin]).T


    batch_dates = pd.date_range(end=end_season_date - pd.DateOffset(days = 1), periods=furthest_distance, freq='D')
    for forecast_datetime in batch_dates:
        start_forecast_season_date = max(start_season_date, forecast_datetime)
        era5_basin = era5[f'{basin}_{forecast_datetime.year}']

        #Seasonal_Forecasts = process_seasonal_forecasts(seasonal_forecasts, basin, forecast_datetime, end_season_date, columns_to_drop=None)        
        History_H0, No_Flow_History_H0, Flat_H0_tensor, Flat_No_Flow_H0_tensor = Process_History(daily_flow[basin], era5_basin, climate_indices, forecast_datetime, device)
        Seasonal_Forecasts_tensor, in_season_mask = Process_Seasonal_Forecast(seasonal_forecasts, basin, forecast_datetime, end_season_date, static_basin_indices, climatological_basin_flow, No_Flow_History_H0, start_forecast_season_date, device)

        # Get the flow values
        Pre_Flow = 0 

        pre_season_flow = daily_flow[basin][(daily_flow[basin].index >= start_season_date) & (daily_flow[basin].index <= forecast_datetime)]['daily_flow']
        if not pre_season_flow.empty:
            pre_season_flow = flow_scaler.inverse_transform(pre_season_flow.values.reshape(-1, 1))
            Pre_Flow = np.sum(pre_season_flow)

        # This needs to be calcuated and done for the batches
        #in_season_mask = torch.tensor(Seasonal_Forecasts['In_Season'].to_numpy()).to(device)
        in_season_list.append(in_season_mask)


        Season_Flow = daily_flow[basin][(daily_flow[basin].index > forecast_datetime) & (daily_flow[basin].index <= end_season_date)]['daily_flow']
        Season_Flow = torch.tensor(Season_Flow.values).to(device)
        True_Flow = daily_flow[basin][(daily_flow[basin].index > start_forecast_season_date) & (daily_flow[basin].index <= end_season_date)]['daily_flow']
        #True_Flow = (Season_Flow*in_season_mask).unsqueeze(0).detach().cpu().numpy()
        

        True_Flow = flow_scaler.inverse_transform(True_Flow.values.reshape(-1, 1))
        True_Flow = np.sum(True_Flow)

        True_Flow = True_Flow + Pre_Flow

        # Process the tensors through the model
        Body_Output = Hydra_Body(Seasonal_Forecasts_tensor, Flat_No_Flow_H0_tensor)

        Climatology = [0,0,0]
        Climatology =  torch.tensor(climatological_basin_flow[forecast_datetime.dayofyear + 1: end_season_date.dayofyear + 1].values, dtype=torch.float32).to(device)

        # Concatenate body output with forcings
        Head_Input = Body_Output
        if feed_forcing == True:
            Head_Input = torch.cat((Head_Input, Seasonal_Forecasts_tensor), dim=-1)


        #  + Climatology
        Basin_Head_Output = ((model_heads[f'{basin}'](Head_Input, Flat_H0_tensor) + Climatology[:,1].view(len(Season_Flow), 1)) * in_season_mask.unsqueeze(-1) ).detach().cpu().numpy()
        General_Head_Output = ( (General_Hydra_Head(Head_Input, Flat_No_Flow_H0_tensor) + Climatology[:,1].view(len(Season_Flow), 1)) * in_season_mask.unsqueeze(-1) ).detach().cpu().numpy()


        # See if we can inverse transform everythng here
        Basin_Head_Output = flow_scaler.inverse_transform(Basin_Head_Output)
        General_Head_Output = flow_scaler.inverse_transform(General_Head_Output)

        # for i in range(3):
        #     Basin_Head_Output[:,i] = flow_scaler.inverse_transform(Basin_Head_Output[:,i].reshape(-1, 1)).reshape(1, -1)
        #     General_Head_Output[:,i] = flow_scaler.inverse_transform(General_Head_Output[:,i].reshape(-1, 1)).reshape(1, -1)

        Basin_Guesses.append(np.sum( Basin_Head_Output, axis = 0) + Pre_Flow)
        General_Guesses.append(np.sum(General_Head_Output, axis = 0) + Pre_Flow) 

        True_Flows.append(True_Flow)
        if len(climatological_basin_flow[start_forecast_season_date.dayofyear + 1 : end_season_date.dayofyear + 1].values) != 0 :
            
            inverted_climatology = flow_scaler.inverse_transform(climatological_basin_flow[start_forecast_season_date.dayofyear + 1 : end_season_date.dayofyear + 1].values)
            #inverted_climatology = climatological_basin_flow[start_forecast_season_date.dayofyear + 1 : end_season_date.dayofyear].values
            # for i in range(3):
            #     inverted_climatology[:,i] = flow_scaler.inverse_transform(climatological_basin_flow[start_forecast_season_date.dayofyear + 1 : end_season_date.dayofyear].values[:,i].reshape(-1, 1)).reshape(1, -1)
        
        Default_Predictions.append(np.sum(inverted_climatology, axis = 0) + Pre_Flow )
        Specific = np.stack(Basin_Guesses)  
        General =  np.stack(General_Guesses)
        Truth = np.stack(True_Flows)
        Climatology = np.stack(Default_Predictions)




        Climatology_10 = Climatology[:,0]
        Climatology_50 = Climatology[:,1]
        Climatology_90 = Climatology[:,2]

    return Specific, General, Truth, Climatology_10, Climatology_50, Climatology_90
