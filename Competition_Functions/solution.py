import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


from datetime import datetime
import zipfile
import os
import random
import joblib
import calendar

import sys
sys.path.append('/data/gbmc/Rodeo_Submission/Competition_Functions') 
from Processing_Functions import process_forecast_date, process_seasonal_forecasts, fit_fourier_to_h0, Get_History_Statistics
from Data_Transforming import read_nested_csvs, generate_daily_flow, use_USGS_flow_data, USGS_to_daily_df_yearly

sys.path.append('/data/gbmc/Rodeo_Submission/Pipeline_Functions')
from Folder_Work import filter_rows_by_year, csv_dictionary, add_day_of_year_column

def predict(
    site_id: str,
    issue_date: str,
    assets: dict[Hashable, Any],
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,
) -> tuple[float, float, float]:
    """A function that generates a forecast for a single site on a single issue
    date. This function will be called for each site and each issue date in the
    test set.

    Args:
        site_id (str): the ID of the site being forecasted.
        issue_date (str): the issue date of the site being forecasted in
            'YYYY-MM-DD' format.
        assets (dict[Hashable, Any]): a dictionary of any assets that you may
            have loaded in the 'preprocess' function. See next section.
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.
    Returns:
        tuple[float, float, float]: forecasted values for the seasonal water
            supply. The three values should be (0.10 quantile, 0.50 quantile,
            0.90 quantile).
    """
    # Load Models
    issue_date = pd.to_datetime(issue_date)
    device = torch.device('cpu')


    print('downloading models')
    Hydra_Body = torch.load('/data/gbmc/Rodeo_Submission/Models/10_01_Models/General_Body.pth')
    Hydra_Body.to(device)
    general_head_path = '/data/gbmc/Rodeo_Submission/Models/10_01_Models/General_Head.pth'
    site_specific_head_path = f'/data/gbmc/Rodeo_Submission/Model/10_01_Models/{site_id}_Head.pth'

    # Check if the site-specific model file exists
    if os.path.exists(site_specific_head_path):
        # Load the site-specific model
        Basin_Head = torch.load(site_specific_head_path)
        Basin_Head.to(device)
    else:
        # Load the general model if the site-specific file doesn't exist
        General_Head = torch.load(general_head_path)
        General_Head.to(device)

    # Load in data
    era5_folder = '/data/gbmc/Rodeo_Submission/Rodeo_Data/era5'
    flow_folder = '/data/gbmc/Rodeo_Submission/Rodeo_Data/train_monthly_naturalized_flow'

    selected_years = [issue_date.year]
    # csv_dictionary takes in a list of the relevant years to extract, outputs a dictionary of dataframes, one for each year

    era5 = csv_dictionary(era5_folder, [site_id], years=selected_years)
    era5 = add_day_of_year_column(era5)
    era5_basin = era5[f'{site_id}_{issue_date.year}']


    flow = csv_dictionary(flow_folder, [site_id])
    flow = filter_rows_by_year(flow, 1998)
    
    # Now, result_dataframes contains the DataFrames for each basin and year (if specified).
    daily_flow = {}
    # Iterate through the dictionary and apply generate_daily_flow to each DataFrame
    for key, df in flow.items():
        new_row = df.iloc[-1:].copy()
        new_row['month'] = new_row['month'] + 1
        df = df.append(new_row)
        daily_flow[key] = generate_daily_flow(df, persistence_factor=0.4)


    static_indices = pd.read_csv('/data/gbmc/Rodeo_Submission/Rodeo_Data/static_indices.csv', index_col= 'site_id')


    climatology_file_path = '/data/gbmc/Rodeo_Submission/Rodeo_Data/climate_indices.csv'
    climate_indices = pd.read_csv(climatology_file_path)
    climate_indices['date'] = pd.to_datetime(climate_indices['date'])
    climate_indices.set_index('date', inplace = True)
    climate_indices.drop('Unnamed: 0', axis = 1, inplace = True)
    climate_indices = climate_indices[~climate_indices.index.duplicated(keep='first')]

    seasonal_forecasts = {}
    root_folder = '/data/gbmc/Rodeo_Submission/Rodeo_Data/seasonal_forecasts'

    # Iterate through each subfolder and its CSV files
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                # Construct the full path to the CSV file
                filepath = os.path.join(foldername, filename)

                # Extract information from the file path
                path_components = filepath.split('/')
                year, site, month_with_extension = path_components[-3], path_components[-2], path_components[-1]
                month = os.path.splitext(month_with_extension)[0]

                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')

                # Store the DataFrame in the dictionary with the specified key
                key = f"{site}_{year}_{month}"
                seasonal_forecasts[key] = df

    USGS_flow_path = f'/data/gbmc/Rodeo_Submission/Rodeo_Data/USGS_streamflows/{site_id}.csv'
    if os.path.exists(USGS_flow_path):
        USGS_flow = pd.read_csv(USGS_flow_path)

        # Replacing monhtly data for normalised USGS when available
        for key in daily_flow:
            if key in USGS_flow:
                daily_flow_df = daily_flow[key]
                usgs_flow_df = USGS_flow[key]

                # Iterate over years and months in daily_flow_df
                for (year, month), month_data in daily_flow_df.groupby([daily_flow_df.index.year, daily_flow_df.index.month]):
                    # Extract that months data from USGS
                    usgs_month_data = usgs_flow_df.loc[(usgs_flow_df.index.year == year) & (usgs_flow_df.index.month == month), '00060_Mean']
                    
                    # Exclude days with NaN values
                    valid_days = usgs_month_data.dropna().index.day
                    
                    # Check if the month in USGS_flow has data for each day in daily_flow_df excluding NaNs
                    if set(month_data.index.day).issubset(set(valid_days)):
                        # Normalize USGS_flow values for the month excluding NaNs
                        normalization_factor = month_data['daily_flow'].sum()/ usgs_month_data.loc[usgs_month_data.index.day.isin(valid_days)].sum()
                        normalized_usgs_month_data = usgs_month_data * normalization_factor

                        normalized_usgs_month_data.index = normalized_usgs_month_data.index.tz_localize(daily_flow_df.index.tz)
                        
                        # Replace entries in daily_flow_df with normalized USGS_flow values
                        daily_flow_df.loc[(daily_flow_df.index.year == year) & (daily_flow_df.index.month == month), 'daily_flow'] = normalized_usgs_month_data
        
    climatological_basin_flow = pd.read_csv(f'/data/gbmc/Rodeo_Submission/Rodeo_Data/climatological_flows/{basin}.csv')
    Static_variables = pd.read_csv('/data/gbmc/Rodeo_Submission/Rodeo_Data/static_indices.csv', index_col= 'site_id')


    # Normalise this using values also used in training
    climate_scaler_filename = '/data/gbmc/Rodeo_Submission/Rodeo_Data/scalers/climate_normalization_scaler.save'
    climate_scaler = joblib.load(climate_scaler_filename) 
    climate_indices = pd.DataFrame(climate_scaler.transform(climate_indices), columns=climate_indices.columns, index=climate_indices.index)

    era5_scaler_filename = '/data/gbmc/Rodeo_Submission/Rodeo_Data/scalers/era5_scaler.save'
    era5_scaler = joblib.load(era5_scaler_filename) 
    era5 = {key: pd.DataFrame(era5_scaler.transform(df), columns=df.columns, index=df.index) for key, df in era5.items()}

    for basin, df in daily_flow.items(): 
        flow_scaler_filename = f'/data/gbmc/Rodeo_Submission/Rodeo_Data/scalers/flows/{basin}_flow_scaler.save'
        flow_scaler = joblib.load(flow_scaler_filename) 
        daily_flow[basin] = pd.DataFrame(flow_scaler.transform(df), columns=df.columns, index=df.index)

    seasonal_scaler_filename = "/data/gbmc/Rodeo_Submission/Rodeo_Data/scalers/seasonal_scaler.save"
    seasonl_scaler = joblib.load(seasonal_scaler_filename)
    seasonal_forecasts = {key: pd.DataFrame(seasonl_scaler.transform(df), columns=df.columns, index=df.index ) for key, df in seasonal_forecasts.items()}

    static_scaler_filename = '/data/gbmc/Rodeo_Submission/Rodeo_Data/scalers/static_scaler.save'
    static_scaler = joblib.load(static_scaler_filename) 
    Static_variables = pd.DataFrame(static_scaler.transform(Static_variables), columns=Static_variables.columns, index=Static_variables.index)

    static_basin_indices = pd.DataFrame(static_indices.loc[basin]).T

    # Get the start and end dates of the season of interest
    forecast_dates_path = '/data/gbmc/Rodeo_Submission/Rodeo_Data/forecast_dates.csv'
    forecast_dates_df = pd.read_csv(forecast_dates_path)
    forecast_dates = forecast_dates_df[forecast_dates_df['site_id'] == site_id]

    start_season_date = forecast_dates['start_day']
    end_season_date = forecast_dates['end_day']

    start_season_date = pd.to_datetime(forecast_dates['start_day']).tolist()[0]
    start_season_date =  start_season_date.replace(year = issue_date.year)
    end_season_date = pd.to_datetime(forecast_dates['end_day']).tolist()[0]
    end_season_date =  end_season_date.replace(year = issue_date.year)

    era5_basin = era5[f'{basin}_{issue_date.year}']

    Seasonal_Forecasts = process_seasonal_forecasts(seasonal_forecasts, basin, issue_date, end_season_date, columns_to_drop=None)        
    
    try:
        History_H0 = process_forecast_date(daily_flow[basin], era5_basin, climate_indices, issue_date)
    except Exception as e:
        print(f"History error occurred: {e}")
        # Handle the error by doing something else, for example, assigning a default value
        History_H0 = process_forecast_date(daily_flow['animas_r_at_durango'], era5_basin, climate_indices, issue_date)

    Flat_H0 = History_H0.values.flatten()


    # Extract values and column names from the single-row dataframe
    static_values = static_basin_indices.values
    static_values = np.tile(static_values, (len(Seasonal_Forecasts), 1))
    static_columns = static_basin_indices.columns
    # Assign values to new columns in Seasonal_Forecasts
    Seasonal_Forecasts[static_columns] = static_values

    climate_values = climatological_basin_flow[issue_date.dayofyear + 1 : end_season_date.dayofyear + 1].values
    climate_columns = ['Climatology_10', 'Climatology_50', 'Climatology_90']
    Seasonal_Forecasts[climate_columns] = climate_values
    # Create the History with no flow data

    No_Flow_History_H0 = History_H0.drop('daily_flow', axis=1)
    Flat_No_Flow_H0 = No_Flow_History_H0.values.flatten()
    # Convert to float 32 Tensor, originally it is 64
    Flat_H0_tensor= torch.tensor(Flat_H0.astype(np.float32)).to(device)
    
    # Bringing Historical data into forecast
    #Seasonal_Forecasts = fit_fourier_to_h0(No_Flow_History_H0.iloc[:,0:5], Seasonal_Forecasts, initial_guess_terms = 3)
    Seasonal_Forecasts = Get_History_Statistics(No_Flow_History_H0, Seasonal_Forecasts, n_variables = 5)

    Initial_Seasonal_Forecasts_tensor = torch.tensor(Seasonal_Forecasts.values.astype(np.float32)).to(device)
    Flat_No_Flow_H0_tensor = torch.tensor(Flat_No_Flow_H0.astype(np.float32)).to(device)




    Seasonal_Forecasts['In_Season'] = False
    Seasonal_Forecasts.loc[(Seasonal_Forecasts.index > max(start_season_date, issue_date)) & (Seasonal_Forecasts.index <= end_season_date), 'In_Season'] = True

    Seasonal_Forecasts['In_Season'] = Seasonal_Forecasts['In_Season'].astype(int)
    In_Season_tensor = torch.tensor(Seasonal_Forecasts['In_Season'].values.astype(np.float32)).to(device)
    Seasonal_Forecasts_tensor = torch.cat([Initial_Seasonal_Forecasts_tensor, In_Season_tensor.unsqueeze(1)], dim=-1)

    # Get the flow values
    Pre_Flow = 0 
    try:
        pre_season_flow = daily_flow[basin][(daily_flow[basin].index >= start_season_date) & (daily_flow[basin].index <= issue_date)]['daily_flow']

        if not pre_season_flow.empty:
            pre_season_flow = flow_scaler.inverse_transform(pre_season_flow.values.reshape(-1, 1))
            Pre_Flow = np.sum(pre_season_flow)
            Pre_Flow = Pre_Flow* (issue_date-start_season_date).days /len(pre_season_flow)
    except Exception as e:
        pre_season_flow = climatological_basin_flow[(daily_flow.index >= start_season_date) & (climatological_basin_flow.index <= issue_date)][:,1]

    # This needs to be calcuated and done for the batches
    in_season_mask = torch.tensor(Seasonal_Forecasts['In_Season'].to_numpy()).to(device)

    # Process the tensors through the model
    Body_Output = Hydra_Body(Seasonal_Forecasts_tensor, Flat_No_Flow_H0_tensor)

    Climatology =  torch.tensor(climatological_basin_flow[issue_date.dayofyear + 1: end_season_date.dayofyear + 1].values, dtype=torch.float32).to(device)

    # Concatenate body output with forcings
    Head_Input = Body_Output

    #  + Climatology
    if 'Basin_Head' in locals():
        Basin_Head_Output = ((Basin_Head(Head_Input, Flat_H0_tensor) + Climatology) * in_season_mask.unsqueeze(-1) ).detach().cpu().numpy()
    else:
        Basin_Head_Output = ( (General_Head(Head_Input, Flat_No_Flow_H0_tensor) + Climatology) * in_season_mask.unsqueeze(-1) ).detach().cpu().numpy()


    # See if we can inverse transform everythng here
    Basin_Head_Output = flow_scaler.inverse_transform(Basin_Head_Output) + Pre_Flow


    # Model Guesses for remaining seasonal cumulative streamflow
    in_season_mask = Seasonal_Forecasts['In_Season'].to_numpy().nonzero()[0]
    Normal_Vector_Guesses = Basin_Head_Output[in_season_mask]
    Normal_Guesses = np.sum(Normal_Vector_Guesses, axis=0)

    # Add previous flow from that season
    Normal_Guesses = Normal_Guesses + Pre_Flow
    Guesses = Normal_Guesses.reshape(-1, 1)


    Guesses = tuple(Guesses.reshape(1,-1)[0])   

    Climatology  = flow_scaler.inverse_transform(Climatology)
    Guesses = np.sort(Guesses)
    Guesses = np.clip(Guesses, 0, 2*np.sum(Climatology[0]))




    return Guesses






