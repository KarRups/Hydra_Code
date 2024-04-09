import os
import random
import calendar

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skew





# Test and works
# def read_seasonal_indices(basin, forecast_month, year = 2000):

#     forecast_month = calendar.month_abbr[forecast_month].lower()
#     file_path = f"/data/gbmc/Rodeo_Submission/Rodeo_Data/seasonal_forecasts/{year}/{basin}/{forecast_month}.csv"
    
#     try:
#         df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
#         return df
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#         return None


# Works, have to choose if the forecast day information is contained in the historical data or the future predictions
def get_H0(history, forecast_year, forecast_month, forecast_day = 1):
    forecast_datetime = pd.to_datetime(f"{forecast_year}-{forecast_month}-{forecast_day}")

    start_date_h0 = forecast_datetime - pd.DateOffset(days=90)
    end_date_h0 = forecast_datetime
    H0 = history[(history.index >= start_date_h0) & (history.index < end_date_h0)]
    return H0

# Seems to work
def get_forecast_dates(flow, forecast_year, max_allowed_month=None): 
    # Check and works
    """    
    Returns a list of forecast dates for a given year, NOT forecast year

    Parameters:
    - flow: DataFrame with datetime index containing river flow data. Ideally daily dates
    - forecast_year: Year for which forecast dates are required.

    Returns:
    - forecast_dates: List of datetime objects representing forecast dates.
    """

    filtered_flow = flow[flow.index.year == forecast_year]

    if max_allowed_month is not None:
        # Filter data to consider only dates before the specified month
        filtered_flow = filtered_flow[filtered_flow.index.month < max_allowed_month]

    available_months = filtered_flow.index.month.unique()

    forecast_dates = []
    for forecast_month in np.unique(available_months):
        month_flow = filtered_flow[filtered_flow.index.month == forecast_month]
        days_in_forecast_month = month_flow.index.day.unique()

        for forecast_day in days_in_forecast_month:
            forecast_datetime = pd.to_datetime(f"{forecast_year}-{forecast_month}-{forecast_day}")
            forecast_dates.append(forecast_datetime)

    return forecast_dates

    
def process_forecast_date(flow, history, climate_indices, forecast_datetime, offset = 90):
    """
    Processes a specific forecast date to prepare historical data and forecast variables.

    Parameters:
    - flow: DataFrame with datetime index containing river flow data.
    - history: DataFrame containing historical data for that river.
    - basin: Name of the river basin.
    - forecast_datetime: Datetime object representing the forecast date.

    Returns:
    - History_H0: Historical data for the LSTM model.
    - Forecast_Variables: Variables for the LSTM model's forecast.
    """

    History_H0 = get_H0(history, forecast_datetime.year, forecast_datetime.month, forecast_datetime.day)

    # For google LSTM change this so start+date is just first available date
    start_date = forecast_datetime - pd.DateOffset(days=offset)
    Past_Flow = flow.loc[start_date:forecast_datetime - pd.DateOffset(days=1)]['daily_flow']
    Past_Climatology = climate_indices.loc[start_date:forecast_datetime - pd.DateOffset(days=1)]

    History_H0 = pd.merge(History_H0, Past_Flow, left_index=True, right_index=True, how='inner')
    History_H0 = pd.merge(History_H0, Past_Climatology, left_index=True, right_index=True, how='inner')



    return History_H0


def process_seasonal_forecasts(seasonal_forecasts, basin, forecast_datetime, end_season_date, columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = []

    S_forecast_month = calendar.month_abbr[forecast_datetime.month].lower()
    
    seasonal_key = f'{basin}_{forecast_datetime.year}_{S_forecast_month}'
    Seasonal_Forecasts = seasonal_forecasts[seasonal_key]
    
    # Drop specified columns
    Seasonal_Forecasts = Seasonal_Forecasts.drop(columns=columns_to_drop, errors='ignore')
    
    # Take relevant columns and filter by date
    Seasonal_Forecasts = Seasonal_Forecasts[
        (Seasonal_Forecasts.index > forecast_datetime) &
        (Seasonal_Forecasts.index <= end_season_date)
    ].copy()


    
    
    # Add 'day_of_year' column
    Seasonal_Forecasts['sin_day_of_year'] = np.sin(2*np.pi * Seasonal_Forecasts.index.dayofyear / 365)
    Seasonal_Forecasts['cos_day_of_year'] = np.cos(2*np.pi * Seasonal_Forecasts.index.dayofyear / 365)
    
    return Seasonal_Forecasts

# Getting history in as a Fourier Forecast

# Define the Fourier series function
def fourier_series(x, *params, data_range = 90):
    result = params[0]  # DC component
    for i, period in enumerate([1,7,30]):
        result += params[i] * np.cos(2 * np.pi * period * x / data_range)
        result += params[i + 1] * np.sin(2 * np.pi * period * x / data_range)

    # Adding a linear term
    result += params[-1] * x / data_range

    return result


def fit_fourier_to_h0(history_h0_df, seasonal_forecasts_df, initial_guess_terms = 4, columns_to_drop = None):
    fourier_parameters = {}
    history_h0_df = history_h0_df.add_prefix('extended_')
    seasonal_forecasts_df_copy = seasonal_forecasts_df.copy()

    if columns_to_drop is None:
        columns_to_drop = []

    for column in history_h0_df.columns:
        if column not in columns_to_drop:
            x_data = np.arange(len(history_h0_df))
            y_data = history_h0_df[column]

            # Fit the Fourier series
            initial_guess = [np.mean(y_data)] + [0] * 2 * initial_guess_terms  # Adjust the number of terms as needed
            params, _ = curve_fit(fourier_series, x_data, y_data, p0=initial_guess)
            fourier_parameters[column] = params


    # Apply Fourier components to the forecast
    for column, params in fourier_parameters.items():
        x_forecast = np.arange(len(history_h0_df), len(history_h0_df) + len(seasonal_forecasts_df_copy))
        
        # Update the corresponding columns with Fourier components
        for i, period in enumerate([1,7,30]):
            seasonal_forecasts_df_copy[f'{column}_fourier_cos_{period}'] = params[2*i] * np.cos(2 * np.pi * period * x_forecast / len(x_forecast))
            seasonal_forecasts_df_copy[f'{column}_fourier_sin_{period}'] = params[2*i + 1] * np.sin(2 * np.pi * period * x_forecast / len(x_forecast))
    
    return seasonal_forecasts_df_copy

def Get_History_Statistics(History, Seasonal, n_variables = 6):
    statistics_df = pd.DataFrame()
    length = len(Seasonal)

    # Calculate and record statistics for each column
    for column_name in History.iloc[:,0:n_variables].columns:
        data = History[column_name].values
        
        # Calculate variance
        variance_value = np.var(data)
        
        # Calculate skewness
        skewness_value = skew(data)
        
        coefficients = np.polyfit(np.arange(len(data)), data, 1)

        # The first coefficient represents the slope (gradient) of the line
        gradient, intercept = coefficients
        
        # Record the statistics in the new DataFrame
        statistics_df[column_name + '_intercept'] = [intercept]
        #statistics_df[column_name + '_Variance'] = [variance_value]
        #statistics_df[column_name + '_Skewness'] = [skewness_value]
        statistics_df[column_name + '_Gradient'] = [gradient]

    # Repeat the statistics_df DataFrame along its index
    statistics_df = statistics_df.loc[statistics_df.index.repeat(length)].reset_index(drop=True)

    # Assign the statistics to corresponding columns in Seasonal
    for column in statistics_df.columns:
        Seasonal[column] = statistics_df[column].values

    return Seasonal
