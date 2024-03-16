import numpy as np
import pandas as pd
import os
import calendar

# Used for seasonal forecasts
def read_nested_csvs(root_folder):
    data = {}

    # Iterate through each subfolder and its CSV files
    for foldername, _, filenames in os.walk(root_folder):
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
                data[key] = df

    return data


def generate_daily_flow(flow_df, prop_variance=0.1, persistence_factor=0.6):
    daily_df = pd.DataFrame()

    for index, row in flow_df.iterrows():
        end_date = pd.to_datetime(f"{row['year']}-{row['month']}-1") + pd.DateOffset(months=1)
        # Calculate the number of days in the month
        days_in_month = (end_date - pd.to_datetime(f"{row['year']}-{row['month']}-1")).days

        # Calculate the mean daily flow
        mean_daily_flow = row['volume'] / days_in_month

        # Generate daily flow values exponentially distributed with a slightly lower scale for reduced variance
        daily_flow = np.random.exponential(scale=mean_daily_flow * prop_variance, size=days_in_month)

        # Adjust the generated values to ensure the sum is equal to the total volume
        daily_flow = daily_flow / np.sum(daily_flow) * prop_variance * row['volume']
        daily_flow = daily_flow + mean_daily_flow * (1 - prop_variance)

        # Add persistence to the generated values
        for i in range(1, days_in_month):
            daily_flow[i] = (persistence_factor * daily_flow[i - 1]) + ((1 - persistence_factor) * daily_flow[i])

        # Adjust the generated values to ensure the sum is equal to the total volume
        daily_flow = daily_flow / np.sum(daily_flow) * row['volume']

        # Create a new DataFrame for the month, may want to consider forecast_year vs year
        month_df = pd.DataFrame({
            'date': pd.date_range(start=f"{row['year']}-{row['month']}-1", periods=days_in_month, freq='D'),
            'daily_flow': daily_flow
        })

        # Append the month DataFrame to the daily DataFrame
        daily_df = pd.concat([daily_df, month_df], ignore_index=True)

        daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df.set_index('date', inplace = True)
    return daily_df


def use_USGS_flow_data(daily_flow, USGS_flow):
    for key in daily_flow:
        if key in USGS_flow:
            daily_flow_df = daily_flow[key]
            usgs_flow_df = USGS_flow[key]

            # Iterate over years and months in daily_flow_df
            for (year, month), month_data in daily_flow_df.groupby([daily_flow_df.index.year, daily_flow_df.index.month]):
                # Extract that month's data from USGS
                usgs_month_data = usgs_flow_df.loc[(usgs_flow_df.index.year == year) & (usgs_flow_df.index.month == month), '00060_Mean']
                
                # Exclude days with NaN values
                valid_days = usgs_month_data.dropna().index.day
                
                # Check if the month in USGS_flow has data for each day in daily_flow_df excluding NaNs
                if set(month_data.index.day).issubset(set(valid_days)):
                    # Normalize USGS_flow values for the month excluding NaNs
                    normalization_factor = month_data['daily_flow'].sum() / usgs_month_data.loc[usgs_month_data.index.day.isin(valid_days)].sum()
                    normalized_usgs_month_data = usgs_month_data * normalization_factor

                    normalized_usgs_month_data.index = normalized_usgs_month_data.index.tz_localize(daily_flow_df.index.tz)
                    
                    # Replace entries in daily_flow_df with normalized USGS_flow values
                    daily_flow_df.loc[(daily_flow_df.index.year == year) & (daily_flow_df.index.month == month), 'daily_flow'] = normalized_usgs_month_data

    return daily_flow


def USGS_to_daily_df_yearly(daily_flow, path, name, normalising_path):
    daily_flow_df = daily_flow['animas_r_at_durango']
    new_dataframe = pd.read_csv(path, parse_dates=['datetime'])
    new_dataframe['datetime'] = pd.to_datetime(new_dataframe['datetime'])
    new_dataframe.set_index('datetime', inplace=True)
    new_dataframe.index = new_dataframe.index.tz_localize(daily_flow_df.index.tz)
    common_dates = set(daily_flow_df.index)
    new_dataframe = new_dataframe[new_dataframe.index.isin(common_dates)]
    new_dataframe = new_dataframe.rename(columns={'00060_Mean': 'daily_flow'})

    yearly_sum = new_dataframe.groupby(new_dataframe.index.year)['daily_flow'].sum()
    normalising_data = pd.read_csv(normalising_path)
    # Merge the yearly_sum with the CSV data on the 'year' column
    merged_data = pd.merge(normalising_data, yearly_sum, left_on='year', right_index=True)
    merged_data['factor'] = (merged_data['volume'] / merged_data['daily_flow'].sum())

    # Extract unique years from the datetime index
    unique_years = new_dataframe.index.year.unique()

    # Select odd years and increase them by one
    adjusted_years = [year + 1 if year % 2 != 0 else year for year in unique_years]

    # Create a new DataFrame with adjusted years
    adjusted_dataframe = pd.DataFrame(index=new_dataframe.index)
    adjusted_dataframe['adjusted_year'] = new_dataframe.index.year.map(dict(zip(unique_years, adjusted_years)))

    # Merge new_dataframe with merged_data on the 'year' column
    result_dataframe = pd.merge(adjusted_dataframe, merged_data[['year', 'factor']], left_on=adjusted_dataframe['adjusted_year'], right_on='year')

    # Multiply 'daily_flow' by the corresponding 'factor', .values handles index errors
    new_dataframe['daily_flow'] = new_dataframe['daily_flow'].values * result_dataframe['factor'].values

    daily_flow[name] = new_dataframe


