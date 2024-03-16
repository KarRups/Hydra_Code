import shutil
import pandas as pd
import numpy as np
import zipfile
import os

def remove_redundant_folders(root_folder):
    for year_folder in os.listdir(root_folder):
        year_folder_path = os.path.join(root_folder, year_folder)

        if os.path.isdir(year_folder_path):
            # Get the list of subfolders within the year folder
            subfolders = os.listdir(year_folder_path)

            # Check if there is only one subfolder and it's a directory
            if len(subfolders) == 1 and os.path.isdir(os.path.join(year_folder_path, subfolders[0])):
                # Get the path of the redundant subfolder
                redundant_folder_path = os.path.join(year_folder_path, subfolders[0])

                # Move the contents of the redundant subfolder to the year folder
                for item in os.listdir(redundant_folder_path):
                    item_path = os.path.join(redundant_folder_path, item)
                    target_path = os.path.join(year_folder_path, item)
                    shutil.move(item_path, target_path)

                # Remove the redundant subfolder
                shutil.rmtree(redundant_folder_path)

                print(f"Removed redundant subfolder: {redundant_folder_path}")

# Example usage
# root_folder = '/data/gbmc/Data/Real_Data/seasonal_indices/'

# remove_redundant_folders(root_folder)


def extract_from_main_zip(zip_folder_path, extracted_folder_path):
    """
    Extracts contents from a main zip folder containing nested zip files.

    Parameters:
    - zip_folder_path (str): Path to the main zip folder.
    - extracted_folder_path (str): Directory to extract the contents of the zip files.
    """

    with zipfile.ZipFile(zip_folder_path, 'r') as main_zip_ref:
        for item in main_zip_ref.namelist():
            item_path = os.path.join(extracted_folder_path, item)
            folder_name = os.path.splitext(os.path.basename(item_path))[0]
            target_folder_path = os.path.join(extracted_folder_path, folder_name)
            os.makedirs(target_folder_path, exist_ok=True)

            with main_zip_ref.open(item) as zip_file:
                with zipfile.ZipFile(zip_file, 'r') as inner_zip_ref:
                    inner_zip_ref.extractall(target_folder_path)

            print(f'ZIP file {item} extracted to: {target_folder_path}')


# # Path to the zip folder
# zip_folder_path = '/data/gbmc/Data/Real_Data/seasonal_forecasts.zip'

# # Directory to extract the contents of the zip files
# extracted_folder_path = '/data/gbmc/Data/Real_Data/'

# zip_file_path = '/data/gbmc/Data/Real_Data/2000.zip'
# extracted_folder_path = 'data/gbmc/Data/Real_Data/2000'

def extract_from_individual_zip(zip_file_path):
    """
    Extracts contents from an individual zip file.

    Parameters:
    - zip_file_path (str): Path to the individual zip file.
    - extracted_folder_path (str): Directory to extract the contents of the zip file.
    """

    folder_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    target_folder_path = os.path.join(os.path.dirname(zip_file_path), folder_name)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder_path)

    print(f'ZIP file extracted to: {target_folder_path}')



def csv_dictionary(folder_path, basins, years=None):
    """
    Reads and returns DataFrames from CSV files for each basin, specified years, and data type.

    Parameters:
    - folder_path (str): Path to the main folder containing CSV files.
    - basins (list): List of basin names.
    - years (list, optional): List of years to consider. If None, all available years are considered.

    Returns:
    - dataframes (dict): Dictionary containing DataFrames for each basin.
    """
    dataframes = {}

    for basin in basins:
        for year in years or [None]:
            # Determine the folder path based on the year
            year_folder_path = os.path.join(folder_path, str(year) if year else '')
            file_path = os.path.join(year_folder_path, f'{basin}.csv')

            # Check if the file exists
            if os.path.exists(file_path):
                # Create DataFrame
                df = pd.read_csv(file_path)

                # Additional processing for the DataFrame

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns='Unnamed: 0')

                # Save DataFrame to the dictionary
                dataframes[f'{basin}_{year}' if year else basin] = df
            else:
                print(f"Warning: File not found for basin '{basin}' and year '{year}' at path: {file_path}")

    return dataframes

# # Example usage
# folder_path = '/data/gbmc/Rodeo_Submission/Rodeo_Data/era5'

# selected_years = [2000, 2022]  # Optional: Specify the years you want to consider

# result_dataframes = csv_dictionary(folder_path, basins, years=selected_years)

# Now, result_dataframes contains the DataFrames for each basin and year (if specified).



def filter_rows_by_year(dataframes, threshold_year):
    """
    Remove rows with 'year' below the specified threshold in each DataFrame.

    Parameters:
    - dataframes (dict): Dictionary of DataFrames.
    - threshold_year (int): Threshold year for filtering.

    Returns:
    - filtered_dataframes (dict): Dictionary containing DataFrames after filtering rows.
    """
    filtered_dataframes = {}

    for key, df in dataframes.items():
        # Check if 'year' column exists in the DataFrame
        if 'year' in df.columns:
            # Filter rows based on the 'year' column
            filtered_df = df[df['year'] >= threshold_year]

            # Save the filtered DataFrame to the dictionary
            filtered_dataframes[key] = filtered_df
        else:
            print(f"Warning: 'year' column not found in DataFrame '{key}', skipping filtering.")

    return filtered_dataframes


def add_day_of_year_column(dataframes):
    """
    Add a 'day_of_year' column to each DataFrame in the dictionary based on the 'year' column.

    Parameters:
    - dataframes (dict): Dictionary of DataFrames.

    Returns:
    - dataframes_with_day_of_year (dict): Dictionary containing DataFrames with the 'day_of_year' column added.
    """
    dataframes_with_day_of_year = {}

    for key, df in dataframes.items():

        # Add 'day_of_year' column based on the 'year' column
        df['day_of_year'] = np.sin(2*np.pi * df.index.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2*np.pi * df.index.dayofyear / 365)
        
         # Save the DataFrame with the new column to the dictionary
        dataframes_with_day_of_year[key] = df
    return dataframes_with_day_of_year

def scale_dataframes(df_dict, csv_path):
    # Step 1: Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(df_dict.values(), ignore_index=True)
    # Step 2: Use MinMaxScaler to fit and transform the concatenated DataFrame
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(concatenated_df)
    # Step 3: Store the scaling parameters in a DataFrame
    scaling_params = pd.DataFrame({'min': scaler.data_min_, 'max': scaler.data_max_}, index=concatenated_df.columns)
    # Save the scaling parameters to a CSV file
    scaling_params.to_csv(csv_path)


    return scaling_params

# era5_normalise_path = '/data/gbmc/Rodeo_Submission/Rodeo_Data/era5/normalising_values.csv'
# scale_dataframes(era5, era5_normalise_path)