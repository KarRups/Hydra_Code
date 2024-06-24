import os
import pickle


# Define the base directory
base_dir = "/data/Hydra_Work/3_Day_No_Forecast_Validation_Models"

# Define the set of Val_Years and Models
val_years = list(range(2000, 2024, 2))  # Even years between 2000 and 2024
models = ["Basin_Head_Model", "General_Head_Model", "General_LSTM_Model", "General_LSTM_No_Flow_Model", "Specific_LSTM_Model"]

# Create the nested folder structure
for year in val_years:
    for model in models:
        dir_path = os.path.join(base_dir, str(year), model)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

# Define the base directory
base_dir = "/data/Hydra_Work/Scaled_Data"

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

# Define dictionaries and DataFrames
dictionaries = {
    'era5': era5,
    'seasonal_forecasts': seasonal_forecasts,
    'daily_flow': daily_flow,
    'climatological_flows': climatological_flows
}

dataframes = {
    'climate_indices': climate_indices,
    'static_variables': static_variables
}

# Save dictionaries
for name, dictionary in dictionaries.items():
    file_path = os.path.join(base_dir, f"{name}.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
    print(f"Saved {name} dictionary to {file_path}")

# Save DataFrames
for name, df in dataframes.items():
    file_path = os.path.join(base_dir, f"{name}.pkl")
    df.to_pickle(file_path)
    print(f"Saved {name} DataFrame to {file_path}")



import shutil
import os
import shutil
# Define the source and destination paths
for test_year in range(2000, 2023, 2):
    source_path = f'/data/Hydra_Work/3_Day_No_Forecast_Validation_Models/{test_year}/General_LSTM_Model/General_LSTM_With_Flags.pth'
    destination_path = f'/data/Hydra_Work/3_Day_No_Forecast_Validation_Models/{test_year}/Flag_LSTM_Model/Flag_LSTM.pth'
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)
    
    source_loss_path = f'/data/Hydra_Work/3_Day_No_Forecast_Validation_Models/{test_year}/General_LSTM_Model/General_LSTM_With_Flags_loss.txt'
    destination_loss_path = f'/data/Hydra_Work/3_Day_No_Forecast_Validation_Models/{test_year}/Flag_LSTM_Model/Flag_LSTM_loss.txt'
    os.makedirs(os.path.dirname(destination_loss_path), exist_ok=True)
    shutil.move(source_loss_path, destination_loss_path)