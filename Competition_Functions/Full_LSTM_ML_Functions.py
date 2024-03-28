import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler



from collections import defaultdict


import sys
sys.path.append('/data/gbmc/Rodeo_Submission/Competition_Functions') 
from Processing_Functions import process_forecast_date, process_seasonal_forecasts, fit_fourier_to_h0, Get_History_Statistics


class Hindcast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False):
        super(Hindcast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
        
        h0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)

        if len(np.shape(x)) == 3:
            h0 = h0.view( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
            c0 = c0.view( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
        # else:
        #     h0 = h0.view( self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        #     c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
            
        out, (hn, cn) = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out, hn, cn

class Forecast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False):
        super(Forecast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x, h0, c0):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
   
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out
    
class Google_LSTMModel(nn.Module):
  def __init__(self, hindcast,forecast):
    super(Google_LSTMModel, self).__init__()
    self.hindcast = hindcast
    self.forecast = forecast
    
  def forward(self, history, forecasts):
    
    # get states from hindcast model
    # need to decide whether the head recieves the raw history or an encoding of it
    hind_out, hn,cn = self.hindcast(history)
    
    # get forecasts from forecast model
    out = self.forecast(forecasts, hn,cn)
    return out, hind_out

def Google_Model_Block(hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout = 0.0, bidirectional = False):
    # For now dropout and bidirectional aren't included here, can change that down the line
    # output_size for Hindcast doesn't actually matter
    Hindcast = Hindcast_LSTM_Block(hindcast_input_size, hidden_size, num_layers, hindcast_output_size, dropout = dropout, bidirectional = bidirectional)
    Forecast = Forecast_LSTM_Block(forecast_input_size, hidden_size, num_layers, forecast_output_size, dropout = dropout, bidirectional = bidirectional)
    Block = Google_LSTMModel(Hindcast, Forecast)
    Block.to(device)

    return Block


def Specific_Heads(basins, hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout = 0.0, bidirectional = False):
    model_heads = {}
    for basin in basins:
        basin_hindcast = Hindcast_LSTM_Block(hindcast_input_size, hidden_size, num_layers, hindcast_output_size, dropout = dropout, bidirectional = bidirectional)
        basin_forecast = Forecast_LSTM_Block(forecast_input_size, hidden_size, num_layers, forecast_output_size, dropout = dropout, bidirectional = bidirectional)
    
        model_heads[f'{basin}'] = Google_LSTMModel(basin_hindcast, basin_forecast)
        model_heads[f'{basin}'].to(device)
    return model_heads



    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False, Sequence_Target = False):
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.Sequence_Target = Sequence_Target

        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2


    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        if not self.Sequence_Target:
            out = out[-1,:]  # Take the output from the last time step
        out = self.fc(out) 
        return out

class SumPinballLoss(nn.Module):
    def __init__(self, quantiles = [0.1,0.5,0.9]):
        super(SumPinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, observed, modeled):
        # Initialize a list to store losses for each output
        output_losses = []
        observed = observed.squeeze()

        #modeled = torch.sum(modeled, dim = 1)
 
        # Calculate the quantile loss for each output and quantile
        for i, quantile in enumerate(self.quantiles):

            modeled_quantile = modeled[...,i]
            loss = torch.nanmean(torch.max(quantile * torch.nansum(observed - modeled_quantile), (quantile - 1) * torch.nansum(observed - modeled_quantile) ))

        
            output_losses.append(loss)

        # Sum the losses for each output and quantile
        overall_loss = sum(output_losses)
        
        return overall_loss
    


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



# Model Running Code
# Key difference here is that we won't need to restrict the history length to 90, it can be variable

def Prepare_Batch(Dates, indices):
    batch_dates = Dates[indices.cpu().numpy().astype(int)]
    min_day = min(180 - forecast_datetime.dayofyear for forecast_datetime in batch_dates) + 1
    random_day = np.random.choice(range(round(0.4 * min_day), min_day + 1))
    proportion_first_value = 0.05
    choices = [min_day]* int(proportion_first_value * 100) + [random_day]* int( (1 - proportion_first_value) * 100)
    final_forcing_distance = np.random.choice(choices)
    return batch_dates, final_forcing_distance

def Prepare_Basin(basins, climatological_flows, static_indices, final_forcing_distance, basin_usage_counter, basin_count):
    basin = np.random.choice(basins)
    climatological_basin_flow = climatological_flows[basin]
    static_basin_indices = pd.DataFrame(static_indices.loc[basin]).T
    basin_key = f"{basin}_{final_forcing_distance}"
    basin_usage_counter[basin_key] += 1
    basin_count[f"{basin}"] += 1
    return basin, climatological_basin_flow, static_basin_indices, basin_usage_counter


def Get_Relevant_Dates(forecast_datetime, final_forcing_distance, group_lengths):
    forecast_length = np.random.choice(group_lengths)
    end_season_date = forecast_datetime + pd.DateOffset(days = final_forcing_distance)
    start_season_date = end_season_date - pd.DateOffset(days=forecast_length)
    start_forecast_season_date = max(start_season_date, forecast_datetime)
    
    return start_season_date, start_forecast_season_date, end_season_date

def Add_Static_To_Series(static_indices, Series):
    # Include Static Values
    static_values = static_indices.values
    static_values = np.tile(static_values, (len(Series), 1))
    static_columns = static_indices.columns
    Series[static_columns] = static_values
    return Series

def Process_History(daily_flow_basin, era5_basin, climate_indices, forecast_datetime, device, offset = 90):
    History_H0 = process_forecast_date(daily_flow_basin, era5_basin, climate_indices, forecast_datetime, offset)
    No_Flow_History_H0 = History_H0.drop('daily_flow', axis=1)
    Flat_H0 = History_H0.values.flatten()
    Flat_No_Flow_H0 = No_Flow_History_H0.values.flatten()
    
    Flat_H0_tensor = torch.tensor(Flat_H0.astype(np.float32)).to(device)
    Flat_No_Flow_H0_tensor = torch.tensor(Flat_No_Flow_H0.astype(np.float32)).to(device)
    
    return History_H0, No_Flow_History_H0, Flat_H0_tensor, Flat_No_Flow_H0_tensor

def Process_Seasonal_Forecast(seasonal_forecasts, basin, forecast_datetime, end_season_date, static_basin_indices, climatological_basin_flow, No_Flow_History_H0, start_forecast_season_date, device):
    Seasonal_Forecasts = process_seasonal_forecasts(seasonal_forecasts, basin, forecast_datetime, end_season_date, columns_to_drop=None)
    #Seasonal_Forecasts = fit_fourier_to_h0(No_Flow_History_H0.iloc[:,0:5], Seasonal_Forecasts, initial_guess_terms = 4)
    Seasonal_Forecasts = Get_History_Statistics(No_Flow_History_H0, Seasonal_Forecasts, n_variables=5)
    Seasonal_Forecasts = Add_Static_To_Series(static_basin_indices, Seasonal_Forecasts)

    climate_values = climatological_basin_flow[forecast_datetime.dayofyear + 1 : end_season_date.dayofyear + 1].values
    climate_columns = ['Climatology_10', 'Climatology_50', 'Climatology_90']
    Seasonal_Forecasts[climate_columns] = climate_values

    Seasonal_Forecasts['In_Season'] = False
    Seasonal_Forecasts.loc[(Seasonal_Forecasts.index > start_forecast_season_date) & (Seasonal_Forecasts.index <= end_season_date), 'In_Season'] = True
    Seasonal_Forecasts['In_Season'] = Seasonal_Forecasts['In_Season'].astype(int)
    in_season_mask = torch.tensor(Seasonal_Forecasts['In_Season'].to_numpy()).to(device)

    Seasonal_Forecasts_tensor = torch.tensor(Seasonal_Forecasts.values.astype(np.float32)).to(device)
    return Seasonal_Forecasts_tensor, in_season_mask

def Calculate_Flow_Data(daily_flow, climatological_basin_flow, basin, start_season_date, forecast_datetime, end_season_date, in_season_mask, device):
    pre_season_flow = daily_flow[basin][(daily_flow[basin].index >= start_season_date) & (daily_flow[basin].index < forecast_datetime)]['daily_flow']
    Pre_Flow = np.sum(pre_season_flow)

    Season_Flow = daily_flow[basin][(daily_flow[basin].index > forecast_datetime) & (daily_flow[basin].index <= end_season_date)]['daily_flow']
    Season_Flow = torch.tensor(Season_Flow.values).to(device)
    True_Flow = torch.sum(Season_Flow * in_season_mask).unsqueeze(0)

    True_Flow = True_Flow + Pre_Flow
    Pre_Flow = torch.tensor(Pre_Flow).unsqueeze(0).to(device)

    Climatology =  torch.tensor(climatological_basin_flow[forecast_datetime.dayofyear + 1 : end_season_date.dayofyear + 1].values, dtype=torch.float32).to(device)
    return Pre_Flow, True_Flow, Season_Flow, Climatology

def Calculate_Head_Outputs(Hydra_Body, General_Hydra_Head, model_heads, basin, Forcing_List_torch, No_Flow_List_torch, H_List_torch, feed_forcing):
    Body_Output, _ = Hydra_Body(No_Flow_List_torch, Forcing_List_torch) 
    Head_Input = Body_Output

    if feed_forcing:
        Head_Input = torch.cat((Head_Input, Forcing_List_torch), dim=-1)

    print('Hindcast', np.shape(H_List_torch))
    print('Forecast', np.shape(Head_Input))
    Basin_Head_Output, _ = model_heads[f'{basin}'](H_List_torch, Head_Input)
    General_Head_Output, _ = General_Hydra_Head(No_Flow_List_torch, Head_Input)
    return Basin_Head_Output, General_Head_Output

# This should be fine. but maybe isn't with how models are defined
def Calculate_Losses_and_Predictions(Output, Climatology_list_torch, in_season_list_torch, Season_Flow_List_torch, Season_Flow, criterion, batch_size):
    Guesses = (Output + Climatology_list_torch[...,1].view(batch_size, len(Season_Flow), 1)) * in_season_list_torch.unsqueeze(-1)
    Climatology_Guesses = Climatology_list_torch * in_season_list_torch.unsqueeze(-1)

    Guesses = torch.sum(Guesses, dim=1)
    Climatology_Guesses = torch.sum(Climatology_Guesses, dim=1)
    Season_Flow_List_torch = torch.sum(Season_Flow_List_torch, dim=1)

    loss = criterion(Season_Flow_List_torch, Guesses)
    Climatology_loss = criterion(Season_Flow_List_torch, Climatology_Guesses)

    return loss, Climatology_loss


def Model_Run(All_Dates, basins, Hydra_Body, General_Hydra_Head, model_heads, era5, daily_flow, climatological_flows, climate_indices, seasonal_forecasts, static_indices, optimizer, scheduler, criterion, early_stopper = None, n_epochs = 20, batch_size = 2, group_lengths = [89, 90, 91, 92] , Train_Mode=True, device = 'cpu', feed_forcing = True):
    basin_usage_counter = defaultdict(int)
    basin_count = defaultdict(int)

    specific_losses, general_losses, climate_losses = [], [], []

    Size = len(All_Dates)

    # Set models to train mode if Train_Mode is True, else set to evaluation mode
    Hydra_Body.train(Train_Mode)
    General_Hydra_Head.train(Train_Mode)
    [model_heads[f'{basin}'].train(Train_Mode) for basin in basins]

    try:
        for epoch in range(n_epochs):
            specific_loss, general_loss = 0, 0
            Climate_loss = 0
            permutation = torch.randperm(len(All_Dates))   
            # print(basin_count)
            for i in range(0, Size, batch_size):
                indices = permutation[i:i + batch_size]
                batch_dates, final_forcing_distance = Prepare_Batch(All_Dates, indices)
                basin, climatological_basin_flow, static_basin_indices, basin_usage_counter = Prepare_Basin(basins, climatological_flows, static_indices, final_forcing_distance, basin_usage_counter, basin_count)

                
                
                H_List, No_Flow_List, Forcing_List, True_Flow_List, Pre_Flow_List, in_season_list, Season_Flow_List = [[] for _ in range(7)]
                Climatology_list = []

                for forecast_datetime in batch_dates:
                    start_season_date, start_forecast_season_date, end_season_date = Get_Relevant_Dates(forecast_datetime, final_forcing_distance, group_lengths)
                    
                    era5_basin = era5[f'{basin}_{forecast_datetime.year}']
                    History_H0, No_Flow_History_H0, Flat_H0_tensor, Flat_No_Flow_H0_tensor = Process_History(daily_flow[basin], era5_basin, climate_indices, forecast_datetime, device)
                    Seasonal_Forecasts_tensor, in_season_mask = Process_Seasonal_Forecast(seasonal_forecasts, basin, forecast_datetime, end_season_date, static_basin_indices, climatological_basin_flow, No_Flow_History_H0, start_forecast_season_date, device)
                    Pre_Flow, True_Flow, Season_Flow, Climatology = Calculate_Flow_Data(daily_flow, climatological_basin_flow, basin, start_season_date, forecast_datetime, end_season_date, in_season_mask, device)

                    History_H0_Tensor = torch.tensor(History_H0.values.astype(np.float32)).to(device)
                    No_Flow_History_H0_Tensor = torch.tensor(No_Flow_History_H0.values.astype(np.float32)).to(device)
    

                    H_List, No_Flow_List, Forcing_List, True_Flow_List, Pre_Flow_List, Season_Flow_List = [ H_List + [History_H0_Tensor], No_Flow_List + [No_Flow_History_H0_Tensor], Forcing_List + [Seasonal_Forecasts_tensor],
                        True_Flow_List + [True_Flow], Pre_Flow_List + [Pre_Flow], Season_Flow_List + [Season_Flow] ]
                    in_season_list.append(in_season_mask)
                    Climatology_list = Climatology_list + [Climatology]

                H_List_torch, No_Flow_List_torch, Forcing_List_torch, True_Flow_List_torch, Pre_Flow_List_torch, in_season_list_torch, Season_Flow_List_torch = [
                    torch.stack(lst, dim=0) for lst in [H_List, No_Flow_List, Forcing_List, True_Flow_List, Pre_Flow_List, in_season_list, Season_Flow_List] ]
                
                Climatology_list_torch = torch.stack(Climatology_list, dim = 0)

                optimizer.zero_grad()   
                Basin_Head_Output, General_Head_Output = Calculate_Head_Outputs(Hydra_Body, General_Hydra_Head, model_heads, basin, Forcing_List_torch, No_Flow_List_torch, H_List_torch, feed_forcing)

                loss_general, Climatology_loss = Calculate_Losses_and_Predictions(General_Head_Output, Climatology_list_torch, in_season_list_torch, Season_Flow_List_torch, Season_Flow, criterion, batch_size)
                loss_specific, _ = Calculate_Losses_and_Predictions(Basin_Head_Output, Climatology_list_torch, in_season_list_torch, Season_Flow_List_torch, Season_Flow, criterion, batch_size)

                if Train_Mode:
                    loss = loss_general + loss_specific
                    loss.backward(retain_graph=True)
                    optimizer.step() 
                    scheduler.step()

                specific_loss += loss_specific.item() 
                general_loss += loss_general.item()
                Climate_loss += Climatology_loss.item()

            if early_stopper != None:               
                # Maybe I should instead define the earlystopper by how its comparison with climatology   
                if early_stopper.early_stop(0.5*(specific_loss + general_loss) - Climate_loss):
                    return general_losses, specific_losses, climate_losses

            print(f'Epoch {epoch + 1}: {"Training" if Train_Mode else "Validation"} Mode')
            print('general difference :', (general_loss - Climate_loss)/Size , '\nspecific difference:', (specific_loss- Climate_loss)/Size)
            print('Climatology loss:', Climate_loss/Size)
            general_losses.append(general_loss/Size) ; climate_losses.append(Climate_loss/Size) ; specific_losses.append(specific_loss/Size)
        return general_losses, specific_losses, climate_losses

    except KeyboardInterrupt:
        return s


def No_Body_Model_Run(All_Dates, basins, model_heads, era5, daily_flow, climatological_flows, climate_indices, seasonal_forecasts, static_indices, optimizer, scheduler, criterion, early_stopper = None, n_epochs = 20, batch_size = 2, group_lengths = [89, 90, 91, 92] , Train_Mode=True, device = 'cpu', specialised = True):
    basin_usage_counter = defaultdict(int)
    basin_count = defaultdict(int)

    Overall_losses, climate_losses = [], []

    Size = len(All_Dates)

    # Set models to train mode if Train_Mode is True, else set to evaluation mode
    if specialised:
        [model_heads[f'{basin}'].train(Train_Mode) for basin in basins]
    else:
        model_heads.train(Train_Mode)
    try:
        for epoch in range(n_epochs):
            Overall_loss = 0
            Climate_loss = 0
            permutation = torch.randperm(len(All_Dates))   
            #print(basin_count)
            for i in range(0, Size, batch_size):
                indices = permutation[i:i + batch_size]
                batch_dates, final_forcing_distance = Prepare_Batch(All_Dates, indices)
                basin, climatological_basin_flow, static_basin_indices, basin_usage_counter = Prepare_Basin(basins, climatological_flows, static_indices, final_forcing_distance, basin_usage_counter, basin_count)
                

                
                
                H_List, No_Flow_List, Forcing_List, True_Flow_List, Pre_Flow_List, in_season_list, Season_Flow_List = [[] for _ in range(7)]
                Climatology_list = []

                for forecast_datetime in batch_dates:
                    start_season_date, start_forecast_season_date, end_season_date = Get_Relevant_Dates(forecast_datetime, final_forcing_distance, group_lengths)
                    
                    era5_basin = era5[f'{basin}_{forecast_datetime.year}']
                    History_H0, No_Flow_History_H0, Flat_H0_tensor, Flat_No_Flow_H0_tensor = Process_History(daily_flow[basin], era5_basin, climate_indices, forecast_datetime, device)
                    Seasonal_Forecasts_tensor, in_season_mask = Process_Seasonal_Forecast(seasonal_forecasts, basin, forecast_datetime, end_season_date, static_basin_indices, climatological_basin_flow, No_Flow_History_H0, start_forecast_season_date, device)
                    Pre_Flow, True_Flow, Season_Flow, Climatology = Calculate_Flow_Data(daily_flow, climatological_basin_flow, basin, start_season_date, forecast_datetime, end_season_date, in_season_mask, device)

                    History_H0_Tensor = torch.tensor(History_H0.values.astype(np.float32)).to(device)
                    No_Flow_History_H0_Tensor = torch.tensor(No_Flow_History_H0.values.astype(np.float32)).to(device)
    

                    H_List, No_Flow_List, Forcing_List, True_Flow_List, Pre_Flow_List, Season_Flow_List = [ H_List + [History_H0_Tensor], No_Flow_List + [No_Flow_History_H0_Tensor], Forcing_List + [Seasonal_Forecasts_tensor],
                        True_Flow_List + [True_Flow], Pre_Flow_List + [Pre_Flow], Season_Flow_List + [Season_Flow] ]
                    in_season_list.append(in_season_mask)
                    Climatology_list = Climatology_list + [Climatology]

                H_List_torch, No_Flow_List_torch, Forcing_List_torch, True_Flow_List_torch, Pre_Flow_List_torch, in_season_list_torch, Season_Flow_List_torch = [
                    torch.stack(lst, dim=0) for lst in [H_List, No_Flow_List, Forcing_List, True_Flow_List, Pre_Flow_List, in_season_list, Season_Flow_List] ]
                
                Climatology_list_torch = torch.stack(Climatology_list, dim = 0)

                optimizer.zero_grad()   

                if specialised:
                    Basin_Head_Output, _ = model_heads[f'{basin}'](H_List_torch, Forcing_List_torch)
                else:
                    Basin_Head_Output, _ = model_heads(H_List_torch, Forcing_List_torch)

                loss, Climatology_loss = Calculate_Losses_and_Predictions(Basin_Head_Output, Climatology_list_torch, in_season_list_torch, Season_Flow_List_torch, Season_Flow, criterion, batch_size)

                if Train_Mode:
                    loss.backward(retain_graph=True)
                    optimizer.step() 
                    scheduler.step()

                Overall_loss += loss.item() 
                Climate_loss += Climatology_loss.item()


            Overall_losses.append(Overall_loss/Size) ; climate_losses.append(Climate_loss/Size)
            if early_stopper != None:               
                # Maybe I should instead define the earlystopper by how its comparison with climatology   
                if early_stopper.early_stop(Overall_loss - Climate_loss):
                    return Overall_losses, climate_losses

            # print(f'Epoch {epoch + 1}: {"Training" if Train_Mode else "Validation"} Mode')
            # print('loss difference :', (Overall_loss - Climate_loss)/Size)
            # print('Climatology loss:', Climate_loss/Size)
           
        return Overall_losses, climate_losses

    except KeyboardInterrupt:
        return s



