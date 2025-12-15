# Hydra Model: Optimizing Loss Functions for Hydrological Forecasting

Code accompanying the paper "Hydra-LSTM: A semi-shared Machine Learning architecture for prediction across Watersheds" published in *Artificial Intelligence for the Earth Systems* (AIES-D-24-0103.1).

## Overview

This repository implements the **Hydra-LSTM model**, a novel semi-shared machine learning architecture designed to improve river discharge forecasting across multiple watersheds. The model addresses a critical challenge in hydrological ML: how to leverage data from many catchments while still accommodating catchment-specific variables and maintaining prediction accuracy.

### Key Innovation

The Hydra-LSTM uses a dual-head architecture:
- **Hydra Body**: A shared LSTM encoder that processes variables available across all catchments (e.g., ERA5 reanalysis data)
- **Hydra Heads**: 
  - **Multi-Catchment Head**: Trained across all catchments for general predictions
  - **Single-Catchment Head**: Trained on individual catchments to incorporate catchment-specific data (e.g., local river discharge measurements)

This architecture allows the model to:
- Benefit from multi-catchment training for generalization
- Incorporate bespoke local data without retraining the entire model
- Predict uncertainty quantiles (10% and 90%) for next-day discharge

## Key Features

- Semi-shared architecture combining global and local information
- 2-day ahead probabilistic discharge forecasts (uncertainty quantiles)
- Easy integration of catchment-specific variables
- State-of-the-art performance vs. traditional single/multi-catchment LSTMs
- Modular design for operational forecasting systems

## Installation

```bash
git clone https://github.com/KarRups/Hydra_Code.git
cd Hydra_Code
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- xarray
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn
- See `requirements.txt` for complete dependencies

## Project Structure

```
Hydra_Code/
├── Models/                        # LSTM model architectures
│   ├── Hydra_LSTM.py             # Main Hydra-LSTM implementation
│   └── baseline_models.py        # Comparison models
├── Pipeline_Functions/           # Data processing pipeline
│   ├── data_loader.py
│   └── preprocessing.py
├── Training/                     # Training scripts
│   ├── train_hydra.py
│   ├── hyperparameter_tuning.py
│   └── cross_validation.py
├── Performance_Evaluation/       # Model evaluation
│   ├── metrics.py
│   └── visualization.py
├── Competition_Functions/        # Benchmark comparisons
├── Tuning/                       # Hyperparameter optimization
├── notebooks/                    # Analysis notebooks
│   ├── Competiton_Training.ipynb
│   ├── Week_Ahead_Evaluation.ipynb
│   └── Scaling_Data.ipynb
└── requirements.txt
```

## Data Requirements

The Hydra-LSTM is trained on:

1. **CARAVAN Dataset**: Multi-basin discharge observations and static catchment attributes
   - 2,450+ catchments across multiple continents
   - [Download from CARAVAN project](https://github.com/kratzert/Caravan)

2. **ERA5 Reanalysis**: Historical meteorological forcing
   - Temperature, precipitation, radiation, etc.
   - Available from Copernicus Climate Data Store

3. **Catchment-specific data** (optional):
   - Local discharge measurements
   - Regional weather station data
   - Can be easily added via Single-Catchment Head

### Data Setup

Due to size constraints, meteorological and discharge data are not included. You will need to:

1. Download CARAVAN basin data
2. Download ERA5 reanalysis data
3. Update paths in configuration files
4. Run preprocessing scripts to prepare training data

See `DATA_SETUP.md` for detailed instructions.

## Quick Start

### Training a Hydra-LSTM Model

```python
from Models.Hydra_LSTM import HydraLSTM
from Pipeline_Functions.data_loader import load_caravan_data

# Load data for multiple catchments
train_data = load_caravan_data(basin_list, start_date='2000-01-01', end_date='2015-12-31')

# Initialize Hydra-LSTM
model = HydraLSTM(
    input_size_shared=10,  # Number of shared meteorological variables
    input_size_specific=1,  # Number of catchment-specific variables
    hidden_size=128,
    num_layers=2
)

# Train model
model.train(train_data, epochs=100, learning_rate=0.001)
```

### Making Predictions

```python
# Predict with Multi-Catchment Head (no local data)
q_10, q_50, q_90 = model.predict(test_data, use_specific_data=False)

# Predict with Single-Catchment Head (with local discharge)
q_10, q_50, q_90 = model.predict(test_data, use_specific_data=True)

# q_10 q_50, q_90 are 10%, 50%, and 90% quantile predictions
```

### Example Workflow

See `notebooks/Competiton_Training.ipynb` for a complete example including:
- Data loading and preprocessing
- Model training and hyperparameter tuning
- Evaluation against baseline methods
- Visualization of results

## Model Architectures

### Hydra Body (Shared Encoder)
Processes globally available meteorological variables (ERA5 data) into a shared encoding space. Uses LSTM layers to capture temporal dependencies.

### Multi-Catchment Head
Trained across all catchments to learn general rainfall-runoff relationships. Can make predictions for any catchment using only shared variables.

### Single-Catchment Head
Trained on individual catchments to incorporate catchment-specific information (e.g., previous discharge). Combines shared encoding with local data for improved accuracy.

## Evaluation Metrics

The model is evaluated using the Cumulative Quantile Efficiency Score, a skill score comparing the sum of quantiles losses for the model against a climatological quantile forecast

Results show Hydra-LSTM achieves:
- Better median and quantile predictions than single-catchment LSTMs
- Improved generalization vs. multi-catchment LSTMs
- Ability to leverage catchment-specific data without full retraining

## Loss Functions

The model uses quantile regression loss to predict the 10th and 90th percentiles:

```
L_quantile(y, ŷ_τ) = Σ max(τ(y - ŷ_τ), (τ-1)(y - ŷ_τ))
```

where τ ∈ {0.1, 0.9} are the target quantiles.

## Hyperparameter Tuning

Key hyperparameters:
- Hidden size: [64, 128, 256]
- Number of layers: [1, 2, 3]
- Learning rate: [0.0001, 0.001, 0.01]
- Dropout: [0.0, 0.2, 0.4]

See `Tuning/` directory for optimization scripts using Optuna or grid search.

## Citation

If you use this code in your research, please cite:

```
Hydra-LSTM: A semi-shared Machine Learning architecture for prediction across Watersheds
Artificial Intelligence for the Earth Systems
DOI: 10.1175/AIES-D-24-0103.1
```

## Contributing

This code is provided for research reproducibility. For questions or issues, please open a GitHub issue or contact [your email].

## License

MIT License - see LICENSE file for details

## Acknowledgments

- CARAVAN dataset: Kratzert et al. (2023)
- ERA5 reanalysis: Copernicus Climate Change Service
- Computational resources: [Your institution/facility]

## Authors


---

**Note for Users**: This is research code developed for a specific study. Paths and configurations will need adjustment for your environment. The model is designed for next-day forecasting; extension to longer lead times would require modifications to the architecture and training procedure.
