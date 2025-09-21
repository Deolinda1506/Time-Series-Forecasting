# Air Quality Forecasting using LSTM Networks

A comprehensive machine learning project for predicting PM2.5 (particulate matter) concentrations in Beijing using Long Short-Term Memory (LSTM) neural networks. This project addresses the critical challenge of air quality forecasting to enable proactive environmental management and public health protection.

## Project Overview

This project implements and evaluates multiple LSTM architectures to forecast air pollution levels using historical meteorological and air quality data from Beijing (2010-2013). The goal is to achieve accurate PM2.5 predictions with RMSE below 4000 on the Kaggle leaderboard.

### Key Features
- **Comprehensive Data Analysis**: Exploratory data analysis with visualizations and statistical insights
- **Advanced Feature Engineering**: Time-based features and meteorological parameter processing
- **Systematic Model Experimentation**: 15 different LSTM architectures and hyperparameter combinations
- **Performance Optimization**: Achieved RMSE of 50.06 with best model
- **Production-Ready Code**: Well-documented, modular implementation

##  Dataset

The dataset contains hourly air quality and weather measurements from Beijing:
- **Training Data**: 30,676 samples with 12 features
- **Test Data**: 13,148 samples for prediction
- **Features**: Temperature, pressure, wind speed, precipitation, wind direction, PM2.5
- **Time Period**: 2010-2013 with hourly granularity

### Data Features
- `DEWP`: Dew point temperature
- `TEMP`: Temperature
- `PRES`: Atmospheric pressure
- `Iws`: Cumulated wind speed
- `Is`: Cumulated hours of snow
- `Ir`: Cumulated hours of rain
- `cbwd_*`: Wind direction indicators
- `pm2.5`: Target variable (PM2.5 concentration)

##  Model Architecture

### Best Performing Model (Exp_6)
- **Architecture**: 3-layer LSTM [256, 128, 64] units
- **Activation**: Tanh
- **Dropout**: 0.2
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Performance**: RMSE = 50.06

### Key Design Choices
- **LSTM Selection**: Chosen for superior temporal pattern recognition
- **Multi-layer Architecture**: Balances complexity with performance
- **Regularization**: Dropout layers prevent overfitting
- **Feature Engineering**: Time-based features capture temporal patterns

## Quick Start

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/[username]/Time-Series-Forecasting-1.git
cd Time-Series-Forecasting
```

2. Install dependencies:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

3. Download the dataset from [Kaggle Competition](https://www.kaggle.com/competitions/assignment-1-time-series-forecasting-septemb-2025)

### Usage
1. Place dataset files in the `data/` directory:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

2. Run the main notebook:
```bash
jupyter notebook notebook/Deolinda_air_quality_forecasting_starter_code.ipynb
```

3. Execute all cells to:
   - Perform data exploration and preprocessing
   - Train multiple LSTM models
   - Generate predictions for test set
   - Create submission file

## 📈 Results

### Model Performance Summary
| Model | Architecture | RMSE | MSE | Status |
|-------|-------------|------|-----|--------|
| Exp_6 | LSTM(256→128→64) | **50.06** | 2505.81 |  Best |
| Exp_15 | LSTM(96→48) | 51.96 | 2700.22 |  Good |
| Exp_9 | LSTM(192→96→48→24) | 52.46 | 2752.39 | Good |
| Model 3 | LSTM(64→32) | 64.56 | 4167.80 |  Baseline |

### Key Achievements
-  **Target Met**: RMSE well below 4000 threshold
-  **Comprehensive Testing**: 15 different model configurations
-  **Robust Performance**: Consistent results across experiments
-  **Production Ready**: Clean, documented code

##  Project Structure

```
Time-Series-Forecasting-1/
├── data/                           # Dataset files
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Test data
│   └── sample_submission.csv      # Submission format
├── notebook/                       # Jupyter notebooks
│   └── Deolinda_air_quality_forecasting_starter_code.ipynb
├── submissions/                    # Generated predictions
│   ├── submission.csv             
│   ├── submission1.csv            
│   ├── submission2.csv            
│   ├── submission3.csv            
│   ├── submission4.csv            
│   ├── submission5.csv            
│   └── submission6.csv            
├──
└── README.md                      # This file
```

## Methodology

### Data Preprocessing
1. **Missing Value Handling**: Mean imputation for PM2.5 missing values
2. **Feature Engineering**: Time-based features (hour, day, month, weekend)
3. **Data Scaling**: MinMaxScaler normalization
4. **Reshaping**: 3D format for LSTM input

### Model Development
1. **Architecture Design**: Multi-layer LSTM with dropout regularization
2. **Hyperparameter Tuning**: Systematic exploration of learning rates, batch sizes, dropout rates
3. **Training Strategy**: Adam optimizer with early stopping
4. **Evaluation**: RMSE and MSE metrics

### Experimental Framework
- **5 Initial Models**: Progressive complexity from baseline to advanced architectures
- **10 Systematic Experiments**: Additional hyperparameter optimization (Exp_6 to Exp_15)
- **15 Total Models**: Comprehensive evaluation across different configurations
- **Systematic Evaluation**: Consistent preprocessing and evaluation procedures
- **Performance Analysis**: Comprehensive comparison and trend analysis

##  Visualizations

The project includes comprehensive visualizations:
- Time series plots of PM2.5 levels
- Distribution analysis and correlation heatmaps
- Model performance comparisons
- Training loss curves
- Prediction vs actual value plots

##  Academic Context

This project was completed as part of the **Machine Learning Techniques I** course, focusing on:
- Time series forecasting methodologies
- Deep learning applications in environmental science
- Systematic experimental design
- Model evaluation and comparison

##  References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*.
2. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
3. World Health Organization. (2021). WHO global air quality guidelines.


