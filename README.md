# Stock Price Prediction Project

Author: Grafton Cook

Contact: grafton.cook@gmail.com

## Table of Contents
- [Stock Price Prediction Project](#stock-price-prediction-project)
  - [Table of Contents](#table-of-contents)
    - [Project Overview](#project-overview)
      - [Time Series Components](#time-series-components)
      - [Traditional Analysis](#traditional-analysis)
      - [Deep Learning](#deep-learning)
      - [Hybrid Model](#hybrid-model)
  - [Directory Structure](#directory-structure)
  - [Setup \& Installation](#setup--installation)
    - [Clone this repository](#clone-this-repository)
    - [Install dependencies](#install-dependencies)
    - [Jupyter Notebooks](#jupyter-notebooks)
  - [Usage \& Examples](#usage--examples)
  - [Results \& Analysis](#results--analysis)
  - [Next Steps](#next-steps)
  - [License \& Acknowledgments](#license--acknowledgments)

### Project Overview
This repository contains a multi-part time series forecasting project focused on predicting stock prices for a financial institution. You will see both classical and deep learning modeling approaches, as well as a final hybrid model that combines the best of both worlds. Specifically, the project is divided into four sub-projects:

#### Time Series Components
- Decompose and examine the time series (trend, seasonality, stationarity).
- Perform ACF, PACF, and ADF tests.

#### Traditional Analysis
- Apply classical forecasting models (Moving Average, Exponential Smoothing, AR, ARIMA).
- Compare results using RMSE and data visualizations.

#### Deep Learning
- Prepare data for deep learning.
- Implement RNN/LSTM models for time series forecasting.
- Compare performance against classical methods.

#### Hybrid Model
- Combine classical and deep learning approaches.
- Evaluate performance on real-world stock data.

## Directory Structure
```text
stock-price-prediction/
├── 1-time-series-components/
│   ├── data/
│   ├── notebooks/
│   │   ├── EDA.ipynb
│   │   └── stationarity_tests.ipynb
│   └── scripts/
├── 2-traditional-analysis/
│   ├── data/
│   ├── notebooks/
│   │   ├── traditional_analysis.ipynb
│   └── scripts/
├── 3-deep-learning/ **(TBD)**
│   ├── data/
│   ├── notebooks/
│   └── scripts/
├── 4-hybrid-model/ **(TBD)**
│   ├── data/
│   ├── notebooks/
│   └── scripts/
├── requirements.txt
└── README.md
```

`part-x/`: Each sub-project directory with its own notebooks, data, and scripts.
`environment.yml` or `requirements.txt`: Conda or pip environment details.

## Setup & Installation
### Clone this repository
```sh
git clone https://github.com/tacotuesday/time-series-stock-forecasting.git
cd time-series-stock-forecasting
```
### Install dependencies
Using Conda:
```sh
conda create --name stock-forecasting-env --file requirements.txt
conda activate stock-forecasting-env
```
Or using Pip:
```sh
pip install -r requirements.txt
```
### Jupyter Notebooks
Ensure Jupyter is installed and launch notebooks:
```sh
jupyter notebook
```
Navigate to the relevant sub-project under `part-x/notebooks`.

## Usage & Examples
- Data: Sample stock price data is located in each data/ folder (or instructions to download from a public source).
- Model Training: Run the notebooks in chronological order to see how the time series is analyzed and modeled.
- Hyperparameter Tuning: Some notebooks contain sections for adjusting hyperparameters (e.g., ARIMA `p/d/q`, LSTM architecture).

## Results & Analysis
- Comparisons: We compare RMSE, MSE, and/or MAE across different models.
- Visualizations: Time series plots, predicted vs. actual, residual analysis.
- Insights: The best-performing model (in this dataset) is typically the hybrid approach, though performance may vary depending on the data.

## Next Steps
- Enhance the hybrid model by experimenting with other neural network architectures (e.g., Transformers).
- Add real-time inference or streaming pipelines for updated price data.
- Extend the code to additional financial instruments (crypto, bonds, etc.).

## License & Acknowledgments
License: MIT License.

Acknowledgments:
This project is part of a Manning LiveProject.
Datasets from Alpha Vantage.