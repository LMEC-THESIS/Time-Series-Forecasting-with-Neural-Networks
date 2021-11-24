# Time-Series-Forecasting-with-Neural-Networks

This project seeks to compare the performance of neural network models on time series forecasting tasks, with a specific focus on macroeconomic and financial time series.

The __main_univariate.py__ and __main_multivariate.py__ master files contain all the necessary functions to run the remaining scripts. The *univariate* script computes basic summary statistics and data analysis on the given series (ADF tests, plots, autocorrelation functions...), then fits two baseline optimized linear models to the data (Random Walk and ARIMA). 

Finally, the script fits five neural network models to the series, namely Feed-Forward, Simple RNN, LSTM, GRU and CNN networks, whose hyperparameters have been optimized by means of a grid-search algorithm. The *multivariate* script follows the same routine but using an ARIMAX specification for the linear model and multivariate inputs in the neural netowrks. All the models are set to provide one-step-ahead forecasts, while the amount of lags varies based on model specification.  After all the models have been fitted, summary tables are printed to compare forecasting performance. 

The two master files contain one function for each model specification and one for each grid search algorithm. The remaining scripts import the functions from the master files and apply them to different datasets.

The __macroeconomic_time_series.py__ script loads a series of YoY % change in unemployment level in the United States from July 1955 to December 2019 (monthly observations) and fits the above models with optimized parameters. The dataset also includes the same YoY % change in CPI, Federal Funds Rate and Industrial production as exogenous regressors in the multivariate models. 
[Data Source: FRED](https://fred.stlouisfed.org/)

The __financial_time_series.py__ scripts loads a series of daily trading volume for the S&P 500 index from January 4th, 2010 to December 30th, 2020 (in millions of shares) and likewise fits both linear and neural network models to the data. In the multivariate case, daily trading volumes from FTSE 100 and NASDAQ Composite are added as regressors, together with daily observations from the CBOE Volatility Index: VIX. 
[Data Source: Yahoo Finance](https://finance.yahoo.com/)

Finally, the __control_time_series.py__ script provides a control time series to check performance against. This is a dataset on daily visits to a [statistical forecasting website](https://people.duke.edu/~rnau/411home.htm) from September 14, 2014 to August 19, 2020. The dependent variable is "Daily Visits", while the regressors are "Unique Visits", "First Time Visits" and "Returning Visits". 
[Data Source: Kaggle](https://www.kaggle.com/bobnau/daily-website-visitors)

The full output is available on [Google Drive](https://drive.google.com/drive/folders/1qKfEdbrJvsy0zYAFOuwb3FWuDVsuvFIe). Results may vary when scripts are run on machines with different CPUs, but within-machine consistency is assured by setting a random seed within each function.

The same analyses can be run on *any series* of choice, simply by importing the master files and calling the required functions. 

The only dataset requirements are:
- The data being in *DataFrame* format (pandas) and each column being a *float* object
- The index being a *DatetimeIndex* without the time component (this can be removed by appending .date() to the line setting the index)
- The date in the *DatetimeIndex* being formatted with hyphens "-" but __not slashes__ "/", as this results in an error when saving the plots.
