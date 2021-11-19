#####################################################################################################################################
#######################################             REQUIRED MODULES INSTALLATION       #############################################
## UNCOMMENT IF SOME PACKAGES ARE MISSING ##
# import subprocess
# import sys
# list_of_packages=['datetime','h5py','keras','matplotlib','numpy','pandas','pathlib','pmdarima','prettytable','python-math','python-time', 'requests','scikit-learn', 'sktime', 'statistics', 'tensorflow','seaborn','zipfile','yfinance']
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# for package in list_of_packages:
#     install(package)
#####################################################################################################################################
#####################################           SET SEED TO ENSURE REPRODUCIBILITY      #############################################
seed_value=12345
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy
numpy.random.seed(seed_value)
import tensorflow
tensorflow.random.set_seed(seed_value)
#####################################################################################################################################
###############################################         IMPORT MAIN SCRIPTS             #############################################
import main_univariate
import main_multivariate
## IMPORT MODULES AND DEFINE PATHS ##
import pandas as pd
import datetime
import yfinance as yf
parent_folder='Financial Time Series'
series_name='S&P 500 Trading Volume'
output_variable='^GSPC'
##LOAD DATA##
url = 'https://drive.google.com/file/d/1ljcV2HDDQQwx_L1nJLbG5_SPNE1ur5uH/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df_multivariate=pd.read_csv(path, index_col=['Date'], sep=',')
df_multivariate.index=pd.to_datetime(df_multivariate.index).date
#####################################################################################################################################
######################################                UNIVARIATE ANALYSIS               #############################################
#####################################################################################################################################
df_univariate=df_multivariate[output_variable].to_frame()
#####################################################################################################################################
#####################################################################################################################################
######################################   GRID SEARCH TO DETERMINE BEST PARAMETER CONFIGURATIONS     #################################
#####################################################################################################################################
#############################################       LIST OF PARAMETERS           ####################################################
lags = [5, 10, 30]                  # use higher values ONLY for larger datasets (e.g. > 1000 observations in the test set)
nodes1 = [10, 50, 100, 200]
nodes2 = [10, 50, 100, 200]
optimizer = ['adam']
epochs = [500]                      # model checkpoint is enabled - large value is selected to allow validation loss to decrease
batch = [32, 64, 128, 256]
filters1 = [32, 64, 128]
filters2 = [32, 64, 128]
cnn_nodes = [10, 50, 100, 200]
################################################        GRID SEARCH FUNCTIONS        ################################################
#main_univariate.grid_search_ffnn_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name, 
#                                            lags=lags, nodes1=nodes1, nodes2=nodes2, optimizer=optimizer,
#                                            epochs=epochs, batch=batch)
#main_univariate.grid_search_rnn_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name,
#                                           lags=lags, nodes1=nodes1, nodes2=nodes2, optimizer=optimizer,
#                                           epochs=epochs, batch=batch)
#main_univariate.grid_search_lstm_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name,
#                                            lags=lags, nodes1=nodes1, nodes2=nodes2, optimizer=optimizer,
#                                            epochs=epochs, batch=batch)
#main_univariate.grid_search_gru_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name,
#                                           lags=lags, nodes1=nodes1, nodes2=nodes2, optimizer=optimizer,
#                                           epochs=epochs, batch=batch)
#main_univariate.grid_search_cnn_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name,
#                                           lags=lags, filters1=filters1, filters2=filters2, cnn_nodes=cnn_nodes,
#                                           optimizer=optimizer, epochs=epochs, batch=batch)
#####################################################################################################################################
#####################################################################################################################################
###################################     PRODUCE UNIVARIATE OUTPUT (OPTIMIZED PARAMETERS)      #######################################
#####################################################################################################################################
##DATA ANALYSIS##
main_univariate.data_analysis_univariate(df_univariate, parent_folder=parent_folder, series_name=series_name)
##ARIMA##
main_univariate.arima(df_univariate, parent_folder=parent_folder, series_name=series_name)
##FEED FORWARD NEURAL NETWORK (FFNN)##
univariate_ffnn_training_time, ffnn_metrics_train, ffnn_metrics_test, ffnn_layers = main_univariate.ffnn_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=5, T=1,
ffnn_nodes1=10, ffnn_nodes2=200, ffnn_epochs=500, ffnn_batch=256, ffnn_optimizer='adam')
##RECURRENT NEURAL NETWORK (RNN)##
univariate_rnn_training_time, rnn_metrics_train, rnn_metrics_test, rnn_layers = main_univariate.rnn_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=10, T=1,
rnn_nodes1=100, rnn_nodes2=100, rnn_epochs=500, rnn_batch=256, rnn_optimizer='adam')
##LONG SHORT-TERM MEMORY (LSTM)##
univariate_lstm_training_time, lstm_metrics_train, lstm_metrics_test, lstm_layers = main_univariate.lstm_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=30, T=1, 
lstm_nodes1=10, lstm_nodes2=50, lstm_epochs=500, lstm_batch=256, lstm_optimizer='adam')
##GATED RECURRENT UNIT (GRU)##
univariate_gru_training_time, gru_metrics_train, gru_metrics_test, gru_layers = main_univariate.gru_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=30, T=1,
gru_nodes1=10, gru_nodes2=10, gru_epochs=500, gru_batch=128, gru_optimizer='adam')
##CONVOLUTIONAL NEURAL NETWORK (CNN)##
univariate_cnn_training_time, cnn_metrics_train, cnn_metrics_test, cnn_layers = main_univariate.cnn_univariate(
df_univariate, parent_folder=parent_folder, series_name=series_name, L=30, T=1,
cnn_filters_1=128, cnn_filters_2=32,cnn_kernel_1=2,cnn_kernel_2=2,cnn_dense_nodes=200,cnn_pool_size=2,cnn_epochs=500,cnn_batch=64,
cnn_optimizer='adam')
##PRINT UNIVARIATE TABLES##
main_univariate.univariate_tables(df_univariate, univariate_ffnn_training_time, univariate_rnn_training_time, 
                       univariate_lstm_training_time, univariate_gru_training_time, univariate_cnn_training_time,
                       ffnn_metrics_train, rnn_metrics_train, lstm_metrics_train, gru_metrics_train, cnn_metrics_train,
                       ffnn_metrics_test, rnn_metrics_test, lstm_metrics_test, gru_metrics_test, cnn_metrics_test,
                       ffnn_layers, rnn_layers, lstm_layers, gru_layers, cnn_layers,
                       parent_folder=parent_folder, series_name=series_name)
#####################################################################################################################################
######################################                MULTIVARIATE ANALYSIS               #############################################
#################################################################################################################################### 
###################################################################################################################################
################################################     GRID SEARCH FUNCTIONS        ################################################
#main_multivariate.grid_search_ffnn_multivariate(df_multivariate, parent_folder=parent_folder, series_name=series_name,
#                                                output_variable=output_variable, lags=lags, nodes1=nodes1, nodes2=nodes2, 
#                                                optimizer=optimizer, epochs=epochs, batch=batch)
#main_multivariate.grid_search_rnn_multivariate(df_multivariate, parent_folder=parent_folder, series_name=series_name,
#                                                output_variable=output_variable, lags=lags, nodes1=nodes1, nodes2=nodes2,
#                                                optimizer=optimizer,epochs=epochs, batch=batch)
#main_multivariate.grid_search_lstm_multivariate(df_multivariate, parent_folder=parent_folder, series_name=series_name,
#                                                output_variable=output_variable, lags=lags, nodes1=nodes1, nodes2=nodes2, 
#                                                optimizer=optimizer, epochs=epochs, batch=batch)
#main_multivariate.grid_search_gru_multivariate(df_multivariate, parent_folder=parent_folder, series_name=series_name, 
#                                               output_variable=output_variable, lags=lags, nodes1=nodes1, nodes2=nodes2,
#                                               optimizer=optimizer, epochs=epochs, batch=batch)
#main_multivariate.grid_search_cnn_multivariate(df_multivariate, parent_folder=parent_folder, series_name=series_name,
#                                                output_variable=output_variable, lags=lags, filters1=filters1, filters2=filters2, 
#                                                cnn_nodes=cnn_nodes, optimizer=optimizer, epochs=epochs, batch=batch)
#####################################################################################################################################
#############################            PRODUCE MULTIVARIATE OUTPUT (OPTIMIZED PARAMETERS)      ####################################
#####################################################################################################################################
##DATA ANALYSIS##
main_multivariate.data_analysis_multivariate(df_multivariate, parent_folder=parent_folder, 
series_name=series_name, output_variable=output_variable)
##ARIMAX##
main_multivariate.arimax(df_multivariate, parent_folder=parent_folder, 
output_variable=output_variable, series_name=series_name)
#FEED FORWARD NEURAL NETWORK (FFNN)##
multivariate_ffnn_training_time, ffnn_metrics_train, ffnn_metrics_test, ffnn_layers = main_multivariate.ffnn_multivariate(
df_multivariate, parent_folder=parent_folder, series_name=series_name, 
output_variable=output_variable, L=10, T=1, ffnn_nodes1=200, ffnn_nodes2=50, ffnn_epochs=500, ffnn_batch=128, ffnn_optimizer='adam')
##RECURRENT NEURAL NETWORK (RNN)##
multivariate_rnn_training_time, rnn_metrics_train, rnn_metrics_test, rnn_layers = main_multivariate.rnn_multivariate(
df_multivariate, parent_folder=parent_folder, series_name=series_name, 
output_variable=output_variable, L=5, T=1, rnn_nodes1=10, rnn_nodes2=50, rnn_epochs=500, rnn_batch=32, rnn_optimizer='adam')
##LONG SHORT-TERM MEMORY (LSTM)##
multivariate_lstm_training_time, lstm_metrics_train, lstm_metrics_test, lstm_layers = main_multivariate.lstm_multivariate(
df_multivariate, parent_folder=parent_folder, series_name=series_name, 
output_variable=output_variable, L=5, T=1, lstm_nodes1=10, lstm_nodes2=100, lstm_epochs=500, lstm_batch=32, lstm_optimizer='adam')
##GATED RECURRENT UNIT (GRU)##
multivariate_gru_training_time, gru_metrics_train, gru_metrics_test, gru_layers = main_multivariate.gru_multivariate(
df_multivariate, parent_folder=parent_folder, series_name=series_name, 
output_variable=output_variable, L=5, T=1, gru_nodes1=10, gru_nodes2=10, gru_epochs=500, gru_batch=64, gru_optimizer='adam')
##CONVOLUTIONAL NEURAL NETWORK (CNN)##
multivariate_cnn_training_time, cnn_metrics_train, cnn_metrics_test, cnn_layers = main_multivariate.cnn_multivariate(
df_multivariate, parent_folder=parent_folder, series_name=series_name, 
output_variable=output_variable, L=10, T=1,cnn_filters_1=128, cnn_filters_2=64, cnn_kernel_1=3, cnn_kernel_2=3, 
cnn_dense_nodes=50, cnn_pool_size=2, cnn_epochs=500, cnn_batch=256, cnn_optimizer='adam')
##PRINT MULTIVARIATE TABLES##
main_multivariate.multivariate_tables(df_multivariate, multivariate_ffnn_training_time, multivariate_rnn_training_time, 
                      multivariate_lstm_training_time, multivariate_gru_training_time, multivariate_cnn_training_time,
                      ffnn_metrics_train, rnn_metrics_train, lstm_metrics_train, gru_metrics_train, cnn_metrics_train,                       ffnn_metrics_test, rnn_metrics_test, lstm_metrics_test, gru_metrics_test, cnn_metrics_test,
                      ffnn_layers, rnn_layers, lstm_layers, gru_layers, cnn_layers,
                       parent_folder=parent_folder, series_name=series_name)
#####################################################################################################################################