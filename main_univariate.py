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
#############################################   UNIVARIATE SCRIPT  (DATA ANALYSIS)   ################################################
def data_analysis_univariate(df, parent_folder, series_name):
    ##CREATE FOLDERS##
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    data_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis/Univariate/Figures'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis/Univariate/Tables'
    list_of_folders=[project_path,parent_folder,series_path,data_path,univariate_folder,univariate_figures,univariate_tables]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    ##FILL DATASET MISSING VALUES WITH PREVIOUS DAY VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SUMMARY STATISTICS##
    def compute_statistics(dataframe):
        """Compute mean, median, standard deviation and variance of each column of a dataset"""
        means=dataframe.mean().tolist()
        medians=dataframe.median().tolist()
        standard_deviations=dataframe.std().tolist()
        variances=dataframe.var().tolist()
        table=pd.DataFrame([means, medians, standard_deviations, variances])
        return table
    univariate_stats=compute_statistics(df)
    ##CREATE TABLE##
    from prettytable import PrettyTable
    from pathlib import Path
    def stat_table(dataframe, table, path_name):
        """Save summary statistics to table"""
        stat_table = PrettyTable()
        stat_table.title = 'Dataset Summary Statistics'
        stat_table.field_names = ['Metric',*list(dataframe)]
        stat_table.add_row(['Mean',*table.iloc[0]])
        stat_table.add_row(['Median',*table.iloc[1]])
        stat_table.add_row(['Standard Deviation',*table.iloc[2]])
        stat_table.add_row(['Variance',*table.iloc[3]])
        Path(f"{path_name}/Dataset Summary Statistics ("+df.columns[0]+").txt").write_text(str(stat_table))
        print(stat_table)
    stat_table(df, univariate_stats, path_name=univariate_tables)
    ##PLOT UNIVARIATE SERIES##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    def plot_series(df, series_name, path_name):
        fig=plt.figure()
        plt.plot(df, color='steelblue')
        plt.xlabel('Date')
        plt.ylabel(series_name)
        plt.title(series_name+' Time Series Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+series_name+" Time Series Plot.png", format="png")
        plt.close()    
    plot_series(df, series_name, path_name=univariate_figures)
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf
    ##SAVE ACF AND PACF PLOTS FOR EACH VARIABLE##
    for col in df.columns:
        plot_acf(df[col], title='Autocorrelation Function for '+col,lags=50)
        plt.xlabel('Lags')
        plt.ylabel('ACF')
        plt.savefig(f"{univariate_figures}/Autocorrelation Function ("+col+").png")
        plt.close()
        plot_pacf(df[col], title='Partial Autocorrelation Function for '+col, lags=50)    
        plt.xlabel('Lags')
        plt.ylabel('PACF')
        plt.savefig(f"{univariate_figures}/Partial Autocorrelation Function ("+col+").png")
        plt.close()
    ## FULL VARIABLE PLOTS ##
    def plot_diagnostics(y, title, lags=20, figsize=(12,8)):
        """Create and save diagnostic plots for time series"""
        fig= plt.figure(figsize=figsize)
        layout=(2,2)
        ts_ax=plt.subplot2grid(layout, (0,0))
        hist_ax=plt.subplot2grid(layout, (0,1))
        acf_ax=plt.subplot2grid(layout, (1,0))
        pacf_ax=plt.subplot2grid(layout, (1,1))
        y.plot(ax=ts_ax)
        ts_ax.set_title(title, fontsize=12, fontweight='bold')
        y.plot(ax=hist_ax, kind='hist', bins=25)
        hist_ax.set_title('Histogram')
        plot_acf(y, lags=lags, ax=acf_ax)
        plot_pacf(y, lags=lags, ax=pacf_ax)
        sns.despine()
        plt.tight_layout()
        return ts_ax, acf_ax, pacf_ax
    plot_diagnostics(df, df.columns[0], lags=20)
    plt.savefig(f"{univariate_figures}/Diagnostic Plots ("+df.columns[0]+").png", format="png")
    plt.close()
    ##CREATE DIFFERENCED DATASET##
    df_diff=df.diff().dropna()
    ##CHECK PLOTS##
    plot_diagnostics(df_diff, df.columns[0]+" (Differenced)")
    plt.savefig(f"{univariate_figures}/Diagnostic Plots ("+df.columns[0]+" Differenced Series).png", format="png")
    plt.close()
    ##CHECK FOR (NON)-STATIONARITY## 
    ##DEFINE ADF TEST##
    from statsmodels.tsa.stattools import adfuller
    def adfuller_test(series, signif=0.05, name='', verbose=False):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        description = []
        r = adfuller(series, autolag='AIC')
        output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
        p_value = output['pvalue'] 
        def adjust(val, length= 6): return str(val).ljust(length)
        description.append(f'    Augmented Dickey-Fuller Test on "{name}"\n   {"-"*47}')
        description.append(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        description.append(f' Significance Level    = {signif}')
        description.append(f' Test Statistic        = {output["test_statistic"]}')
        description.append(f' No. Lags Chosen       = {output["n_lags"]}')
        for key,val in r[4].items():
            description.append(f' Critical value {adjust(key)} = {round(val, 3)}')
        if p_value <= signif:
            description.append(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            description.append(f" => Series is Stationary.")
        else:
            description.append(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            description.append(f" => Series is Non-Stationary.") 
        description = "\n".join(description)
        print(description)
        return description
    ##PERFORM ADF TEST ON SERIES##
    Path(f"{univariate_tables}/Augmented Dickey Fuller Test on "+df.columns[0]+" Series.txt").write_text(adfuller_test(df, name=df.columns[0]))
    ##AND ON DIFFERENCED SERIES##
    Path(f"{univariate_tables}/Augmented Dickey Fuller Test on "+df.columns[0]+" Series (Differenced).txt").write_text(adfuller_test(df_diff, name=df.columns[0]+' (Differenced Series)'))
#####################################################################################################################################
##################################################   UNIVARIATE SCRIPT  (ARIMA)       ###############################################
def arima(df, parent_folder, series_name):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    ##CREATE FOLDERS##
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    model_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Univariate/Figures'
    univariate_naive = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Univariate/Figures/Random Walk'
    univariate_arima = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name /'Baseline Models/Univariate/Figures/ARIMA'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Univariate/Tables'
    list_of_folders=[project_path,parent_folder,series_path, model_path, univariate_folder, univariate_figures,univariate_naive,univariate_arima,univariate_tables]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    ##FILL DATASET MISSING VALUES WITH PREVIOUS DAY VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##IMPORT NECESSARY MODULES##
    from prettytable import PrettyTable
    from pathlib import Path
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    import pmdarima as pm
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from time import time
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(len(df_train))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ######################################       FIT BASELINE MODELS      ################################
    ######################################        RANDOM WALK MODEL       ################################
    start = time()
    baseline_model_forecasts_train=df_train.shift(1)
    baseline_model_forecasts_test=df_test.shift(1)
    univariate_random_walk_training_time=time()-start
    ##PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    ##DEFINE METRICS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    baseline_metrics_train=performance_metrics_calculator(df_train[1:].to_numpy(),baseline_model_forecasts_train[1:].to_numpy())
    baseline_metrics_test=performance_metrics_calculator(df_test[1:].to_numpy(),baseline_model_forecasts_test[1:].to_numpy())
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+df.columns[0]+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+df.columns[0]+" Series).txt").write_text(str(table))
        print(table)
        return table
    baseline_table=performance_table(df, baseline_metrics_train, path_name=univariate_tables, name='Random Walk Model (In-Sample)')
    baseline_table=performance_table(df, baseline_metrics_test, path_name=univariate_tables, name='Random Walk Model (Out-of-Sample)')
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+df.columns[0]+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    plot_performance(df, df_test[1:],baseline_model_forecasts_test[1:], 1, path_name=univariate_naive, model_name='Random Walk', label='Test', color='orangered')
    plot_performance(df, df_test[1:],baseline_model_forecasts_test[1:], 5, path_name=univariate_naive, model_name='Random Walk', label='Test', color='orangered')
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    scatter_plot(df_test[1:],baseline_model_forecasts_test[1:], model_name='Random Walk Model', path_name=univariate_naive)
    ############################################    ARIMA MODEL     ############################################
    ##FIT ARIMA MODEL (WITH MODEL SELECTION CRITERIA)##
    start = time()
    arima_model = pm.auto_arima(df_train, start_p=1, start_q=1, test='adf', max_p=10, max_q=10,max_d=5,m=1,seasonal=False,start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    univariate_arima_training_time=time()-start
    ##SAVE OPTIMIZED ORDER##
    order = arima_model.get_params()['order']
    ##SAVE SUMMARY AND PLOT##
    Path(f"{univariate_tables}/ARIMA "+str(order)+" Output "+df.columns[0]+" Series.txt").write_text(str(arima_model.summary()))
    arima_model.plot_diagnostics()
    plt.savefig(f"{univariate_arima}/ARIMA "+str(order)+" Model Plot Diagnostics "+df.columns[0]+".png", format="png")
    plt.close()
    ##PREDICT ON TRAINING SET##
    arima_train_model=ARIMA(df_train, order=(order),enforce_stationarity=False)
    arima_train_fit=arima_train_model.fit()
    arima_forecasts_train=ARIMAResults.predict(arima_train_fit)
    ##PREDICT ON TEST SET##
    train=df_train.values
    test=df_test.values
    endog = [x for x in train]
    start = time()
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(endog, order=(order),enforce_stationarity=False)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        endog.append(obs)
        print('Forecasting observation', t+1, 'of', test_size, '---> Predicted Value=%f, Actual Value=%f' % (yhat, obs)+' ---> '+str(int(100 * (t + 1) / test_size)) + '%'+' complete')
    univariate_arima_test_time=time()-start
    univariate_arima_training_test_time=univariate_arima_training_time+univariate_arima_test_time
    ##CONVERT FORECASTS TO DATAFRAME##
    arima_forecasts_test=pd.DataFrame(predictions, index=df_test.index)
    ##PERFORMANCE METRICS##
    arima_metrics_train=performance_metrics_calculator(df_train.to_numpy(), arima_forecasts_train.to_numpy())
    arima_metrics_test=performance_metrics_calculator(df_test.to_numpy(), arima_forecasts_test.to_numpy())
    ##CREATE TABLE##
    arima_train_table=performance_table(df, arima_metrics_train,  path_name=univariate_tables, name='ARIMA  '+str(order)+" In-Sample")
    arima_test_table=performance_table(df, arima_metrics_test,  path_name=univariate_tables, name='ARIMA  '+str(order)+" Out-of-Sample")
    ##PLOT PERFORMANCE (TRAINING SET)##
    plot_performance(df, df_train, arima_forecasts_train, 1, path_name=univariate_arima, model_name='ARIMA '+str(order)+' In-Sample', label='Training', color='lawngreen')
    plot_performance(df, df_train, arima_forecasts_train, 5, path_name=univariate_arima, model_name='ARIMA  '+str(order)+' In-Sample', label='Training', color='lawngreen')
    ##PLOT PERFORMANCE (TEST SET)##
    plot_performance(df, df_test, arima_forecasts_test, 1, path_name=univariate_arima, model_name='ARIMA '+str(order)+' Out-of-Sample', label='Test', color='orangered')
    plot_performance(df, df_test, arima_forecasts_test, 5, path_name=univariate_arima, model_name='ARIMA  '+str(order)+' Out-of-Sample', label='Test', color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train, arima_forecasts_train, model_name='ARIMA '+str(order)+' In-Sample', path_name=univariate_arima)
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test, arima_forecasts_test, model_name='ARIMA '+str(order)+' Out-of-Sample', path_name=univariate_arima)
    ##CREATE SUMMARY TABLES##
    ##TRAINING TIMES##
    univariate_baseline_training_times=[univariate_random_walk_training_time, univariate_arima_training_test_time]
    univariate_baseline_training_times=[round(item, 5) for item in univariate_baseline_training_times]
    univariate_baseline_training_times_table = PrettyTable()
    univariate_baseline_training_times_table.title = 'Univariate Baseline Models Training Times ('+df.columns[0]+')'
    univariate_baseline_training_times_table.field_names = [column_names[0],"Random Walk Model", "ARIMA "+str(order)]
    univariate_baseline_training_times_table.add_row(['Training Time (seconds)', *univariate_baseline_training_times])
    Path(f"{univariate_tables}/Univariate Baseline Models Training Times.txt").write_text(str(univariate_baseline_training_times_table))
    print(univariate_baseline_training_times_table)
    ##IN-SAMPLE METRICS##
    univariate_baseline_table_train = PrettyTable()
    univariate_baseline_table_train.title = 'Univariate In-Sample Baseline Model Performances ('+df.columns[0]+')'
    univariate_baseline_table_train.field_names = [column_names[0], "Random Walk Model", "ARIMA "+str(order)]
    for i in range(len(performance_metrics)):
        univariate_baseline_table_train.add_row([performance_metrics[i], baseline_metrics_train[i], arima_metrics_train[i]])
    Path(f"{univariate_tables}/Univariate In-Sample Baseline Models Performance.txt").write_text(str(univariate_baseline_table_train))
    print(univariate_baseline_table_train)
    ##OUT-OF-SAMPLE METRICS##
    univariate_baseline_table_test = PrettyTable()
    univariate_baseline_table_test.title = 'Univariate Out-of-Sample Model Performances ('+df.columns[0]+')'
    univariate_baseline_table_test.field_names = [column_names[0], "Random Walk Model", "ARIMA "+str(order)]
    for i in range(len(performance_metrics)):
        univariate_baseline_table_test.add_row([performance_metrics[i],baseline_metrics_test[i], arima_metrics_test[i]])
    Path(f"{univariate_tables}/Univariate Out-of-Sample Baseline Models Performance.txt").write_text(str(univariate_baseline_table_test))
    print(univariate_baseline_table_test)
#####################################################################################################################################
#####################################           UNIVARIATE SCRIPT (NEURAL NETWORKS)    ##############################################
##FEED FORWARD NEURAL NETWORK (FFNN)##
def ffnn_univariate(df, parent_folder, series_name, L, ffnn_nodes1, ffnn_nodes2, ffnn_epochs, ffnn_batch, ffnn_optimizer, T=1):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures'
    univariate_ffnn = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures/Feed-Forward Neural Network (FFNN)'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    univariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_figures,univariate_ffnn,univariate_tables, univariate_models]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(round(len(df_train)-len(df_train)*0.2))+" - Validation Set observations: "+str(round(len(df_train)*0.2))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ##DATA PREPARATION##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, GRU, Dense
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.regularizers import L1L2
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from time import time
    ##PERFORMANCE METRICS##
    ##IMPORT MODULES##
    import statsmodels.tsa.api as smt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    from prettytable import PrettyTable
    from pathlib import Path
    ##DEFINE METRICS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+df.columns[0]+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+df.columns[0]+" Series).txt").write_text(str(table))
        print(table)
        return table
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+df.columns[0]+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    #NEURAL NETWORKS#
    ##TENSOR CREATION##
    def data_transform(data, L, T=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        var_names = data.columns.tolist()
        cols, names = list(), list()
        for i in range(L, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
        for i in range(0, T):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    ##APPLY ALGORTIHM##
    T=1
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T:]
    train_x_data=data_train.iloc[: , :L]
    test_y_data=data_test.iloc[: , -T:]
    test_x_data=data_test.iloc[: , :L]
    ##SCALE DATA##
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    ##DEFINE THE FFNN MODEL##
    if len(df_train)>1000:
        ffnn_model = Sequential()
        ffnn_model.add(Dense(ffnn_nodes1, activation='relu'))
        ffnn_model.add(Dense(ffnn_nodes2, activation='relu'))
        ffnn_model.add(Dense(1))
        ffnn_model.compile(loss='mse', optimizer=ffnn_optimizer, metrics='mae')
    else:
        ffnn_model = Sequential()
        ffnn_model.add(Dense(ffnn_nodes1, activation='relu'))
        ffnn_model.add(Dense(1))
        ffnn_model.compile(loss='mse', optimizer=ffnn_optimizer, metrics='mae')
    ##SAVE BEST MODEL##
    ffnn_best_model=ModelCheckpoint(f"{univariate_models}/ffnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA AND RECORD TRAINING TIME##
    start=time()
    ffnn_fit=ffnn_model.fit(train_x, train_y, epochs=ffnn_epochs, batch_size=ffnn_batch, validation_split=0.2, verbose=2, callbacks=[ffnn_best_model])
    univariate_ffnn_training_time=time()-start
    ##EVALUATE FIT DURING TRAINING##
    def loss_plot(df, model,  path_name, name=''):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+df.columns[0]+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+df.columns[0]+" Series).png", format="png")
    ffnn_loss_plot_univariate=loss_plot(df, ffnn_fit, path_name=univariate_ffnn, name="FFNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(df, model, path_name, name=''):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+df.columns[0]+" Series).png", format="png")
    ffnn_mae_plot_univariate=mae_plot(df, ffnn_fit, path_name=univariate_ffnn, name='FFNN')
    ##LOAD BEST MODEL##
    ffnn_best_model=load_model(f"{univariate_models}/ffnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{univariate_models}/FFNN best model summary ("+series_name+").txt", 'w') as f:
        ffnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    ffnn_train_predict=ffnn_best_model.predict(train_x,verbose=2)
    ##FIT MODEL ON TEST DATA##
    ffnn_test_predict=ffnn_best_model.predict(test_x,verbose=2)
    ##INVERT SCALING AND RESHAPE VECTORS##
    train_y_inverse = scaler_y.inverse_transform(train_y)
    test_y_inverse = scaler_y.inverse_transform(test_y)
    ffnn_train_predict_inv=scaler_y.inverse_transform(ffnn_train_predict)
    ffnn_test_predict_inv=scaler_y.inverse_transform(ffnn_test_predict)
    ##CREATE REFERENCE DATASETS##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    ffnn_train_predict_inv=pd.DataFrame(ffnn_train_predict_inv, index=df_train_for_plotting.index)
    ffnn_test_predict_inv=pd.DataFrame(ffnn_test_predict_inv, index=df_test_for_plotting.index)
    ##COMPUTE PERFORMANCE METRICS##
    ffnn_metrics_train=performance_metrics_calculator(df_train_for_plotting.to_numpy(), ffnn_train_predict_inv.to_numpy())
    ffnn_metrics_test=performance_metrics_calculator(df_test_for_plotting.to_numpy(), ffnn_test_predict_inv.to_numpy())
    ##CREATE TABLES##
    ffnn_table_train=performance_table(df, ffnn_metrics_train,  path_name=univariate_tables, name='FFNN - In-Sample')
    ffnn_table_test=performance_table(df, ffnn_metrics_test,  path_name=univariate_tables, name='FFNN - Out-of-Sample')
    ##PLOTS (FULL)##
    plot_performance(df, df_train_for_plotting, ffnn_train_predict_inv, 1, path_name=univariate_ffnn, model_name='FFNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, ffnn_test_predict_inv, 1, path_name=univariate_ffnn, model_name='FFNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS##
    plot_performance(df, df_train_for_plotting, ffnn_train_predict_inv, 5, path_name=univariate_ffnn, model_name='FFNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, ffnn_test_predict_inv, 5, path_name=univariate_ffnn, model_name='FFNN - Out-of-Sample', label="Test", color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting, ffnn_train_predict_inv, path_name=univariate_ffnn, model_name='FFNN (Layers='+str(len(ffnn_model.layers))+') In-Sample')
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting, ffnn_test_predict_inv, path_name=univariate_ffnn, model_name='FFNN (Layers='+str(len(ffnn_model.layers))+') Out-of-Sample')
    plt.close('all')
    ##RETRIEVE NUMBER OF LAYERS##
    ffnn_layers = len(ffnn_model.layers)
    return univariate_ffnn_training_time, ffnn_metrics_train, ffnn_metrics_test, ffnn_layers
##RECURRENT NEURAL NETWORK (RNN)##
def rnn_univariate(df, parent_folder, series_name, L, rnn_nodes1, rnn_nodes2, rnn_epochs, rnn_batch, rnn_optimizer, T=1):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures'
    univariate_rnn = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures/Recurrent Neural Network (RNN)'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    univariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_figures,univariate_rnn,univariate_tables, univariate_models]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(round(len(df_train)-len(df_train)*0.2))+" - Validation Set observations: "+str(round(len(df_train)*0.2))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ##DATA PREPARATION##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, GRU, Dense
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.regularizers import L1L2
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from time import time
    ##PERFORMANCE METRICS##
    ##IMPORT MODULES##
    import statsmodels.tsa.api as smt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    from prettytable import PrettyTable
    from pathlib import Path
    ##DEFINE METRICS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+df.columns[0]+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+df.columns[0]+" Series).txt").write_text(str(table))
        print(table)
        return table
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+df.columns[0]+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    #NEURAL NETWORKS#
    ##TENSOR CREATION##
    def data_transform(data, L, T=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        var_names = data.columns.tolist()
        cols, names = list(), list()
        for i in range(L, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
        for i in range(0, T):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    ##APPLY ALGORTIHM##
    T=1
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T:]
    train_x_data=data_train.iloc[: , :L]
    test_y_data=data_test.iloc[: , -T:]
    test_x_data=data_test.iloc[: , :L]
    ##SCALE DATA##
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    features=df.shape[1]
    X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
    X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
    y_train= train_y.reshape(train_y.shape[0], features)
    y_test=test_y.reshape(test_y.shape[0], features)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    ##DEFINE THE RNN MODEL##
    if len(df_train)>1000:
        rnn_model = Sequential()
        rnn_model.add(SimpleRNN(rnn_nodes1, activation='tanh', return_sequences=True))
        rnn_model.add(SimpleRNN(rnn_nodes2, activation='tanh', return_sequences=False))
        rnn_model.add(Dense(1))
        rnn_model.compile(loss='mse', optimizer=rnn_optimizer, metrics='mae')
    else:
        rnn_model = Sequential()
        rnn_model.add(SimpleRNN(rnn_nodes1, activation='tanh', return_sequences=False))
        rnn_model.add(Dense(1))
        rnn_model.compile(loss='mse', optimizer=rnn_optimizer, metrics='mae')        
    ##SAVE BEST MODEL##
    rnn_best_model=ModelCheckpoint(f"{univariate_models}/rnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    rnn_fit=rnn_model.fit(X_train, y_train, epochs=rnn_epochs, batch_size=rnn_batch, validation_split=0.2, verbose=2, callbacks=[rnn_best_model])
    univariate_rnn_training_time=time()-start
    ##EVALUATE FIT DURING TRAINING##
    def loss_plot(df, model,  path_name, name=''):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+df.columns[0]+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+df.columns[0]+" Series).png", format="png")
    rnn_loss_plot_univariate=loss_plot(df, rnn_fit, path_name=univariate_rnn, name="Simple RNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(df, model, path_name, name=''):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+df.columns[0]+" Series).png", format="png")
    rnn_mae_plot_univariate=mae_plot(df, rnn_fit, path_name=univariate_rnn, name='Simple RNN')
    ##LOAD BEST MODEL##
    rnn_best_model=load_model(f"{univariate_models}/rnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{univariate_models}/RNN best model summary ("+series_name+").txt", 'w') as f:
        rnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    rnn_train_predict=rnn_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    rnn_test_predict=rnn_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING AND RESHAPE VECTORS##
    y_train_inverse = scaler_y.inverse_transform(y_train)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    rnn_train_predict_inv=scaler_y.inverse_transform(rnn_train_predict)
    rnn_test_predict_inv=scaler_y.inverse_transform(rnn_test_predict)
    ##CREATE DATAFRAMES##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    rnn_train_predict_inv=pd.DataFrame(rnn_train_predict_inv, index=df_train_for_plotting.index)
    rnn_test_predict_inv=pd.DataFrame(rnn_test_predict_inv, index=df_test_for_plotting.index)
    ##PERFORMANCE METRICS##
    rnn_metrics_train=performance_metrics_calculator(df_train_for_plotting.to_numpy(), rnn_train_predict_inv.to_numpy())
    rnn_metrics_test=performance_metrics_calculator(df_test_for_plotting.to_numpy(), rnn_test_predict_inv.to_numpy())
    ##CREATE TABLE##
    rnn_table_train=performance_table(df, rnn_metrics_train,  path_name=univariate_tables, name='RNN - In-Sample')
    rnn_table_test=performance_table(df, rnn_metrics_test,  path_name=univariate_tables, name='RNN - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    rnn_train_predict_inv_dataframe=pd.DataFrame(rnn_train_predict_inv, index=df_train_for_plotting.index)
    rnn_test_predict_inv_dataframe=pd.DataFrame(rnn_test_predict_inv, index=df_test_for_plotting.index)
    ##PLOTS (FULL)##
    plot_performance(df, df_train_for_plotting, rnn_train_predict_inv_dataframe, 1, path_name=univariate_rnn, model_name='RNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, rnn_test_predict_inv_dataframe, 1, path_name=univariate_rnn, model_name='RNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df, df_train_for_plotting, rnn_train_predict_inv_dataframe, 5, path_name=univariate_rnn, model_name='RNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, rnn_test_predict_inv_dataframe, 5, path_name=univariate_rnn, model_name='RNN - Out-of-Sample', label="Test", color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting, rnn_train_predict_inv_dataframe, path_name=univariate_rnn, model_name='RNN (Layers='+str(len(rnn_model.layers))+') In-Sample')
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting, rnn_test_predict_inv_dataframe, path_name=univariate_rnn, model_name='RNN (Layers='+str(len(rnn_model.layers))+') Out-of-Sample')
    plt.close('all')
    ##RETRIEVE NUMBER OF LAYERS##
    rnn_layers = len(rnn_model.layers)
    return univariate_rnn_training_time, rnn_metrics_train, rnn_metrics_test, rnn_layers
##LONG SHORT-TERM MEMORY (LSTM)##
def lstm_univariate(df, parent_folder, series_name, L, lstm_nodes1, lstm_nodes2, lstm_epochs, lstm_batch, lstm_optimizer, T=1):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures'
    univariate_lstm = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures/Long Short-Term Memory (LSTM)'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    univariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_figures,univariate_lstm,univariate_tables,univariate_models]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(round(len(df_train)-len(df_train)*0.2))+" - Validation Set observations: "+str(round(len(df_train)*0.2))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ##DATA PREPARATION##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, GRU, Dense
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.regularizers import L1L2
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from time import time
    ##PERFORMANCE METRICS##
    ##IMPORT MODULES##
    import statsmodels.tsa.api as smt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    from prettytable import PrettyTable
    from pathlib import Path
     ##DEFINE METRICS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+df.columns[0]+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+df.columns[0]+" Series).txt").write_text(str(table))
        print(table)
        return table
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+df.columns[0]+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    #NEURAL NETWORKS#
    ##TENSOR CREATION##
    def data_transform(data, L, T=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        var_names = data.columns.tolist()
        cols, names = list(), list()
        for i in range(L, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
        for i in range(0, T):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    ##APPLY ALGORTIHM##
    T=1
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T:]
    train_x_data=data_train.iloc[: , :L]
    test_y_data=data_test.iloc[: , -T:]
    test_x_data=data_test.iloc[: , :L]
    ##SCALE DATA##
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    features=df.shape[1]
    X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
    X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
    y_train= train_y.reshape(train_y.shape[0], features)
    y_test=test_y.reshape(test_y.shape[0], features)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    ##DEFINE THE LSTM MODEL##
    if len(df_train)>1000:        
        lstm_model = Sequential()
        lstm_model.add(LSTM(lstm_nodes1, activation='tanh', return_sequences=True))
        lstm_model.add(LSTM(lstm_nodes2, activation='tanh', return_sequences=False))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mse', optimizer=lstm_optimizer, metrics='mae')
    else:
        lstm_model = Sequential()
        lstm_model.add(LSTM(lstm_nodes1, activation='tanh', return_sequences=False))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mse', optimizer=lstm_optimizer, metrics='mae')        
    ##SAVE BEST MODEL##
    lstm_best_model=ModelCheckpoint(f"{univariate_models}/lstm_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    lstm_fit=lstm_model.fit(X_train, y_train, epochs=lstm_epochs, batch_size=lstm_batch, validation_split=0.2, verbose=2, callbacks=[lstm_best_model])
    univariate_lstm_training_time=time()-start
    ##EVALUATE FIT##
    def loss_plot(df, model,  path_name, name=''):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+df.columns[0]+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+df.columns[0]+" Series).png", format="png")
    lstm_loss_plot_univariate=loss_plot(df, lstm_fit, path_name=univariate_lstm, name="LSTM")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(df, model, path_name, name=''):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+df.columns[0]+" Series).png", format="png")
    lstm_mae_plot_univariate=mae_plot(df, lstm_fit, path_name=univariate_lstm, name='LSTM')
    ##LOAD BEST MODEL##
    lstm_best_model=load_model(f"{univariate_models}/lstm_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{univariate_models}/LSTM best model summary ("+series_name+").txt", 'w') as f:
        lstm_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    lstm_train_predict=lstm_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    lstm_test_predict=lstm_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING##
    lstm_train_predict_inv=scaler_y.inverse_transform(lstm_train_predict)
    lstm_test_predict_inv=scaler_y.inverse_transform(lstm_test_predict)
    ##CREATE DATAFRAMES##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    lstm_train_predict_inv=pd.DataFrame(lstm_train_predict_inv, index=df_train_for_plotting.index)
    lstm_test_predict_inv=pd.DataFrame(lstm_test_predict_inv, index=df_test_for_plotting.index)
    ##PERFORMANCE METRICS##
    lstm_metrics_train=performance_metrics_calculator(df_train_for_plotting.to_numpy(), lstm_train_predict_inv.to_numpy())
    lstm_metrics_test=performance_metrics_calculator(df_test_for_plotting.to_numpy(), lstm_test_predict_inv.to_numpy())
    ##CREATE TABLES##
    lstm_table_train=performance_table(df, lstm_metrics_train,  path_name=univariate_tables, name='LSTM - In-Sample')
    lstm_table_test=performance_table(df, lstm_metrics_test,  path_name=univariate_tables, name='LSTM - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    lstm_train_predict_inv_dataframe=pd.DataFrame(lstm_train_predict_inv, index=df_train_for_plotting.index)
    lstm_test_predict_inv_dataframe=pd.DataFrame(lstm_test_predict_inv, index=df_test_for_plotting.index)
    ##PLOTS (FULL)##
    plot_performance(df, df_train_for_plotting, lstm_train_predict_inv_dataframe, 1, path_name=univariate_lstm, model_name='LSTM - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, lstm_test_predict_inv_dataframe, 1, path_name=univariate_lstm, model_name='LSTM - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df, df_train_for_plotting, lstm_train_predict_inv_dataframe, 5, path_name=univariate_lstm, model_name='LSTM - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, lstm_test_predict_inv_dataframe, 5, path_name=univariate_lstm, model_name='LSTM - Out-of-Sample', label="Test", color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting, lstm_train_predict_inv_dataframe, path_name=univariate_lstm, model_name='LSTM (Layers='+str(len(lstm_model.layers))+') In-Sample')
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting, lstm_test_predict_inv_dataframe, path_name=univariate_lstm, model_name='LSTM (Layers='+str(len(lstm_model.layers))+') Out-of-Sample')
    plt.close('all')
    ##RETRIEVE NUMBER OF LAYERS##
    lstm_layers = len(lstm_model.layers)
    return univariate_lstm_training_time, lstm_metrics_train, lstm_metrics_test, lstm_layers
##GATED RECURRENT UNIT (GRU)##
def gru_univariate(df, parent_folder, series_name, L, gru_nodes1, gru_nodes2, gru_epochs, gru_batch, gru_optimizer, T=1):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures'
    univariate_gru = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures/Gated Recurrent Unit (GRU)'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    univariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_figures,univariate_gru,univariate_tables,univariate_models]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(round(len(df_train)-len(df_train)*0.2))+" - Validation Set observations: "+str(round(len(df_train)*0.2))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ##DATA PREPARATION##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, GRU, Dense
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.regularizers import L1L2
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from time import time
    ##PERFORMANCE METRICS##
    ##IMPORT MODULES##
    import statsmodels.tsa.api as smt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    from prettytable import PrettyTable
    from pathlib import Path
    ##DEFINE METRICS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+df.columns[0]+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+df.columns[0]+" Series).txt").write_text(str(table))
        print(table)
        return table
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+df.columns[0]+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    #NEURAL NETWORKS#
    ##TENSOR CREATION##
    def data_transform(data, L, T=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        var_names = data.columns.tolist()
        cols, names = list(), list()
        for i in range(L, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
        for i in range(0, T):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    ##APPLY ALGORTIHM##
    T=1
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T:]
    train_x_data=data_train.iloc[: , :L]
    test_y_data=data_test.iloc[: , -T:]
    test_x_data=data_test.iloc[: , :L]
    ##SCALE DATA##
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    features=df.shape[1]
    X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
    X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
    y_train= train_y.reshape(train_y.shape[0], features)
    y_test=test_y.reshape(test_y.shape[0], features)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    ##DEFINE THE GRU MODEL##
    if len(df_train)>1000:
        gru_model = Sequential()
        gru_model.add(GRU(gru_nodes1, activation='tanh', return_sequences=True))
        gru_model.add(GRU(gru_nodes2, activation='tanh', return_sequences=False))
        gru_model.add(Dense(1))
        gru_model.compile(loss='mse', optimizer=gru_optimizer, metrics='mae')
    else:
        gru_model = Sequential()
        gru_model.add(GRU(gru_nodes1, activation='tanh', return_sequences=False))
        gru_model.add(Dense(1))
        gru_model.compile(loss='mse', optimizer=gru_optimizer, metrics='mae')        
    ##SAVE BEST MODEL##
    gru_best_model=ModelCheckpoint(f"{univariate_models}/gru_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    gru_fit=gru_model.fit(X_train, y_train, epochs=gru_epochs, batch_size=gru_batch, validation_split=0.2, verbose=2, callbacks=[gru_best_model])
    univariate_gru_training_time=time()-start
    ##EVALUATE FIT##
    def loss_plot(df, model,  path_name, name=''):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+df.columns[0]+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+df.columns[0]+" Series).png", format="png")
    gru_loss_plot_univariate=loss_plot(df, gru_fit, path_name=univariate_gru, name="GRU")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(df, model, path_name, name=''):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+df.columns[0]+" Series).png", format="png")
    gru_mae_plot_univariate=mae_plot(df, gru_fit, path_name=univariate_gru, name='GRU')
    ##LOAD BEST MODEL##
    gru_best_model=load_model(f"{univariate_models}/gru_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{univariate_models}/GRU best model summary ("+series_name+").txt", 'w') as f:
        gru_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    gru_train_predict=gru_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    gru_test_predict=gru_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING##
    gru_train_predict_inv=scaler_y.inverse_transform(gru_train_predict)
    gru_test_predict_inv=scaler_y.inverse_transform(gru_test_predict)
    ##CREATE DATAFRAMES##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    gru_train_predict_inv=pd.DataFrame(gru_train_predict_inv, index=df_train_for_plotting.index)
    gru_test_predict_inv=pd.DataFrame(gru_test_predict_inv, index=df_test_for_plotting.index)
    ##PERFORMANCE METRICS##
    gru_metrics_train=performance_metrics_calculator(df_train_for_plotting.to_numpy(), gru_train_predict_inv.to_numpy())
    gru_metrics_test=performance_metrics_calculator(df_test_for_plotting.to_numpy(), gru_test_predict_inv.to_numpy())
    ##CREATE TABLES##
    gru_table_train=performance_table(df, gru_metrics_train, path_name=univariate_tables, name='GRU - In-Sample')
    gru_table_test=performance_table(df, gru_metrics_test, path_name=univariate_tables, name='GRU - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    gru_train_predict_inv_dataframe=pd.DataFrame(gru_train_predict_inv, index=df_train_for_plotting.index)
    gru_test_predict_inv_dataframe=pd.DataFrame(gru_test_predict_inv, index=df_test_for_plotting.index)
    ##PLOTS (FULL)##
    plot_performance(df, df_train_for_plotting, gru_train_predict_inv_dataframe, 1, path_name=univariate_gru, model_name='GRU - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, gru_test_predict_inv_dataframe, 1, path_name=univariate_gru, model_name='GRU - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df, df_train_for_plotting, gru_train_predict_inv_dataframe, 5, path_name=univariate_gru, model_name='GRU - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, gru_test_predict_inv_dataframe, 5, path_name=univariate_gru, model_name='GRU - Out-of-Sample', label="Test", color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting, gru_train_predict_inv_dataframe, path_name=univariate_gru, model_name='GRU (Layers='+str(len(gru_model.layers))+') In-Sample')
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting, gru_test_predict_inv_dataframe, path_name=univariate_gru, model_name='GRU (Layers='+str(len(gru_model.layers))+') Out-of-Sample')
    plt.close('all')
    ##RETRIEVE NUMBER OF LAYERS##
    gru_layers = len(gru_model.layers)
    return univariate_gru_training_time, gru_metrics_train, gru_metrics_test, gru_layers
##CONVOLUTIONAL NEURAL NETWORK (CNN)##
def cnn_univariate(df, parent_folder, series_name, L, cnn_filters_1, cnn_filters_2, cnn_dense_nodes, cnn_epochs, cnn_batch, cnn_optimizer, cnn_kernel_1=3, cnn_kernel_2=3, cnn_pool_size=2, T=1):
    ##SET SEED TO ENSURE REPRODUCIBILITY##
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures'
    univariate_cnn = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Figures/Convolutional Neural Network (CNN)'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    univariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_figures,univariate_cnn,univariate_tables,univariate_models]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import pandas as pd
    import datetime
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        count=df.isnull().values.sum()
        if missing_values==True:
            print('The number of missing values is '+str(count))
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    ##FILL REMAINING MISSING VALUES WITH ZEROS##
    df=df.fillna(0)
    ##SPLIT TRAINING AND TEST SETS##
    def train_test (dataset, train):
        """Define and operate train-test split on dataset and return the length of the test set as scalar"""
        df=dataset.astype(float)
        train_size = int(len(df) * train)
        test_size = len(df) - train_size
        df_train, df_test = df[0:train_size], df[train_size:len(df)]
        print("Training Set observations: "+str(round(len(df_train)-len(df_train)*0.2))+" - Validation Set observations: "+str(round(len(df_train)*0.2))+" - Test Set observations: "+str(len(df_test)))
        return df_train, df_test, test_size
    df_train, df_test, test_size = train_test(df, train=0.8)
    ##DATA PREPARATION##
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, GRU, Dense
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.models import load_model
    from keras.regularizers import L1L2
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    from time import time
    ##PERFORMANCE METRICS##
    ##IMPORT MODULES##
    import statsmodels.tsa.api as smt
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
    from prettytable import PrettyTable
    from pathlib import Path
    ##DEFINE FUNCTIONS##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    def performance_metrics_calculator(y_true,y_pred):
        """Compute metric scores for ML models"""
        mse=mean_squared_error(y_true,y_pred)
        rmse=sqrt(mse)
        mae=mean_absolute_error(y_true,y_pred)
        mape=mean_absolute_percentage_error(y_true,y_pred)
        mda=numpy.mean((numpy.sign(y_true[1:]-y_true[:-1])==numpy.sign(y_pred[1:]-y_pred[:-1])).astype(int))
        r2=r2_score(y_true, y_pred)
        metrics=[mse,rmse,mae,mape,mda,r2]
        return metrics
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+df.columns[0]+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+df.columns[0]+" Series).txt").write_text(str(table))
        print(table)
        return table
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name='', label='', color=''):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(df.columns[0])
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model ("+df.columns[0]+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##SCATTER PLOT PREDICTED VS. ACTUAL VALUES##
    def scatter_plot(actual, pred, path_name, model_name=''):
        fig=plt.figure()
        plt.scatter(actual, pred, color='steelblue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        ax = plt.gca()
        x = numpy.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')
        plt.title(series_name+'\n'+model_name+' Scatter Plot')
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+model_name+" Scatter Plot ("+series_name+").png", format="png")
        plt.close
    #NEURAL NETWORKS#
    ##TENSOR CREATION##
    def data_transform(data, L, T=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        var_names = data.columns.tolist()
        cols, names = list(), list()
        for i in range(L, 0, -1):
            cols.append(df.shift(i))
            names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
        for i in range(0, T):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    ##APPLY ALGORTIHM##
    T=1
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T:]
    train_x_data=data_train.iloc[: , :L]
    test_y_data=data_test.iloc[: , -T:]
    test_x_data=data_test.iloc[: , :L]
    ##SCALE DATA##
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    features=df.shape[1]
    X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
    X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
    y_train= train_y.reshape(train_y.shape[0], features)
    y_test=test_y.reshape(test_y.shape[0], features)
    ##IMPORT CNN MODULES##
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    ##DEFINE THE CNN MODEL##
    if len(df_train)>1000:
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=cnn_filters_1, kernel_size=cnn_kernel_1, activation='relu'))
        cnn_model.add(Conv1D(filters=cnn_filters_2, kernel_size=cnn_kernel_2, activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=cnn_pool_size))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(cnn_dense_nodes, activation='relu'))
        cnn_model.add(Dense(1))
        cnn_model.compile(loss='mse', optimizer=cnn_optimizer, metrics='mae')
    else:
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=cnn_filters_1, kernel_size=cnn_kernel_1, activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=cnn_pool_size))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(cnn_dense_nodes, activation='relu'))
        cnn_model.add(Dense(1))
        cnn_model.compile(loss='mse', optimizer=cnn_optimizer, metrics='mae')
    ##SAVE BEST MODEL##
    cnn_best_model=ModelCheckpoint(f"{univariate_models}/cnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    cnn_fit=cnn_model.fit(X_train, y_train, epochs=cnn_epochs, batch_size=cnn_batch, validation_split=0.2, verbose=2, callbacks=[cnn_best_model])
    univariate_cnn_training_time=time()-start
    ##EVALUATE FIT##
    def loss_plot(df, model,  path_name, name=''):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+df.columns[0]+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+df.columns[0]+" Series).png", format="png")
    cnn_loss_plot_univariate = loss_plot(df, cnn_fit, path_name=univariate_cnn, name="CNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(df, model, path_name, name=''):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+df.columns[0]+" Series).png", format="png")
    cnn_mae_plot_univariate = mae_plot(df, cnn_fit, path_name=univariate_cnn, name='CNN')
    ##LOAD BEST MODEL##
    cnn_best_model=load_model(f"{univariate_models}/cnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{univariate_models}/CNN best model summary ("+series_name+").txt", 'w') as f:
        cnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))    
    ##FIT MODEL ON TRAINING DATA##
    cnn_train_predict=cnn_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    cnn_test_predict=cnn_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING##
    cnn_train_predict_inv=scaler_y.inverse_transform(cnn_train_predict)
    cnn_test_predict_inv=scaler_y.inverse_transform(cnn_test_predict)
    ##CREATE DATAFRAMES##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    cnn_train_predict_inv=pd.DataFrame(cnn_train_predict_inv, index=df_train_for_plotting.index)
    cnn_test_predict_inv=pd.DataFrame(cnn_test_predict_inv, index=df_test_for_plotting.index)
    ##PERFORMANCE METRICS##
    cnn_metrics_train=performance_metrics_calculator(df_train_for_plotting.to_numpy(), cnn_train_predict_inv.to_numpy())
    cnn_metrics_test=performance_metrics_calculator(df_test_for_plotting.to_numpy(), cnn_test_predict_inv.to_numpy())
    ##CREATE TABLES##
    cnn_table_train=performance_table(df, cnn_metrics_train, path_name=univariate_tables, name='CNN - In-Sample')
    cnn_table_test=performance_table(df, cnn_metrics_test, path_name=univariate_tables, name='CNN - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    cnn_train_predict_inv_dataframe=pd.DataFrame(cnn_train_predict_inv, index=df_train_for_plotting.index)
    cnn_test_predict_inv_dataframe=pd.DataFrame(cnn_test_predict_inv, index=df_test_for_plotting.index)
    ##PLOTS (FULL)##
    plot_performance(df, df_train_for_plotting, cnn_train_predict_inv_dataframe, 1, path_name=univariate_cnn, model_name='CNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, cnn_test_predict_inv_dataframe, 1, path_name=univariate_cnn, model_name='CNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df, df_train_for_plotting, cnn_train_predict_inv_dataframe, 5, path_name=univariate_cnn, model_name='CNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df, df_test_for_plotting, cnn_test_predict_inv_dataframe, 5, path_name=univariate_cnn, model_name='CNN - Out-of-Sample', label="Test", color='orangered')
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting, cnn_train_predict_inv_dataframe, model_name='CNN (Layers='+str(len(cnn_model.layers))+') In-Sample', path_name=univariate_cnn)
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting, cnn_test_predict_inv_dataframe, model_name='CNN (Layers='+str(len(cnn_model.layers))+') Out-of-Sample', path_name=univariate_cnn)
    plt.close('all')
    ##RETRIEVE NUMBER OF LAYERS##
    cnn_layers = len(cnn_model.layers)
    return univariate_cnn_training_time, cnn_metrics_train, cnn_metrics_test, cnn_layers
#####################################################################################################################################
####################################              CREATE UNIVARIATE TABLES              #############################################
def univariate_tables(df, univariate_ffnn_training_time, univariate_rnn_training_time, univariate_lstm_training_time, univariate_gru_training_time, univariate_cnn_training_time,
                      ffnn_metrics_train, rnn_metrics_train, lstm_metrics_train, gru_metrics_train, cnn_metrics_train,
                      ffnn_metrics_test, rnn_metrics_test, lstm_metrics_test, gru_metrics_test, cnn_metrics_test,
                      ffnn_layers, rnn_layers, lstm_layers, gru_layers, cnn_layers,
                      parent_folder, series_name):
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Tables'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_tables]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    from prettytable import PrettyTable
    from pathlib import Path
    ##TRAINING TIMES##
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)','Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    univariate_training_times=[univariate_ffnn_training_time, univariate_rnn_training_time, univariate_lstm_training_time, univariate_gru_training_time, univariate_cnn_training_time]
    univariate_training_times=[round(item, 5) for item in univariate_training_times]
    univariate_training_times_table = PrettyTable()
    univariate_training_times_table.title = 'Univariate Neural Networks Training Times ('+df.columns[0]+')'
    univariate_training_times_table.field_names = [column_names[0],"Feed-Forward NN "+"(Layers="+str(ffnn_layers)+")", "Simple RNN "+"(Layers="+str(rnn_layers)+")", "LSTM "+"(Layers="+str(lstm_layers)+")", "GRU "+"(Layers="+str(gru_layers)+")", "CNN "+"(Layers="+str(cnn_layers)+")"]
    univariate_training_times_table.add_row(['Training Time (seconds)', *univariate_training_times])
    Path(f"{univariate_tables}/Univariate Neural Network Training Times.txt").write_text(str(univariate_training_times_table))
    print(univariate_training_times_table)
    ##IN-SAMPLE METRICS##
    univariate_table_train = PrettyTable()
    univariate_table_train.title = 'Univariate In-Sample Model Performances ('+df.columns[0]+')'
    univariate_table_train.field_names = [column_names[0],"Feed-Forward NN "+"(Layers="+str(ffnn_layers)+")", "Simple RNN "+"(Layers="+str(rnn_layers)+")", "LSTM "+"(Layers="+str(lstm_layers)+")", "GRU "+"(Layers="+str(gru_layers)+")", "CNN "+"(Layers="+str(cnn_layers)+")"]
    for i in range(len(performance_metrics)):
        univariate_table_train.add_row([performance_metrics[i], ffnn_metrics_train[i],rnn_metrics_train[i],lstm_metrics_train[i],gru_metrics_train[i],cnn_metrics_train[i]])
    Path(f"{univariate_tables}/Univariate In-Sample Model Performance.txt").write_text(str(univariate_table_train))
    print(univariate_table_train)
    ##OUT-OF-SAMPLE METRICS##
    univariate_table_test = PrettyTable()
    univariate_table_test.title = 'Univariate Out-of-Sample Model Performances ('+df.columns[0]+')'
    univariate_table_test.field_names = [column_names[0],"Feed-Forward NN "+"(Layers="+str(ffnn_layers)+")", "Simple RNN "+"(Layers="+str(rnn_layers)+")", "LSTM "+"(Layers="+str(lstm_layers)+")", "GRU "+"(Layers="+str(gru_layers)+")", "CNN "+"(Layers="+str(cnn_layers)+")"]
    for i in range(len(performance_metrics)):
        univariate_table_test.add_row([performance_metrics[i],ffnn_metrics_test[i],rnn_metrics_test[i],lstm_metrics_test[i],gru_metrics_test[i],cnn_metrics_test[i]])
    Path(f"{univariate_tables}/Univariate Out-of-Sample Model Performance.txt").write_text(str(univariate_table_test))
    print(univariate_table_test)
#####################################################################################################################################
####################################               GRID SEARCH FUNCTIONS                #############################################
##FEED FORWARD NEURAL NETWORK (FFNN)##
def grid_search_ffnn_univariate(df, parent_folder, series_name, lags, nodes1, nodes2, optimizer, epochs, batch):
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def ffnn_univariate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, ffnn_nodes1, ffnn_nodes2, ffnn_epochs, ffnn_batch, ffnn_optimizer = config
        from sklearn.preprocessing import MinMaxScaler
        def train_test (dataset, train):
            """Define and operate train-test split on dataset and return the length of the test set as scalar"""
            df=dataset.astype(float)
            train_size = int(len(df) * train)
            test_size = len(df) - train_size
            df_train, df_test = df[0:train_size], df[train_size:len(df)]
            return df_train, df_test
        df_train, df_test = train_test(df, train=0.8)
        def data_transform(data, L, T=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            var_names = data.columns.tolist()
            cols, names = list(), list()
            for i in range(L, 0, -1):
                cols.append(df.shift(i))
                names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
            for i in range(0, T):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
                else:
                    names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        ##APPLY ALGORTIHM##
        T=1
        data_train=data_transform(df_train, L,T)
        data_test=data_transform(df_test,L,T)
        train_y_data=data_train.iloc[: , -T:]
        train_x_data=data_train.iloc[: , :L]
        test_y_data=data_test.iloc[: , -T:]
        test_x_data=data_test.iloc[: , :L]
        ##SCALE DATA##
        scaler_x = MinMaxScaler(feature_range=(0,1))
        scaler_y = MinMaxScaler(feature_range=(0,1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        #################################################       FEED-FORWARD NN     ##############################################
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.models import load_model
        ##DEFINE THE FFNN MODEL##
        if len(df_train)>1000:
            ffnn_model = Sequential()
            ffnn_model.add(Dense(ffnn_nodes1, activation='relu'))
            ffnn_model.add(Dense(ffnn_nodes2, activation='relu'))
            ffnn_model.add(Dense(1))
            ffnn_model.compile(loss='mse', optimizer=ffnn_optimizer, metrics='mae')
        else:
            ffnn_model = Sequential()
            ffnn_model.add(Dense(ffnn_nodes1, activation='relu'))
            ffnn_model.add(Dense(1))
            ffnn_model.compile(loss='mse', optimizer=ffnn_optimizer, metrics='mae')
        ##SAVE BEST MODEL##
        ffnn_best_model=ModelCheckpoint(f"{univariate_grid_search}/ffnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL##
        ffnn_fit=ffnn_model.fit(train_x, train_y, epochs=ffnn_epochs, batch_size=ffnn_batch, validation_split=0.2, verbose=2, callbacks=[ffnn_best_model])
        ##LOAD BEST MODEL##
        ffnn_best_model=load_model(f"{univariate_grid_search}/ffnn_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        ffnn_test_predict=ffnn_best_model.predict(test_x,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        test_y_inv = scaler_y.inverse_transform(test_y)
        ffnn_test_predict_inv=scaler_y.inverse_transform(ffnn_test_predict)
        ##COMPUTE RMSE##
        from math import sqrt
        from sklearn.metrics import mean_squared_error
        rmse_ffnn=sqrt(mean_squared_error(test_y_inv, ffnn_test_predict_inv))
        return rmse_ffnn
    ##GRID SEARCH##
    def evaluate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, ffnn_nodes1, ffnn_nodes2, ffnn_epochs, ffnn_batch, ffnn_optimizer = config
        key = str(config)
        scores = ffnn_univariate(config, df)
        result = scores
        print('> Model[%s] %.3f' % (key, result))
        return (key, result)
    def grid_search(cfg_list, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        scores = [evaluate(cfg, df) for cfg in cfg_list]
        scores.sort(key=lambda tup: tup[1])
        return scores
    def model_configs():
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L = lags
        ffnn_nodes1 = nodes1
        ffnn_nodes2 = nodes2
        ffnn_epochs = epochs
        ffnn_batch = batch
        ffnn_optimizer = optimizer
        configs = list()
        for i in L:
            for j in ffnn_nodes1:
                for k in ffnn_nodes2:
                    for l in ffnn_epochs:
                        for m in ffnn_batch:
                            for n in ffnn_optimizer:
                                cfg = [i, j, k, l, m, n]
                                configs.append(cfg)
        print('Total configs: %d' % len(configs))
        return configs
    ##SPECIFY SERIES AND REMOVE MISSING VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        if missing_values==True:
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    df=df.fillna(0)
    ##SPECIFY NUMBER OF CONFIGURATIONS##
    cfg_list = model_configs()
    ##GRID SEARCH##
    import datetime
    begin_time = datetime.datetime.now().replace(microsecond=0)
    scores = grid_search(cfg_list, df)
    execution_time=datetime.datetime.now().replace(microsecond=0) - begin_time
    print("Configurations Tried : "+str(len(cfg_list)))
    print("Execution time is "+str(execution_time))
    ##LIST AND SAVE COMBINATIONS AND THEIR RMSE##
    with open(f"{univariate_grid_search}/FFNN - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Feed Forward Neural Network (FFNN) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel Configuration: [Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]\n", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##RECURRENT NEURAL NETWORK (RNN)##
def grid_search_rnn_univariate(df, parent_folder, series_name, lags, nodes1, nodes2, optimizer, epochs, batch):
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
        ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def rnn_univariate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, rnn_nodes1, rnn_nodes2, rnn_epochs, rnn_batch, rnn_optimizer = config
        from sklearn.preprocessing import MinMaxScaler
        def train_test (dataset, train):
            """Define and operate train-test split on dataset and return the length of the test set as scalar"""
            df=dataset.astype(float)
            train_size = int(len(df) * train)
            test_size = len(df) - train_size
            df_train, df_test = df[0:train_size], df[train_size:len(df)]
            return df_train, df_test
        df_train, df_test = train_test(df, train=0.8)
        def data_transform(data, L, T=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            var_names = data.columns.tolist()
            cols, names = list(), list()
            for i in range(L, 0, -1):
                cols.append(df.shift(i))
                names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
            for i in range(0, T):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
                else:
                    names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        ##APPLY ALGORTIHM##
        data_train=data_transform(df_train, L,1)
        data_test=data_transform(df_test,L,1)
        train_y_data=data_train.iloc[: , -1:]
        train_x_data=data_train.iloc[: , :L]
        test_y_data=data_test.iloc[: , -1:]
        test_x_data=data_test.iloc[: , :L]
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##KERAS TENSOR##
        features=df.shape[1]
        X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
        X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
        y_train= train_y.reshape(train_y.shape[0], features)
        y_test=test_y.reshape(test_y.shape[0], features)
        #################################################       RNN     ##############################################
        from keras.models import Sequential
        from keras.layers import SimpleRNN, LSTM, GRU, Dense
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.models import load_model
        ##DEFINE THE RNN MODEL##
        if len(df_train)>1000:
            rnn_model = Sequential()
            rnn_model.add(SimpleRNN(rnn_nodes1, activation='tanh', return_sequences=True))
            rnn_model.add(SimpleRNN(rnn_nodes2, activation='tanh', return_sequences=False))
            rnn_model.add(Dense(1))
            rnn_model.compile(loss='mse', optimizer=rnn_optimizer, metrics='mae')
        else:
            rnn_model = Sequential()
            rnn_model.add(SimpleRNN(rnn_nodes1, activation='tanh', return_sequences=False))
            rnn_model.add(Dense(1))
            rnn_model.compile(loss='mse', optimizer=rnn_optimizer, metrics='mae')
        ##SAVE BEST MODEL##
        rnn_best_model=ModelCheckpoint(f"{univariate_grid_search}/rnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        rnn_fit=rnn_model.fit(X_train, y_train, epochs=rnn_epochs, batch_size=rnn_batch, validation_split=0.2, verbose=2, callbacks=[rnn_best_model])
        ##LOAD BEST MODEL##
        rnn_best_model=load_model(f"{univariate_grid_search}/rnn_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        rnn_test_predict=rnn_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(y_test)
        rnn_test_predict_inv=scaler_y.inverse_transform(rnn_test_predict)
        ##COMPUTE RMSE##
        from math import sqrt
        from sklearn.metrics import mean_squared_error
        rmse_rnn=sqrt(mean_squared_error(y_test_inv, rnn_test_predict_inv))
        return rmse_rnn
    ##GRID SEARCH##
    def evaluate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, rnn_nodes1, rnn_nodes2, rnn_epochs, rnn_batch, rnn_optimizer = config
        key = str(config)
        scores = rnn_univariate(config, df)
        result = scores
        print('> Model[%s] %.3f' % (key, result))
        return (key, result)
    def grid_search(cfg_list, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        scores = [evaluate(cfg, df) for cfg in cfg_list]
        scores.sort(key=lambda tup: tup[1])
        return scores
    def model_configs():
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L = lags                
        rnn_nodes1 = nodes1
        rnn_nodes2 = nodes2
        rnn_epochs = epochs
        rnn_batch = batch
        rnn_optimizer = optimizer
        configs = list()
        for i in L:
            for j in rnn_nodes1:
                for k in rnn_nodes2:
                    for l in rnn_epochs:
                        for m in rnn_batch:
                            for n in rnn_optimizer:
                                cfg = [i, j, k, l, m, n]
                                configs.append(cfg)
        print('Total configs: %d' % len(configs))
        return configs
    ##SPECIFY SERIES AND REMOVE MISSING VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        if missing_values==True:
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    df=df.fillna(0)
    ##SPECIFY NUMBER OF CONFIGURATIONS##
    cfg_list = model_configs()
    ##GRID SEARCH##
    import datetime
    begin_time = datetime.datetime.now().replace(microsecond=0)
    scores = grid_search(cfg_list, df)
    execution_time=datetime.datetime.now().replace(microsecond=0) - begin_time
    print("Configurations Tried : "+str(len(cfg_list)))
    print("Execution time is "+str(execution_time))
    ##LIST AND SAVE COMBINATIONS AND THEIR RMSE##
    with open(f"{univariate_grid_search}/RNN - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Recurrent Neural Network (RNN) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel configuration: [Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]\n", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##LONG SHORT-TERM MEMORY (LSTM)##
def grid_search_lstm_univariate(df, parent_folder, series_name, lags, nodes1, nodes2, optimizer, epochs, batch):
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
        ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def lstm_univariate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, lstm_nodes1, lstm_nodes2, lstm_epochs, lstm_batch, lstm_optimizer = config
        from sklearn.preprocessing import MinMaxScaler
        def train_test (dataset, train):
            """Define and operate train-test split on dataset and return the length of the test set as scalar"""
            df=dataset.astype(float)
            train_size = int(len(df) * train)
            test_size = len(df) - train_size
            df_train, df_test = df[0:train_size], df[train_size:len(df)]
            return df_train, df_test
        df_train, df_test = train_test(df, train=0.8)
        def data_transform(data, L, T=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            var_names = data.columns.tolist()
            cols, names = list(), list()
            for i in range(L, 0, -1):
                cols.append(df.shift(i))
                names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
            for i in range(0, T):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
                else:
                    names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        ##APPLY ALGORTIHM##
        data_train=data_transform(df_train, L,1)
        data_test=data_transform(df_test,L,1)
        train_y_data=data_train.iloc[: , -1:]
        train_x_data=data_train.iloc[: , :L]
        test_y_data=data_test.iloc[: , -1:]
        test_x_data=data_test.iloc[: , :L]
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##KERAS TENSOR##
        features=df.shape[1]
        X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
        X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
        y_train= train_y.reshape(train_y.shape[0], features)
        y_test=test_y.reshape(test_y.shape[0], features)
        #################################################       LSTM     ##############################################
        from keras.models import Sequential
        from keras.layers import SimpleRNN, LSTM, GRU, Dense
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.models import load_model
        ##DEFINE THE LSTM MODEL##
        if len(df_train)>1000:
            lstm_model = Sequential()
            lstm_model.add(LSTM(lstm_nodes1, activation='tanh', return_sequences=True))
            lstm_model.add(LSTM(lstm_nodes2, activation='tanh', return_sequences=False))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mse', optimizer=lstm_optimizer, metrics='mae')
        else:    
            lstm_model = Sequential()
            lstm_model.add(LSTM(lstm_nodes1, activation='tanh', return_sequences=False))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mse', optimizer=lstm_optimizer, metrics='mae')
        ##SAVE BEST MODEL##
        lstm_best_model=ModelCheckpoint(f"{univariate_grid_search}/lstm_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        lstm_fit=lstm_model.fit(X_train, y_train, epochs=lstm_epochs, batch_size=lstm_batch, validation_split=0.2, verbose=2, callbacks=[lstm_best_model])
        ##LOAD BEST MODEL##
        lstm_best_model=load_model(f"{univariate_grid_search}/lstm_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        lstm_test_predict=lstm_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(y_test)
        lstm_test_predict_inv=scaler_y.inverse_transform(lstm_test_predict)
        ##COMPUTE RMSE##
        from math import sqrt
        from sklearn.metrics import mean_squared_error
        rmse_lstm=sqrt(mean_squared_error(y_test_inv, lstm_test_predict_inv))
        return rmse_lstm
    ##GRID SEARCH##
    def evaluate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, lstm_nodes1, lstm_nodes2, lstm_epochs, lstm_batch, lstm_optimizer = config
        key = str(config)
        scores = lstm_univariate(config, df)
        result = scores
        print('> Model[%s] %.3f' % (key, result))
        return (key, result)
    def grid_search(cfg_list, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        scores = [evaluate(cfg, df) for cfg in cfg_list]
        scores.sort(key=lambda tup: tup[1])
        return scores
    def model_configs():
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L = lags                
        lstm_nodes1 = nodes1
        lstm_nodes2 = nodes2
        lstm_epochs = epochs
        lstm_batch = batch
        lstm_optimizer = optimizer
        configs = list()
        for i in L:
            for j in lstm_nodes1:
                for k in lstm_nodes2:
                    for l in lstm_epochs:
                        for m in lstm_batch:
                            for n in lstm_optimizer:
                                cfg = [i, j, k, l, m, n]
                                configs.append(cfg)
        print('Total configs: %d' % len(configs))
        return configs
    ##SPECIFY SERIES AND REMOVE MISSING VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        if missing_values==True:
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    df=df.fillna(0)
    ##SPECIFY NUMBER OF CONFIGURATIONS##
    cfg_list = model_configs()
    ##GRID SEARCH##
    import datetime
    begin_time = datetime.datetime.now().replace(microsecond=0)
    scores = grid_search(cfg_list, df)
    execution_time=datetime.datetime.now().replace(microsecond=0) - begin_time
    print("Configurations Tried : "+str(len(cfg_list)))
    print("Execution time is "+str(execution_time))
    ##LIST AND SAVE COMBINATIONS AND THEIR RMSE##
    with open(f"{univariate_grid_search}/LSTM - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Long Short-Term Memory (LSTM) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel configuration: [Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]\n", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##GATED RECURRENT UNIT (GRU)##
def grid_search_gru_univariate(df, parent_folder, series_name, lags, nodes1, nodes2, optimizer, epochs, batch):
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
        ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def gru_univariate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, gru_nodes1, gru_nodes2, gru_epochs, gru_batch, gru_optimizer = config
        from sklearn.preprocessing import MinMaxScaler
        def train_test (dataset, train):
            """Define and operate train-test split on dataset and return the length of the test set as scalar"""
            df=dataset.astype(float)
            train_size = int(len(df) * train)
            test_size = len(df) - train_size
            df_train, df_test = df[0:train_size], df[train_size:len(df)]
            return df_train, df_test
        df_train, df_test = train_test(df, train=0.8)
        def data_transform(data, L, T=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            var_names = data.columns.tolist()
            cols, names = list(), list()
            for i in range(L, 0, -1):
                cols.append(df.shift(i))
                names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
            for i in range(0, T):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
                else:
                    names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        ##APPLY ALGORTIHM##
        data_train=data_transform(df_train, L,1)
        data_test=data_transform(df_test,L,1)
        train_y_data=data_train.iloc[: , -1:]
        train_x_data=data_train.iloc[: , :L]
        test_y_data=data_test.iloc[: , -1:]
        test_x_data=data_test.iloc[: , :L]
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##KERAS TENSOR##
        features=df.shape[1]
        X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
        X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
        y_train= train_y.reshape(train_y.shape[0], features)
        y_test=test_y.reshape(test_y.shape[0], features)
        #################################################       GRU     ##############################################
        from keras.models import Sequential
        from keras.layers import SimpleRNN, LSTM, GRU, Dense
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.models import load_model
        ##DEFINE THE GRU MODEL##
        if len(df_train)>1000:
            gru_model = Sequential()
            gru_model.add(GRU(gru_nodes1, activation='tanh', return_sequences=True))
            gru_model.add(GRU(gru_nodes2, activation='tanh', return_sequences=False))
            gru_model.add(Dense(1))
            gru_model.compile(loss='mse', optimizer=gru_optimizer, metrics='mae')
        else:
            gru_model = Sequential()
            gru_model.add(GRU(gru_nodes1, activation='tanh', return_sequences=False))
            gru_model.add(Dense(1))
            gru_model.compile(loss='mse', optimizer=gru_optimizer, metrics='mae')            
        ##SAVE BEST MODEL##
        gru_best_model=ModelCheckpoint(f"{univariate_grid_search}/gru_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        gru_fit=gru_model.fit(X_train, y_train, epochs=gru_epochs, batch_size=gru_batch, validation_split=0.2, verbose=2, callbacks=[gru_best_model])
        ##LOAD BEST MODEL##
        gru_best_model=load_model(f"{univariate_grid_search}/gru_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        gru_test_predict=gru_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(y_test)
        gru_test_predict_inv=scaler_y.inverse_transform(gru_test_predict)
        ##COMPUTE RMSE##
        from math import sqrt
        from sklearn.metrics import mean_squared_error
        rmse_gru=sqrt(mean_squared_error(y_test_inv, gru_test_predict_inv))
        return rmse_gru
    ##GRID SEARCH##
    def evaluate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, gru_nodes1, gru_nodes2, gru_epochs, gru_batch, gru_optimizer = config
        key = str(config)
        scores = gru_univariate(config, df)
        result = scores
        print('> Model[%s] %.3f' % (key, result))
        return (key, result)
    def grid_search(cfg_list, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        scores = [evaluate(cfg, df) for cfg in cfg_list]
        scores.sort(key=lambda tup: tup[1])
        return scores
    def model_configs():
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L = lags                
        gru_nodes1 = nodes1
        gru_nodes2 = nodes2
        gru_epochs = epochs
        gru_batch = batch
        gru_optimizer = optimizer
        configs = list()
        for i in L:
            for j in gru_nodes1:
                for k in gru_nodes2:
                    for l in gru_epochs:
                        for m in gru_batch:
                            for n in gru_optimizer:
                                cfg = [i, j, k, l, m, n]
                                configs.append(cfg)
        print('Total configs: %d' % len(configs))
        return configs
    ##SPECIFY SERIES AND REMOVE MISSING VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        if missing_values==True:
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    df=df.fillna(0)
    ##SPECIFY NUMBER OF CONFIGURATIONS##
    cfg_list = model_configs()
    ##GRID SEARCH##
    import datetime
    begin_time = datetime.datetime.now().replace(microsecond=0)
    scores = grid_search(cfg_list, df)
    execution_time=datetime.datetime.now().replace(microsecond=0) - begin_time
    print("Configurations Tried : "+str(len(cfg_list)))
    print("Execution time is "+str(execution_time))
    ##LIST AND SAVE COMBINATIONS AND THEIR RMSE##
    with open(f"{univariate_grid_search}/GRU - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Gated Recurrent Unit (GRU) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel configuration: [Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]\n", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##CONVOLUTIONAL NEURAL NETWORK (CNN)##
def grid_search_cnn_univariate(df, parent_folder, series_name, lags, filters1, filters2, cnn_nodes, optimizer, epochs, batch):
    seed_value=12345
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    import random
    random.seed(seed_value)
    import numpy
    numpy.random.seed(seed_value)
    import tensorflow
    tensorflow.random.set_seed(seed_value)
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    univariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate'
    univariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Univariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,univariate_folder,univariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
        ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def cnn_univariate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, cnn_filters_1, cnn_filters_2, cnn_dense_nodes, cnn_epochs, cnn_batch, cnn_optimizer = config
        from sklearn.preprocessing import MinMaxScaler
        def train_test (dataset, train):
            """Define and operate train-test split on dataset and return the length of the test set as scalar"""
            df=dataset.astype(float)
            train_size = int(len(df) * train)
            test_size = len(df) - train_size
            df_train, df_test = df[0:train_size], df[train_size:len(df)]
            return df_train, df_test
        df_train, df_test = train_test(df, train=0.8)
        def data_transform(data, L, T=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            var_names = data.columns.tolist()
            cols, names = list(), list()
            for i in range(L, 0, -1):
                cols.append(df.shift(i))
                names += [('%s (t-%d)' % (var_names[j], i)) for j in range(n_vars)]
            for i in range(0, T):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('%s (t)' % (var_names[j])) for j in range(n_vars)]
                else:
                    names += [('%s (t+%d)' % (var_names[j], i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        ##APPLY ALGORTIHM##
        data_train=data_transform(df_train, L,1)
        data_test=data_transform(df_test,L,1)
        train_y_data=data_train.iloc[: , -1:]
        train_x_data=data_train.iloc[: , :L]
        test_y_data=data_test.iloc[: , -1:]
        test_x_data=data_test.iloc[: , :L]
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##KERAS TENSOR##
        features=df.shape[1]
        X_train = train_x.reshape((train_x.shape[0], train_x.shape[1], features))
        X_test = test_x.reshape((test_x.shape[0], test_x.shape[1], features))
        y_train= train_y.reshape(train_y.shape[0], features)
        y_test=test_y.reshape(test_y.shape[0], features)
        #################################################       CNN     ##############################################
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers.convolutional import Conv1D
        from keras.layers.convolutional import MaxPooling1D
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.models import load_model
        ##DEFINE THE CNN MODEL##
        if len(df_train)>1000:
            cnn_model = Sequential()
            cnn_model.add(Conv1D(filters=cnn_filters_1, kernel_size=2, activation='relu'))
            cnn_model.add(Conv1D(filters=cnn_filters_2, kernel_size=2, activation='relu'))
            cnn_model.add(MaxPooling1D(pool_size=2))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(cnn_dense_nodes, activation='relu'))
            cnn_model.add(Dense(1))
            cnn_model.compile(loss='mse', optimizer=cnn_optimizer, metrics='mae')
        else:
            cnn_model = Sequential()
            cnn_model.add(Conv1D(filters=cnn_filters_1, kernel_size=3, activation='relu'))
            cnn_model.add(MaxPooling1D(pool_size=2))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(cnn_dense_nodes, activation='relu'))
            cnn_model.add(Dense(1))
            cnn_model.compile(loss='mse', optimizer=cnn_optimizer, metrics='mae')
        ##SAVE BEST MODEL##
        cnn_best_model=ModelCheckpoint(f"{univariate_grid_search}/cnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        cnn_fit=cnn_model.fit(X_train, y_train, epochs=cnn_epochs, batch_size=cnn_batch, validation_split=0.2, verbose=2, callbacks=[cnn_best_model])
        ##LOAD BEST MODEL##
        cnn_best_model=load_model(f"{univariate_grid_search}/cnn_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        cnn_test_predict=cnn_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(y_test)
        cnn_test_predict_inv=scaler_y.inverse_transform(cnn_test_predict)
        ##COMPUTE RMSE##
        from math import sqrt
        from sklearn.metrics import mean_squared_error
        rmse_cnn=sqrt(mean_squared_error(y_test_inv, cnn_test_predict_inv))
        return rmse_cnn
    ##GRID SEARCH##
    def evaluate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, cnn_filters_1, cnn_filters_2, cnn_dense_nodes, cnn_epochs, cnn_batch, cnn_optimizer = config
        key = str(config)
        scores = cnn_univariate(config, df)
        result = scores
        print('> Model[%s] %.3f' % (key, result))
        return (key, result)
    def grid_search(cfg_list, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        scores = [evaluate(cfg, df) for cfg in cfg_list]
        scores.sort(key=lambda tup: tup[1])
        return scores
    def model_configs():
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L = lags
        cnn_filters_1 = filters1
        cnn_filters_2 = filters2
        cnn_dense_nodes = cnn_nodes
        cnn_epochs = epochs
        cnn_batch = batch
        cnn_optimizer = optimizer
        configs = list()
        for i in L:
            for j in cnn_filters_1:
                for k in cnn_filters_2:
                    for l in cnn_dense_nodes:
                        for m in cnn_epochs:
                            for n in cnn_batch:
                                for o in cnn_optimizer:
                                    cfg = [i, j, k, l, m, n, o]
                                    configs.append(cfg)
        print('Total configs: %d' % len(configs))
        return configs
    ##SPECIFY SERIES AND REMOVE MISSING VALUES##
    def fill_missing_values(dataframe, method='ffill'):
        """Fill missing values of dataset with previous day values (default)"""
        missing_values=dataframe.isnull().values.any()
        if missing_values==True:
            dataframe.fillna(method='ffill', inplace=True)
            print("Missing values have been filled with previous day values")
        else:
            print("The dataset does not contain missing values")
    fill_missing_values(df)
    df=df.fillna(0)
    ##SPECIFY NUMBER OF CONFIGURATIONS##
    cfg_list = model_configs()
    ##GRID SEARCH##
    import datetime
    begin_time = datetime.datetime.now().replace(microsecond=0)
    scores = grid_search(cfg_list, df)
    execution_time=datetime.datetime.now().replace(microsecond=0) - begin_time
    print("Configurations Tried : "+str(len(cfg_list)))
    print("Execution time is "+str(execution_time))
    ##LIST AND SAVE COMBINATIONS AND THEIR RMSE##
    with open(f"{univariate_grid_search}/CNN - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Convolutional Neural Network (CNN) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel configuration: [Inputs, First Layer Filters, Second Layer Filters, Third Layer Neurons, Epochs, Batch Size, Optimizer]\n", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
#####################################################################################################################################
#####################################################################################################################################
