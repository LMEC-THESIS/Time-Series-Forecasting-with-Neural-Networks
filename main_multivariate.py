#####################################################################################################################################
#######################################             REQUIRED MODULES INSTALLATION       #############################################
## UNCOMMENT IF SOME PACKAGES ARE MISSING ##
# import subprocess
# import sys
# list_of_packages=['datetime','h5py','keras','matplotlib','numpy','pandas','pathlib','pmdarima','prettytable','python-math','python-time', 'requests','scikit-learn', 'scipy','sktime', 'statistics', 'tensorflow','seaborn','zipfile','yfinance']
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
###########################################   MULTIVARIATE SCRIPT  (DATA ANALYSIS)   ################################################
def data_analysis_multivariate(df, parent_folder, series_name, output_variable):
    ##CREATE FOLDERS##
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    data_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis/Multivariate/Figures'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Data Analysis/Multivariate/Tables'
    list_of_folders=[project_path,parent_folder,series_path,data_path,multivariate_folder,multivariate_figures,multivariate_tables]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import datetime
    import pandas as pd
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
    multivariate_stats=compute_statistics(df)
    ##CREATE TABLE##
    from prettytable import PrettyTable
    from pathlib import Path
    def stat_table(dataframe, table, path_name, output_variable=output_variable):
        """Save summary statistics to table"""
        stat_table = PrettyTable()
        stat_table.title = 'Dataset Summary Statistics'
        stat_table.field_names = ['Metric',*list(dataframe)]
        stat_table.add_row(['Mean',*table.iloc[0]])
        stat_table.add_row(['Median',*table.iloc[1]])
        stat_table.add_row(['Standard Deviation',*table.iloc[2]])
        stat_table.add_row(['Variance',*table.iloc[3]])
        Path(f"{path_name}/Dataset Summary Statistics ("+output_variable+").txt").write_text(str(stat_table))
        print(stat_table)
    stat_table(df, multivariate_stats, path_name=multivariate_tables)
    ############################################    MULTIVARIATE EXERCISE     ############################################
    ############################################    ORGANIZE DATA     ############################################
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set(font_scale=0.8)
    ##SAVE PLOTS OF SINGLE SERIES##
    for col in df.columns:
        fig=plt.figure()
        plt.plot(df[col], color='steelblue')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.title(col+' Time Series Plot')
        plt.tight_layout()
        plt.savefig(f"{multivariate_figures}/"+col+" Time Series Plot.png", format="png")
        plt.close()    
    ##CHECK CORRELATIONS BETWEEN SERIES AND PRINT HEATMAP##
    from pandas import DataFrame
    from pandas import concat
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
                names += [('%s' % (var_names[j])) for j in range(n_vars)]
            else:
                names += [('%s+%d' % (var_names[j], i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    cross_corr_df=data_transform(df, L=5)
    correlation=cross_corr_df.corr()
    correlation.to_csv(f"{multivariate_tables}/Lagged Correlations.csv", index=True, sep=',')
    correlation_plot=sns.heatmap(correlation, xticklabels=True, yticklabels=True)
    correlation_plot.set_xticklabels(correlation_plot.get_xticklabels(),rotation=45,horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(f"{multivariate_figures}/Correlation Heatmap.png", format="png", dpi=720)
    plt.close()
    ##SAVE ACF AND PACF PLOTS FOR EACH VARIABLE##
    for col in df.columns:
        plot_acf(df[col], title='Autocorrelation Function for '+col,lags=50)
        plt.xlabel('Lags')
        plt.ylabel('ACF')
        plt.savefig(f"{multivariate_figures}/Autocorrelation Function ("+col+").png")
        plt.close()
        plot_pacf(df[col], title='Partial Autocorrelation Function for '+col, lags=50)    
        plt.xlabel('Lags')
        plt.ylabel('PACF')
        plt.savefig(f"{multivariate_figures}/Partial Autocorrelation Function ("+col+").png")
        plt.close()
    ##CROSS-CORRELATION DIAGNOSTICS##
    for col in df.columns:
        fig=plt.figure()
        plt.xcorr(df[col], df[output_variable])
        plt.xlabel('Lags')
        plt.ylabel('Cross-correlation')
        plt.title('Cross Correlation Function for '+output_variable+' and '+col)
        plt.savefig(f"{multivariate_figures}/Cross-Correlation Function ("+output_variable+" and "+col+").png")
        plt.close()
    ##PLOT DIAGNOSTICS##
    def plot_diagnostics(y, title, lags=None, figsize=(12,8)):
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
        plot_pacf(y, lags=lags, ax=pacf_ax, method='ywm')
        sns.despine()
        plt.tight_layout()
        return ts_ax, acf_ax, pacf_ax
    for column in df:
        plot_diagnostics(df[column].dropna(), title=column, lags=50)
        plt.savefig(f"{multivariate_figures}/"+column+" Diagnostic Plot.png", format="png")
        plt.close()
    ##CREATE DIFFERENCED DATASET##
    df_diff = df.diff().dropna()
    ##PLOT DIAGNOSTICS##
    for column in df_diff:
        plot_diagnostics(df_diff[column].dropna(), title=column+' (Differenced)', lags=50)
        plt.savefig(f"{multivariate_figures}/"+column+" Diagnostic Plot (Differenced).png", format="png")
        plt.close()
    plt.close('all')
    ##PLOT FULL SERIES (LEVELS AND DIFFERENCED)##
    dates_format = mdates.DateFormatter('%Y')
    def ts_plot(df, df_diff, path_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax1.plot(df[column], color='orangered')
        ax2.plot(df_diff[col], color='dodgerblue')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax1.set_title(column+' Plot (Levels)')
        ax1.tick_params(labelsize=8)
        ax1.xaxis.set_major_formatter(dates_format)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax2.set_title(col+' Plot (Differenced)')
        ax2.tick_params(labelsize=8)
        ax2.xaxis.set_major_formatter(dates_format)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.tight_layout()
        plt.savefig(f"{path_name}/"+column+" Time-Series Plot.png", format='png')
    ##ITERATE OVER COLUMNS##
    for column, col in zip(df, df_diff):
        ts_plot(df, df_diff, path_name=multivariate_figures)
        plt.close()
    plt.close('all')
    ##CHECK FOR (NON)-STATIONARITY## 
    ##DEFINE ADF TEST##
    from statsmodels.tsa.stattools import adfuller
    def adfuller_test(series, signif=0.05, name='', verbose=False):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        description = []
        r = adfuller(series, autolag='AIC', regression='ct')
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
    def adf_test_calculator(dataset, series=''):
        description = []
        for name, column in dataset.iteritems():
            description.append(adfuller_test(column, name=column.name))
        description = "\n\n".join(description)
        Path(f"{multivariate_tables}/Augmented Dickey Fuller Test on Dataset ("+series+").txt").write_text(description)
    #PERFORM ADF TEST ON SERIES##
    adf_test_calculator(df, series='Multivariate')
    ##AND ON DIFFERENCED SERIES##
    adf_test_calculator(df_diff, series='Multivariate Differenced')
#####################################################################################################################################
##################################################   MULTIVARIATE SCRIPT  (ARIMAX)       ###############################################
def arimax(df, parent_folder, series_name, output_variable):
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
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Multivariate/Figures'
    multivariate_arima = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Multivariate/Figures/ARIMAX'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Baseline Models/Multivariate/Tables'
    list_of_folders=[project_path,parent_folder,series_path,model_path,multivariate_folder,multivariate_figures,multivariate_arima,multivariate_tables]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    import datetime
    import pandas as pd
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
    df=df.fillna(0)
    ##IMPORT MODULES##
    from prettytable import PrettyTable
    from pathlib import Path
    import matplotlib
    from matplotlib import pyplot as plt
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
    ##SET DEPENDENT AND INDEPENDENT VARIABLES##
    arima_train_endogenous=df_train[output_variable]
    arima_test_endogenous=df_test[output_variable]
    arima_train_exogenous=df_train.loc[:, df_train.columns!=output_variable]
    arima_test_exogenous=df_test.loc[:, df_test.columns!=output_variable]
    ############################################    ARIMA MODEL     ############################################
    ##FIT ARIMA MODEL (WITH MODEL SELECTION CRITERIA)##
    start = time()
    arima_model = pm.auto_arima(arima_train_endogenous, exogenous=arima_train_exogenous, start_p=1, start_q=1, test='adf', max_p=10, max_q=10, max_d=5, m=1,seasonal=False,start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    multivariate_arima_training_time=time()-start
    ##SAVE OPTIMIZED ORDER##
    order = arima_model.get_params()['order']
    ##SAVE PLOTS AND SUMMARY##
    Path(f"{multivariate_tables}/ARIMAX "+str(order)+" Output "+output_variable+" Series.txt").write_text(str(arima_model.summary()))
    arima_model.plot_diagnostics(figsize=(8,6))
    plt.savefig(f"{multivariate_arima}/ARIMAX "+str(order)+" Model Plot Diagnostics "+output_variable+".png", format="png")
    plt.close()
    ##PREDICT ON TRAINING SET##
    arima_train_model=ARIMA(arima_train_endogenous, exog=arima_train_exogenous, order=(order))
    arima_train_fit=arima_train_model.fit()
    arima_forecasts_train=ARIMAResults.predict(arima_train_fit)
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    arima_error_train = forecast_error(df_train[output_variable],arima_forecasts_train)
    arima_error_train = numpy.stack(arima_error_train, axis=0)
    arima_error_train = arima_error_train.reshape(arima_error_train.shape[0],1)
    ##PREDICT ON TEST SET##
    train_y=arima_train_endogenous.values.tolist()
    train_x=arima_train_exogenous.values.tolist()
    test_y=arima_test_endogenous.values.tolist()
    test_x=arima_test_exogenous.values.tolist()
    start = time()
    predictions = list()
    for i in range(len(test_y)):
        model = ARIMA(train_y, exog=train_x, order=(order))
        model_fit = model.fit()
        output = model_fit.forecast(exog=test_x[i])
        yhat = output[0]
        predictions.append(yhat)
        y_obs = test_y[i]
        train_y.append(y_obs)
        x_obs = test_x[i]
        train_x.append(x_obs)
        print('Forecasting observation', i+1, 'of', test_size, ' ---> Predicted Value=%f, Actual Value=%f' % (yhat, y_obs)+' ---> '+str(int(100 * (i+1) / test_size)) + '%'+' complete')
    multivariate_arima_test_time=time()-start
    multivariate_arima_training_test_time=multivariate_arima_training_time+multivariate_arima_test_time
    ##CONVERT TO DATAFRAME##
    arima_forecasts_test=pd.DataFrame(predictions, index=df_test.index)
    ##COMPUTE FORECAST ERROR##
    arima_error_test = forecast_error(df_test[output_variable],arima_forecasts_test)
    arima_error_test = numpy.stack(arima_error_test, axis=0)
    arima_error_test = arima_error_test.reshape(arima_error_test.shape[0],1)
    ##PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
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
    arima_metrics_train=performance_metrics_calculator(df_train[output_variable].to_numpy(), arima_forecasts_train.to_numpy())
    arima_metrics_test=performance_metrics_calculator(df_test[output_variable].to_numpy(), arima_forecasts_test.to_numpy())
    ##CREATE TABLE##
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+series_name+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+series_name+" Series).txt").write_text(str(table))
        print(table)
        return table
    performance_table(df, arima_metrics_train, path_name=multivariate_tables, name='ARIMAX '+str(order)+' In-Sample Performance')
    performance_table(df, arima_metrics_test, path_name=multivariate_tables, name='ARIMAX '+str(order)+' Out-of-Sample Performance')
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(df, actual, predicted, splits, path_name, model_name, label, color, output_variable=output_variable):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(output_variable)
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model - Actual vs. Predicted Values ("+output_variable+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##PERFORMANCE ON TRAINING SET##
    plot_performance(df, df_train[output_variable], arima_forecasts_train, 1, path_name=multivariate_arima, model_name='ARIMAX '+str(order)+' In-Sample', label='Training', color='lawngreen')
    ##PERFORMANCE ON TEST SET##
    plot_performance(df, df_test[output_variable], arima_forecasts_test, 1, path_name=multivariate_arima, model_name='ARIMAX '+str(order)+' Out-of-Sample', label='Test', color='orangered')
    plot_performance(df, df_test[output_variable], arima_forecasts_test, 5, path_name=multivariate_arima, model_name='ARIMAX '+str(order)+' Out-of-Sample', label='Test', color='orangered')
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
    scatter_plot(df_train[output_variable], arima_forecasts_train, model_name='ARIMAX '+str(order)+' In-Sample', path_name=multivariate_arima)
    scatter_plot(df_test[output_variable], arima_forecasts_test, model_name='ARIMAX '+str(order)+' Out-of-Sample', path_name=multivariate_arima)
    ##CREATE FULL MULTIVARIATE TABLES##
    ##TRAINING TIMES##
    multivariate_training_times=[multivariate_arima_training_test_time]
    multivariate_training_times=[round(item, 5) for item in multivariate_training_times]
    multivariate_training_times_table = PrettyTable()
    multivariate_training_times_table.title = 'ARIMAX Fitting Time ('+series_name+')'
    multivariate_training_times_table.field_names = [column_names[0],"ARIMAX "+str(order)]
    multivariate_training_times_table.add_row(['Training Time (seconds)', *multivariate_training_times])
    Path(f"{multivariate_tables}/ARIMAX Fitting Time.txt").write_text(str(multivariate_training_times_table))
    print(multivariate_training_times_table)
    ##IN-SAMPLE METRICS##
    multivariate_table_train = PrettyTable()
    multivariate_table_train.title = 'ARIMAX In-Sample Model Performances ('+series_name+')'
    multivariate_table_train.field_names = [column_names[0], "ARIMAX "+str(order)]
    for i in range(len(performance_metrics)):
        multivariate_table_train.add_row([performance_metrics[i], arima_metrics_train[i]])
    Path(f"{multivariate_tables}/ARIMAX In-Sample Model Performance.txt").write_text(str(multivariate_table_train))
    print(multivariate_table_train)
    ##OUT-OF-SAMPLE METRICS##
    multivariate_table_test = PrettyTable()
    multivariate_table_test.title = 'ARIMAX Out-of-Sample Model Performances ('+series_name+')'
    multivariate_table_test.field_names = [column_names[0], "ARIMAX "+str(order)]
    for i in range(len(performance_metrics)):
        multivariate_table_test.add_row([performance_metrics[i], arima_metrics_test[i]])
    Path(f"{multivariate_tables}/ARIMAX Out-of-Sample Model Performance.txt").write_text(str(multivariate_table_test))
    print(multivariate_table_test)
    return arima_error_train, arima_error_test
#####################################################################################################################################
#####################################           MULTIVARIATE SCRIPT (NEURAL NETWORKS)    ##############################################
#####################################################################################################################################
##FEED FORWARD NEURAL NETWORK (FFNN)##
def ffnn_multivariate(df, parent_folder, series_name, output_variable, L, ffnn_nodes1, ffnn_nodes2, ffnn_epochs, ffnn_batch, ffnn_optimizer, T=1):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures'
    multivariate_ffnn = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures/Feed-Forward Neural Network (FFNN)'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Tables'
    multivariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_figures,multivariate_ffnn,multivariate_tables,multivariate_models]
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
    df=df.fillna(0)
    from time import time
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
    ############################################    NEURAL NETWORK MODELS     ############################################
    ############################################    DATA PREPARATION     ############################################
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
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    ##DATA PREPARATION##
    features=df.shape[1]
    T=1
    ##TRANSFORM DATA TO TENSOR##
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
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T*features:]
    train_y_data=train_y_data[output_variable+" (t)"].to_frame()
    train_x_data=data_train.iloc[: , :L*features]
    test_y_data=data_test.iloc[: , -T*features:]
    test_y_data=test_y_data[output_variable+" (t)"].to_frame()
    test_x_data=data_test.iloc[: , :L*features]
    ##SCALING##
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    #################################################       FEED-FORWARD NN     ##############################################
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
    ffnn_best_model=ModelCheckpoint(f"{multivariate_models}/ffnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA AND RECORD TRAINING TIME##
    start=time()
    ffnn_fit=ffnn_model.fit(train_x, train_y, epochs=ffnn_epochs, batch_size=ffnn_batch, validation_split=0.2, verbose=2, callbacks=[ffnn_best_model])
    multivariate_ffnn_training_time=time()-start
    ##EVALUATE FIT DURING TRAINING##
    def loss_plot(model,  path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+output_variable+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+output_variable+" Series).png", format="png")
    loss_plot(ffnn_fit, path_name=multivariate_ffnn, name="FFNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(model, path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+output_variable+" Series).png", format="png")
    mae_plot(ffnn_fit, path_name=multivariate_ffnn, name='FFNN')
    ##LOAD BEST MODEL##
    ffnn_best_model=load_model(f"{multivariate_models}/ffnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{multivariate_models}/FFNN best model summary ("+series_name+").txt", 'w') as f:
        ffnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    ffnn_train_predict=ffnn_best_model.predict(train_x,verbose=2)
    ##FIT MODEL ON TEST DATA##
    ffnn_test_predict=ffnn_best_model.predict(test_x,verbose=2)
    ##INVERT SCALING AND RESHAPE VECTORS##
    train_y_inv = scaler_y.inverse_transform(train_y)
    test_y_inv = scaler_y.inverse_transform(test_y)
    ffnn_train_predict_inv=scaler_y.inverse_transform(ffnn_train_predict)
    ffnn_test_predict_inv=scaler_y.inverse_transform(ffnn_test_predict)
    ##COMPUTE PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
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
    ffnn_metrics_train=performance_metrics_calculator(train_y_inv, ffnn_train_predict_inv)
    ffnn_metrics_test=performance_metrics_calculator(test_y_inv, ffnn_test_predict_inv)
    ##CREATE TABLE##
    from prettytable import PrettyTable
    from pathlib import Path
    def performance_table(list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+series_name+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+series_name+" Series).txt").write_text(str(table))
        print(table)
        return table
    performance_table(ffnn_metrics_train,  path_name=multivariate_tables, name='FFNN - In-Sample')
    performance_table(ffnn_metrics_test,  path_name=multivariate_tables, name='FFNN - Out-of-Sample')
    ##REMOVE OBSERVATIONS FROM LOOK BACK WINDOW##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    ffnn_train_predict_inv_dataframe=pd.DataFrame(ffnn_train_predict_inv, index=df_train_for_plotting.index)
    ffnn_test_predict_inv_dataframe=pd.DataFrame(ffnn_test_predict_inv, index=df_test_for_plotting.index)
    ##FORECAST ERROR##
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    ffnn_error_train = forecast_error(df_train_for_plotting[output_variable],ffnn_train_predict_inv_dataframe)
    ffnn_error_test = forecast_error(df_test_for_plotting[output_variable], ffnn_test_predict_inv_dataframe)
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(actual, predicted, splits, path_name, model_name, label, color, output_variable=output_variable):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(output_variable)
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model - Actual vs. Predicted Values ("+output_variable+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##PLOTS (FULL)##
    plot_performance(df_train_for_plotting[output_variable], ffnn_train_predict_inv_dataframe, 1, path_name=multivariate_ffnn, model_name='FFNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], ffnn_test_predict_inv_dataframe, 1, path_name=multivariate_ffnn, model_name='FFNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS##
    plot_performance(df_train_for_plotting[output_variable], ffnn_train_predict_inv_dataframe, 5, path_name=multivariate_ffnn, model_name='FFNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], ffnn_test_predict_inv_dataframe, 5, path_name=multivariate_ffnn, model_name='FFNN - Out-of-Sample', label="Test", color='orangered')
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
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting[output_variable], ffnn_train_predict_inv_dataframe, model_name='FFNN (Layers='+str(len(ffnn_model.layers))+') In-Sample', path_name=multivariate_ffnn)
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting[output_variable], ffnn_test_predict_inv_dataframe, model_name='FFNN (Layers='+str(len(ffnn_model.layers))+') Out-of-Sample', path_name=multivariate_ffnn)
    plt.close('all')
    ffnn_layers=len(ffnn_model.layers)
    return multivariate_ffnn_training_time, ffnn_metrics_train, ffnn_metrics_test, ffnn_layers, ffnn_error_train, ffnn_error_test
#####################################################################################################################################
##RECURRENT NEURAL NETWORK (RNN)##
def rnn_multivariate(df, parent_folder, series_name, output_variable, L, rnn_nodes1, rnn_nodes2, rnn_epochs, rnn_batch, rnn_optimizer, T=1):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures'
    multivariate_rnn = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures/Recurrent Neural Network (RNN)'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Tables'
    multivariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_figures,multivariate_rnn,multivariate_tables,multivariate_models]
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
    df=df.fillna(0)
    from time import time
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
    ############################################    NEURAL NETWORK MODELS     ############################################
    ############################################    DATA PREPARATION     ############################################
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
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    ##DATA PREPARATION##
    features=df.shape[1]
    T=1
    ##TRANSFORM DATA TO TENSOR##
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
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T*features:]
    train_y_data=train_y_data[output_variable+" (t)"].to_frame()
    train_x_data=data_train.iloc[: , :L*features]
    test_y_data=data_test.iloc[: , -T*features:]
    test_y_data=test_y_data[output_variable+" (t)"].to_frame()
    test_x_data=data_test.iloc[: , :L*features]
    ##SCALING##
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##SEED##
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    #################################################       SIMPLE RNN     ##############################################
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    X_train = train_x.reshape((train_x.shape[0], L, features))
    X_test = test_x.reshape((test_x.shape[0], L, features))
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
    rnn_best_model=ModelCheckpoint(f"{multivariate_models}/rnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    rnn_fit=rnn_model.fit(X_train, train_y, epochs=rnn_epochs, batch_size=rnn_batch, validation_split=0.2, verbose=2, callbacks=[rnn_best_model])
    multivariate_rnn_training_time=time()-start
    ##EVALUATE FIT DURING TRAINING##
    def loss_plot(model, path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+output_variable+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+output_variable+" Series).png", format="png")
    loss_plot(rnn_fit, path_name=multivariate_rnn, name="Simple RNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(model, path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+output_variable+" Series).png", format="png")
    mae_plot(rnn_fit, path_name=multivariate_rnn, name='Simple RNN')
    ##LOAD BEST MODEL##
    rnn_best_model=load_model(f"{multivariate_models}/rnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{multivariate_models}/RNN best model summary ("+series_name+").txt", 'w') as f:
        rnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))    
    ##FIT MODEL ON TRAINING DATA##
    rnn_train_predict=rnn_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    rnn_test_predict=rnn_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING AND RESHAPE VECTORS##
    y_train_inv = scaler_y.inverse_transform(train_y)
    y_test_inv = scaler_y.inverse_transform(test_y)
    rnn_train_predict_inv=scaler_y.inverse_transform(rnn_train_predict)
    rnn_test_predict_inv=scaler_y.inverse_transform(rnn_test_predict)
    ##PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
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
    rnn_metrics_train=performance_metrics_calculator(y_train_inv, rnn_train_predict_inv)
    rnn_metrics_test=performance_metrics_calculator(y_test_inv, rnn_test_predict_inv)
    ##CREATE TABLE##
    from prettytable import PrettyTable
    from pathlib import Path
    def performance_table(list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+series_name+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+series_name+" Series).txt").write_text(str(table))
        print(table)
        return table
    performance_table(rnn_metrics_train,  path_name=multivariate_tables, name='RNN - In-Sample')
    performance_table(rnn_metrics_test,  path_name=multivariate_tables, name='RNN - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    rnn_train_predict_inv_dataframe=pd.DataFrame(rnn_train_predict_inv, index=df_train_for_plotting.index)
    rnn_test_predict_inv_dataframe=pd.DataFrame(rnn_test_predict_inv, index=df_test_for_plotting.index)
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    rnn_error_train = forecast_error(df_train_for_plotting[output_variable],rnn_train_predict_inv_dataframe)
    rnn_error_test = forecast_error(df_test_for_plotting[output_variable], rnn_test_predict_inv_dataframe)
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(actual, predicted, splits, path_name, model_name, label, color, output_variable=output_variable):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(output_variable)
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model - Actual vs. Predicted Values ("+output_variable+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##PLOTS (FULL)##
    plot_performance(df_train_for_plotting[output_variable], rnn_train_predict_inv_dataframe, 1, path_name=multivariate_rnn, model_name='RNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], rnn_test_predict_inv_dataframe, 1, path_name=multivariate_rnn, model_name='RNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df_train_for_plotting[output_variable], rnn_train_predict_inv_dataframe, 5, path_name=multivariate_rnn, model_name='RNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], rnn_test_predict_inv_dataframe, 5, path_name=multivariate_rnn, model_name='RNN - Out-of-Sample', label="Test", color='orangered')
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
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting[output_variable], rnn_train_predict_inv_dataframe, model_name='RNN (Layers='+str(len(rnn_model.layers))+') In-Sample', path_name=multivariate_rnn)
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting[output_variable], rnn_test_predict_inv_dataframe, model_name='RNN (Layers='+str(len(rnn_model.layers))+') Out-of-Sample', path_name=multivariate_rnn)
    plt.close('all')
    rnn_layers = len(rnn_model.layers)
    return multivariate_rnn_training_time, rnn_metrics_train, rnn_metrics_test, rnn_layers, rnn_error_train, rnn_error_test
#####################################################################################################################################
##LONG SHORT-TERM MEMORY (LSTM)##
def lstm_multivariate(df, parent_folder, series_name, output_variable, L, lstm_nodes1, lstm_nodes2, lstm_epochs, lstm_batch, lstm_optimizer, T=1):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures'
    multivariate_lstm = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures/Long Short-Term Memory (LSTM)'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Tables'
    multivariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_figures,multivariate_lstm,multivariate_tables,multivariate_models]
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
    df=df.fillna(0)
    from time import time
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
    ############################################    NEURAL NETWORK MODELS     ############################################
    ############################################    DATA PREPARATION     ############################################
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
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    ##DATA PREPARATION##
    features=df.shape[1]
    T=1
    ##TRANSFORM DATA TO TENSOR##
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
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T*features:]
    train_y_data=train_y_data[output_variable+" (t)"].to_frame()
    train_x_data=data_train.iloc[: , :L*features]
    test_y_data=data_test.iloc[: , -T*features:]
    test_y_data=test_y_data[output_variable+" (t)"].to_frame()
    test_x_data=data_test.iloc[: , :L*features]
    ##SCALING##
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    X_train = train_x.reshape((train_x.shape[0], L, features))
    X_test = test_x.reshape((test_x.shape[0], L, features))
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
    lstm_best_model=ModelCheckpoint(f"{multivariate_models}/lstm_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    lstm_fit=lstm_model.fit(X_train, train_y, epochs=lstm_epochs, batch_size=lstm_batch, validation_split=0.2, verbose=2, callbacks=[lstm_best_model])
    multivariate_lstm_training_time=time()-start
    ##EVALUATE FIT##
    def loss_plot(model,  path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+output_variable+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+output_variable+" Series).png", format="png")
    loss_plot(lstm_fit, path_name=multivariate_lstm, name="LSTM")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(model, path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+output_variable+" Series).png", format="png")
    mae_plot(lstm_fit, path_name=multivariate_lstm, name='LSTM')
    ##LOAD BEST MODEL##
    lstm_best_model=load_model(f"{multivariate_models}/lstm_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{multivariate_models}/LSTM best model summary ("+series_name+").txt", 'w') as f:
        lstm_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    lstm_train_predict=lstm_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    lstm_test_predict=lstm_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING##
    y_train_inv = scaler_y.inverse_transform(train_y)
    y_test_inv = scaler_y.inverse_transform(test_y)
    lstm_train_predict_inv=scaler_y.inverse_transform(lstm_train_predict)
    lstm_test_predict_inv=scaler_y.inverse_transform(lstm_test_predict)
    ##PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
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
    lstm_metrics_train=performance_metrics_calculator(y_train_inv, lstm_train_predict_inv)
    lstm_metrics_test=performance_metrics_calculator(y_test_inv, lstm_test_predict_inv)
    ##CREATE TABLES##
    from prettytable import PrettyTable
    from pathlib import Path
    def performance_table(list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+series_name+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+series_name+" Series).txt").write_text(str(table))
        print(table)
        return table
    performance_table(lstm_metrics_train,  path_name=multivariate_tables, name='LSTM - In-Sample')
    performance_table(lstm_metrics_test,  path_name=multivariate_tables, name='LSTM - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    lstm_train_predict_inv_dataframe=pd.DataFrame(lstm_train_predict_inv, index=df_train_for_plotting.index)
    lstm_test_predict_inv_dataframe=pd.DataFrame(lstm_test_predict_inv, index=df_test_for_plotting.index)
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    lstm_error_train = forecast_error(df_train_for_plotting[output_variable],lstm_train_predict_inv_dataframe)
    lstm_error_test = forecast_error(df_test_for_plotting[output_variable], lstm_test_predict_inv_dataframe)
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(actual, predicted, splits, path_name, model_name, label, color, output_variable=output_variable):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(output_variable)
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model - Actual vs. Predicted Values ("+output_variable+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##PLOTS (FULL)##
    plot_performance(df_train_for_plotting[output_variable], lstm_train_predict_inv_dataframe, 1, path_name=multivariate_lstm, model_name='LSTM - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], lstm_test_predict_inv_dataframe, 1, path_name=multivariate_lstm, model_name='LSTM - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df_train_for_plotting[output_variable], lstm_train_predict_inv_dataframe, 5, path_name=multivariate_lstm, model_name='LSTM - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], lstm_test_predict_inv_dataframe, 5, path_name=multivariate_lstm, model_name='LSTM - Out-of-Sample', label="Test", color='orangered')
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
    ##SCATTER PLOT (TRAINING)##
    scatter_plot(df_train_for_plotting[output_variable], lstm_train_predict_inv_dataframe, model_name='LSTM (Layers='+str(len(lstm_model.layers))+') In-Sample', path_name=multivariate_lstm)
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting[output_variable], lstm_test_predict_inv_dataframe, model_name='LSTM (Layers='+str(len(lstm_model.layers))+') Out-of-Sample', path_name=multivariate_lstm)
    plt.close('all')
    lstm_layers = len(lstm_model.layers)
    return multivariate_lstm_training_time, lstm_metrics_train, lstm_metrics_test, lstm_layers, lstm_error_train, lstm_error_test
#####################################################################################################################################
##GATED RECURRENT UNIT (GRU)##
def gru_multivariate(df, parent_folder, series_name, output_variable, L, gru_nodes1, gru_nodes2, gru_epochs, gru_batch, gru_optimizer, T=1):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures'
    multivariate_gru = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures/Gated Recurrent Unit (GRU)'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Tables'
    multivariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_figures,multivariate_gru,multivariate_tables,multivariate_models]
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
    df=df.fillna(0)
    from time import time
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
    ############################################    NEURAL NETWORK MODELS     ############################################
    ############################################    DATA PREPARATION     ############################################
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
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    ##DATA PREPARATION##
    features=df.shape[1]
    T=1
    ##TRANSFORM DATA TO TENSOR##
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
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T*features:]
    train_y_data=train_y_data[output_variable+" (t)"].to_frame()
    train_x_data=data_train.iloc[: , :L*features]
    test_y_data=data_test.iloc[: , -T*features:]
    test_y_data=test_y_data[output_variable+" (t)"].to_frame()
    test_x_data=data_test.iloc[: , :L*features]
    ##SCALING##
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    X_train = train_x.reshape((train_x.shape[0], L, features))
    X_test = test_x.reshape((test_x.shape[0], L, features))
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
    gru_best_model=ModelCheckpoint(f"{multivariate_models}/gru_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    gru_fit=gru_model.fit(X_train, train_y, epochs=gru_epochs, batch_size=gru_batch, validation_split=0.2, verbose=2, callbacks=[gru_best_model])
    multivariate_gru_training_time=time()-start
    ##EVALUATE FIT##
    def loss_plot(model,  path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+output_variable+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+output_variable+" Series).png", format="png")
    loss_plot(gru_fit, path_name=multivariate_gru, name="GRU")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(model, path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+output_variable+" Series).png", format="png")
    mae_plot(gru_fit, path_name=multivariate_gru, name='GRU')
    ##LOAD BEST MODEL##
    gru_best_model=load_model(f"{multivariate_models}/gru_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{multivariate_models}/GRU best model summary ("+series_name+").txt", 'w') as f:
        gru_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    gru_train_predict=gru_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    gru_test_predict=gru_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING##
    y_train_inv = scaler_y.inverse_transform(train_y)
    y_test_inv = scaler_y.inverse_transform(test_y)
    gru_train_predict_inv=scaler_y.inverse_transform(gru_train_predict)
    gru_test_predict_inv=scaler_y.inverse_transform(gru_test_predict)
    ##PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
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
    gru_metrics_train=performance_metrics_calculator(y_train_inv, gru_train_predict_inv)
    gru_metrics_test=performance_metrics_calculator(y_test_inv, gru_test_predict_inv)
    ##CREATE TABLES##
    from prettytable import PrettyTable
    from pathlib import Path
    def performance_table(list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+series_name+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+series_name+" Series).txt").write_text(str(table))
        print(table)
        return table
    performance_table(gru_metrics_train, path_name=multivariate_tables, name='GRU - In-Sample')
    performance_table(gru_metrics_test, path_name=multivariate_tables, name='GRU - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    gru_train_predict_inv_dataframe=pd.DataFrame(gru_train_predict_inv, index=df_train_for_plotting.index)
    gru_test_predict_inv_dataframe=pd.DataFrame(gru_test_predict_inv, index=df_test_for_plotting.index)
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    gru_error_train = forecast_error(df_train_for_plotting[output_variable],gru_train_predict_inv_dataframe)
    gru_error_test = forecast_error(df_test_for_plotting[output_variable], gru_test_predict_inv_dataframe)
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(actual, predicted, splits, path_name, model_name, label, color, output_variable=output_variable):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(output_variable)
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model - Actual vs. Predicted Values ("+output_variable+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##PLOTS (FULL)##
    plot_performance(df_train_for_plotting[output_variable], gru_train_predict_inv_dataframe, 1, path_name=multivariate_gru, model_name='GRU - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], gru_test_predict_inv_dataframe, 1, path_name=multivariate_gru, model_name='GRU - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df_train_for_plotting[output_variable], gru_train_predict_inv_dataframe, 5, path_name=multivariate_gru, model_name='GRU - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], gru_test_predict_inv_dataframe, 5, path_name=multivariate_gru, model_name='GRU - Out-of-Sample', label="Test", color='orangered')
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
    ##SCATTER PLOT##
    scatter_plot(df_train_for_plotting[output_variable], gru_train_predict_inv_dataframe, model_name='GRU (Layers='+str(len(gru_model.layers))+') In-Sample', path_name=multivariate_gru)
    ##SCATTER PLOT##
    scatter_plot(df_test_for_plotting[output_variable], gru_test_predict_inv_dataframe, model_name='GRU (Layers='+str(len(gru_model.layers))+') Out-of-Sample', path_name=multivariate_gru)
    plt.close('all')
    gru_layers = len(gru_model.layers)
    return multivariate_gru_training_time, gru_metrics_train, gru_metrics_test, gru_layers, gru_error_train, gru_error_test
#####################################################################################################################################
##CONVOLUTIONAL NEURAL NETWORK (CNN)##
def cnn_multivariate(df, parent_folder, series_name, output_variable, L, cnn_filters_1, cnn_filters_2, cnn_dense_nodes, cnn_epochs, cnn_batch, cnn_optimizer, cnn_kernel_1=3, cnn_kernel_2=3, cnn_pool_size=2, T=1):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_figures = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures'
    multivariate_cnn = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Figures/Convolutional Neural Network (CNN)'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Tables'
    multivariate_models= pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Models'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_figures,multivariate_cnn,multivariate_tables,multivariate_models]
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
    df=df.fillna(0)
    from time import time
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
    ############################################    NEURAL NETWORK MODELS     ############################################
    ############################################    DATA PREPARATION     ############################################
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
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat
    ##DATA PREPARATION##
    features=df.shape[1]
    T=1
    ##TRANSFORM DATA TO TENSOR##
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
    data_train=data_transform(df_train, L,T)
    data_test=data_transform(df_test,L,T)
    train_y_data=data_train.iloc[: , -T*features:]
    train_y_data=train_y_data[output_variable+" (t)"].to_frame()
    train_x_data=data_train.iloc[: , :L*features]
    test_y_data=data_test.iloc[: , -T*features:]
    test_y_data=test_y_data[output_variable+" (t)"].to_frame()
    test_x_data=data_test.iloc[: , :L*features]
    ##SCALING##
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y=scaler_y.fit_transform(train_y_data)
    train_x=scaler_x.fit_transform(train_x_data)
    test_y=scaler_y.fit_transform(test_y_data)
    test_x=scaler_x.fit_transform(test_x_data)
    ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
    X_train = train_x.reshape((train_x.shape[0], L, features))
    X_test = test_x.reshape((test_x.shape[0], L, features))
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
    cnn_best_model=ModelCheckpoint(f"{multivariate_models}/cnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ##FIT MODEL ON TRAINING DATA##
    start=time()
    cnn_fit=cnn_model.fit(X_train, train_y, epochs=cnn_epochs, batch_size=cnn_batch, validation_split=0.2, verbose=2, callbacks=[cnn_best_model])
    multivariate_cnn_training_time=time()-start
    ##EVALUATE FIT##
    def loss_plot(model,  path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Loss against Validation Loss and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['loss'], label='Training loss')
        plt.plot(model.history['val_loss'], label='Validation loss')
        plt.title(name+' Training Loss vs. Validation Loss ('+output_variable+' Series)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training Loss vs. Validation Loss ("+output_variable+" Series).png", format="png")
    loss_plot(cnn_fit, path_name=multivariate_cnn, name="CNN")
    ##EVALUATE TRAINING AND VALIDATION MAE##
    def mae_plot(model, path_name, name, output_variable=output_variable):
        """Plot Neural Network Training Mean Absolute Error (MAE) against Validation and save figure to specified path"""
        fig=plt.figure()
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(name+' Training MAE vs. Validation MAE')
        plt.legend(loc='best')
        plt.savefig(f"{path_name}/"+name+" - Training MAE vs. Validation MAE ("+output_variable+" Series).png", format="png")
    mae_plot(cnn_fit, path_name=multivariate_cnn, name='CNN')
    ##LOAD BEST MODEL##
    cnn_best_model=load_model(f"{multivariate_models}/cnn_best_model.h5")
    ##SAVE MODEL SUMMARY##
    with open(f"{multivariate_models}/CNN best model summary ("+series_name+").txt", 'w') as f:
        cnn_best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    ##FIT MODEL ON TRAINING DATA##
    cnn_train_predict=cnn_best_model.predict(X_train,verbose=2)
    ##FIT MODEL ON TEST DATA##
    cnn_test_predict=cnn_best_model.predict(X_test,verbose=2)
    ##INVERT SCALING##
    y_train_inv = scaler_y.inverse_transform(train_y)
    y_test_inv = scaler_y.inverse_transform(test_y)
    cnn_train_predict_inv=scaler_y.inverse_transform(cnn_train_predict)
    cnn_test_predict_inv=scaler_y.inverse_transform(cnn_test_predict)
    ##PERFORMANCE METRICS##
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    from math import sqrt
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
    cnn_metrics_train=performance_metrics_calculator(y_train_inv, cnn_train_predict_inv)
    cnn_metrics_test=performance_metrics_calculator(y_test_inv, cnn_test_predict_inv)
    ##CREATE TABLES##
    from prettytable import PrettyTable
    from pathlib import Path
    def performance_table(df, list, path_name, name=''):
        """Create Table Comparing Different Model Performances for each metric"""
        table = PrettyTable()
        table.title = name+' Model Performance on '+series_name+' Series'
        table.field_names = [column_names[0], column_names[1]]
        for i in range(len(performance_metrics)):
            table.add_row([performance_metrics[i],list[i]])
        Path(f"{path_name}/"+name+" Model Performance Metrics ("+series_name+" Series).txt").write_text(str(table))
        print(table)
        return table
    performance_table(df, cnn_metrics_train, path_name=multivariate_tables, name='CNN - In-Sample')
    performance_table(df, cnn_metrics_test, path_name=multivariate_tables, name='CNN - Out-of-Sample')
    ##PLOTTING DATAFRAME##
    df_train_for_plotting=df_train.iloc[L:,:]
    df_test_for_plotting=df_test.iloc[L:,:]
    cnn_train_predict_inv_dataframe=pd.DataFrame(cnn_train_predict_inv, index=df_train_for_plotting.index)
    cnn_test_predict_inv_dataframe=pd.DataFrame(cnn_test_predict_inv, index=df_test_for_plotting.index)
    def forecast_error(y_true,y_pred):
        y_true=y_true.values
        y_pred=y_pred.values
        y_error=[y_true[i]-y_pred[i] for i in range(len(y_true))]
        return y_error
    cnn_error_train = forecast_error(df_train_for_plotting[output_variable], cnn_train_predict_inv_dataframe)
    cnn_error_test = forecast_error(df_test_for_plotting[output_variable], cnn_test_predict_inv_dataframe)
    ##PLOT PERFORMANCE##
    from matplotlib.ticker import MaxNLocator
    def plot_performance(actual, predicted, splits, path_name, model_name, label, color, output_variable=output_variable):
        """Plot Model performance choosing how many plots to create and save figures to specified path"""
        test_list=numpy.array_split(actual, splits)
        pred_list=numpy.array_split(predicted, splits)
        for test, pred in zip(test_list, pred_list):
            fig= plt.figure()
            plt.plot(test, label=label, color=color, linestyle='solid')
            plt.plot(pred, label="Prediction", color='black', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel(output_variable)
            plt.legend(loc='best')
            plt.title(model_name+' - Actual vs. Predicted Values\n('+str(test.index[0])+' - '+str(test.index[-1])+')')
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(15))
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(f"{path_name}/"+model_name+" Model - Actual vs. Predicted Values ("+output_variable+") "+str(test.index[0])+" - "+str(test.index[-1])+".png", format="png")
            plt.close()
    ##PLOTS (FULL)##
    plot_performance(df_train_for_plotting[output_variable], cnn_train_predict_inv_dataframe, 1, path_name=multivariate_cnn, model_name='CNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], cnn_test_predict_inv_dataframe, 1, path_name=multivariate_cnn, model_name='CNN - Out-of-Sample', label="Test", color='orangered')
    ##PLOTS (DETAILS)##
    plot_performance(df_train_for_plotting[output_variable], cnn_train_predict_inv_dataframe, 5, path_name=multivariate_cnn, model_name='CNN - In Sample', label='Training', color='lawngreen')
    plot_performance(df_test_for_plotting[output_variable], cnn_test_predict_inv_dataframe, 5, path_name=multivariate_cnn, model_name='CNN - Out-of-Sample', label="Test", color='orangered')
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
    ##SCATTER PLOT (TRAIN)##
    scatter_plot(df_test_for_plotting[output_variable], cnn_test_predict_inv_dataframe, model_name='CNN (Layers='+str(len(cnn_model.layers))+') In-Sample', path_name=multivariate_cnn)
    ##SCATTER PLOT (TEST)##
    scatter_plot(df_test_for_plotting[output_variable], cnn_test_predict_inv_dataframe, model_name='CNN (Layers='+str(len(cnn_model.layers))+') Out-of-Sample', path_name=multivariate_cnn)
    plt.close('all')
    cnn_layers = len(cnn_model.layers)
    return multivariate_cnn_training_time, cnn_metrics_train, cnn_metrics_test, cnn_layers, cnn_error_train, cnn_error_test
#####################################################################################################################################
####################################              CREATE MULTIVARIATE TABLES              ###########################################
def multivariate_tables(multivariate_ffnn_training_time, multivariate_rnn_training_time, multivariate_lstm_training_time, 
                      multivariate_gru_training_time, multivariate_cnn_training_time,
                      ffnn_metrics_train, rnn_metrics_train, lstm_metrics_train, gru_metrics_train, cnn_metrics_train,
                      ffnn_metrics_test, rnn_metrics_test, lstm_metrics_test, gru_metrics_test, cnn_metrics_test,
                      ffnn_layers, rnn_layers, lstm_layers, gru_layers, cnn_layers,
                      parent_folder, series_name):
    import pathlib
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_tables = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Tables'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_tables]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    from prettytable import PrettyTable
    from pathlib import Path
    performance_metrics=['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)', 'Mean Directional Accuracy (MDA)','R-Squared (R^2)']
    column_names=['Metric', 'Value']
    ##TRAINING TIMES##
    multivariate_training_times=[multivariate_ffnn_training_time, multivariate_rnn_training_time, multivariate_lstm_training_time, multivariate_gru_training_time, multivariate_cnn_training_time]
    multivariate_training_times=[round(item, 5) for item in multivariate_training_times]
    multivariate_training_times_table = PrettyTable()
    multivariate_training_times_table.title = 'Multivariate Models Training Times ('+series_name+')'
    multivariate_training_times_table.field_names = [column_names[0],"Feed-Forward NN "+"(Layers="+str(ffnn_layers)+")", "Simple RNN "+"(Layers="+str(rnn_layers)+")", "LSTM "+"(Layers="+str(lstm_layers)+")", "GRU "+"(Layers="+str(gru_layers)+")", "CNN "+"(Layers="+str(cnn_layers)+")"]
    multivariate_training_times_table.add_row(['Training Time (seconds)', *multivariate_training_times])
    Path(f"{multivariate_tables}/Multivariate Models Training Times.txt").write_text(str(multivariate_training_times_table))
    print(multivariate_training_times_table)
    ##IN-SAMPLE METRICS##
    multivariate_table_train = PrettyTable()
    multivariate_table_train.title = 'Multivariate In-Sample Model Performances ('+series_name+')'
    multivariate_table_train.field_names = [column_names[0],"Feed-Forward NN "+"(Layers="+str(ffnn_layers)+")", "Simple RNN "+"(Layers="+str(rnn_layers)+")", "LSTM "+"(Layers="+str(lstm_layers)+")", "GRU "+"(Layers="+str(gru_layers)+")", "CNN "+"(Layers="+str(cnn_layers)+")"]
    for i in range(len(performance_metrics)):
        multivariate_table_train.add_row([performance_metrics[i], ffnn_metrics_train[i],rnn_metrics_train[i],lstm_metrics_train[i],gru_metrics_train[i],cnn_metrics_train[i]])
    Path(f"{multivariate_tables}/Multivariate In-Sample Model Performance.txt").write_text(str(multivariate_table_train))
    print(multivariate_table_train)
    ##OUT-OF-SAMPLE METRICS##
    multivariate_table_test = PrettyTable()
    multivariate_table_test.title = 'Multivariate Out-of-Sample Model Performances ('+series_name+')'
    multivariate_table_test.field_names = [column_names[0],"Feed-Forward NN "+"(Layers="+str(ffnn_layers)+")", "Simple RNN "+"(Layers="+str(rnn_layers)+")", "LSTM "+"(Layers="+str(lstm_layers)+")", "GRU "+"(Layers="+str(gru_layers)+")", "CNN "+"(Layers="+str(cnn_layers)+")"]
    for i in range(len(performance_metrics)):
        multivariate_table_test.add_row([performance_metrics[i], ffnn_metrics_test[i],rnn_metrics_test[i],lstm_metrics_test[i],gru_metrics_test[i],cnn_metrics_test[i]])
    Path(f"{multivariate_tables}/Multivariate Out-of-Sample Model Performance.txt").write_text(str(multivariate_table_test))
    print(multivariate_table_test)
#####################################################################################################################################
####################################            STATISTICAL SIGNIFICANCE TEST             ###########################################
def diebold_mariano_test(parent_folder, series_name, L_ffnn,L_rnn,L_lstm,L_gru,L_cnn, arima_error_train, arima_error_test, ffnn_error_train,ffnn_error_test,
                         rnn_error_train,rnn_error_test,lstm_error_train,lstm_error_test,gru_error_train,gru_error_test,
                         cnn_error_train,cnn_error_test):
    import pathlib
    from dm_test import dm_test
    project_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project'
    parent_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder
    series_path = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    diebold_mariano_test_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Diebold-Mariano Test'    
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,diebold_mariano_test_folder]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    from pathlib import Path
    from matplotlib import pyplot as plt
    models=['FFNN','RNN', 'LSTM', 'GRU', 'CNN']
    in_sample_nn=[ffnn_error_train,rnn_error_train,lstm_error_train,gru_error_train,cnn_error_train]
    out_sample_nn=[ffnn_error_test,rnn_error_test,lstm_error_test,gru_error_test,cnn_error_test]
    window_size=[L_ffnn,L_rnn,L_lstm,L_gru,L_cnn]
    ##IN-SAMPLE##
    in_sample_dm=[]
    in_sample_pval=[]
    for nn_pred_in, L in zip(in_sample_nn, window_size):
        dm_stat, dm_pval = dm_test(arima_error_train[L:],nn_pred_in, h=1)
        in_sample_dm.append(dm_stat)
        in_sample_pval.append(dm_pval)
    with open(f"{diebold_mariano_test_folder}/Diebold-Mariano Test Output (In-Sample).txt", 'w') as f:
        f.write("Baseline Comparison: ARIMAX Model\n")
        for model, dm, pval in zip(models, in_sample_dm, in_sample_pval):
            f.write("%s Model --> Diebold-Mariano Statistic: %f --> p-value: %f\n" % (model, dm, pval))
    ##PLOT LOSS DIFFERENTIALS##
    for nn_pred_in, L, model in zip(in_sample_nn, window_size, models):
        plt.figure()
        plt.plot(numpy.subtract(arima_error_train[L:],nn_pred_in))
        plt.title('In-sample Loss Differentials Plots')
        plt.ylabel('Loss Differential (ARIMA - '+model+')')
        plt.xlabel('Observation')
        plt.savefig(f"{diebold_mariano_test_folder}/In-sample Loss Differential Plot (ARIMA - "+model+").png")
    ##OUT-OF-SAMPLE##
    out_sample_dm=[]
    out_sample_pval=[]
    for nn_pred_out, L in zip(out_sample_nn, window_size):
        dm_stat, dm_pval = dm_test(arima_error_test[L:],nn_pred_out, h=1)
        out_sample_dm.append(dm_stat)
        out_sample_pval.append(dm_pval)
    with open(f"{diebold_mariano_test_folder}/Diebold-Mariano Test Output (Out-of-Sample).txt", 'w') as f:
        f.write("Baseline Comparison: ARIMAX Model\n")
        for model, dm, pval in zip(models, out_sample_dm, out_sample_pval):
            f.write("%s Model --> Diebold-Mariano Statistic: %f --> p-value: %f\n" % (model, dm, pval))
    ##PLOT LOSS DIFFERENTIALS##
    for nn_pred_out, L, model in zip(out_sample_nn, window_size, models):
        plt.figure()
        plt.plot(numpy.subtract(arima_error_test[L:],nn_pred_out))
        plt.title('Out-of-sample Loss Differentials Plots')
        plt.ylabel('Loss Differential (ARIMA - '+model+')')
        plt.xlabel('Observation')
        plt.savefig(f"{diebold_mariano_test_folder}/Out-of-sample Loss Differential Plot (ARIMA - "+model+").png")
#####################################################################################################################################
##################################################   GRID SEARCH FUNCTIONS     ######################################################
##FEED FORWARD NEURAL NETWORK##
def grid_search_ffnn_multivariate(df, parent_folder, series_name, output_variable, lags, nodes1, nodes2, optimizer, epochs, batch):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def ffnn_multivariate(config, df):
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        tensorflow.random.set_seed(seed_value)
        L, ffnn_nodes1, ffnn_nodes2, ffnn_epochs, ffnn_batch, ffnn_optimizer = config
        def train_test (dataset, train):
            """Define and operate train-test split on dataset and return the length of the test set as scalar"""
            df=dataset.astype(float)
            train_size = int(len(df) * train)
            test_size = len(df) - train_size
            df_train, df_test = df[0:train_size], df[train_size:len(df)]
            return df_train, df_test, test_size
        df_train, df_test, test_size = train_test(df, train=0.8)
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        from pandas import DataFrame
        from pandas import concat
        ##DATA PREPARATION##
        features=df.shape[1]
        T=1
        ##TRANSFORM DATA TO TENSOR##
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
        data_train=data_transform(df_train, L,T)
        data_test=data_transform(df_test,L,T)
        train_y_data=data_train.iloc[: , -T*features:]
        train_y_data=train_y_data[output_variable+" (t)"].to_frame()
        train_x_data=data_train.iloc[: , :L*features]
        test_y_data=data_test.iloc[: , -T*features:]
        test_y_data=test_y_data[output_variable+" (t)"].to_frame()
        test_x_data=data_test.iloc[: , :L*features]
        ##SCALING##
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        #################################################       FEED-FORWARD NN     ##############################################
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
        ffnn_best_model=ModelCheckpoint(f"{multivariate_grid_search}/ffnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA AND RECORD TRAINING TIME##
        ffnn_fit=ffnn_model.fit(train_x, train_y, epochs=ffnn_epochs, batch_size=ffnn_batch, validation_split=0.2, verbose=2, callbacks=[ffnn_best_model])
        ##LOAD BEST MODEL##
        ffnn_best_model=load_model(f"{multivariate_grid_search}/ffnn_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        ffnn_test_predict=ffnn_best_model.predict(test_x,verbose=2)        
        ##INVERT SCALING##
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
        scores = ffnn_multivariate(config, df)
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
    ##REMOVE MISSING VALUES##
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
    with open(f"{multivariate_grid_search}/FFNN - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Multivariate Feed-Forward Neural Network Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\n[Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##RECURRENT NEURAL NETWORK##
def grid_search_rnn_multivariate(df, parent_folder, series_name, output_variable, lags, nodes1, nodes2, optimizer, epochs, batch):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def rnn_multivariate(config, df):
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
        ##DATA PREPARATION##
        features=df.shape[1]
        T=1
        ##TRANSFORM DATA TO TENSOR##
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
        data_train=data_transform(df_train, L,T)
        data_test=data_transform(df_test,L,T)
        train_y_data=data_train.iloc[: , -T*features:]
        train_y_data=train_y_data[output_variable+" (t)"].to_frame()
        train_x_data=data_train.iloc[: , :L*features]
        test_y_data=data_test.iloc[: , -T*features:]
        test_y_data=test_y_data[output_variable+" (t)"].to_frame()
        test_x_data=data_test.iloc[: , :L*features]
        ##SCALING##
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
        X_train = train_x.reshape((train_x.shape[0], L, features))
        X_test = test_x.reshape((test_x.shape[0], L, features))
        #################################################       RNN     ##############################################
        from keras.models import Sequential
        from keras.layers import SimpleRNN, Dense
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
        rnn_best_model=ModelCheckpoint(f"{multivariate_grid_search}/rnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        rnn_fit=rnn_model.fit(X_train, train_y, epochs=rnn_epochs, batch_size=rnn_batch, validation_split=0.2, verbose=2, callbacks=[rnn_best_model])
        ##LOAD BEST MODEL##
        rnn_best_model=load_model(f"{multivariate_grid_search}/rnn_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        rnn_test_predict=rnn_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(test_y)
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
        scores = rnn_multivariate(config, df)
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
    with open(f"{multivariate_grid_search}/RNN - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Multivariate Recurrent Neural Network (RNN) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel Configuration:[Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##LONG SHORT-TERM MEMORY##
def grid_search_lstm_multivariate(df, parent_folder, series_name, output_variable, lags, nodes1, nodes2, optimizer, epochs, batch):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def lstm_multivariate(config, df):
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
        ##DATA PREPARATION##
        features=df.shape[1]
        T=1
        ##TRANSFORM DATA TO TENSOR##
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
        data_train=data_transform(df_train, L,T)
        data_test=data_transform(df_test,L,T)
        train_y_data=data_train.iloc[: , -T*features:]
        train_y_data=train_y_data[output_variable+" (t)"].to_frame()
        train_x_data=data_train.iloc[: , :L*features]
        test_y_data=data_test.iloc[: , -T*features:]
        test_y_data=test_y_data[output_variable+" (t)"].to_frame()
        test_x_data=data_test.iloc[: , :L*features]
        ##SCALING##
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
        X_train = train_x.reshape((train_x.shape[0], L, features))
        X_test = test_x.reshape((test_x.shape[0], L, features))
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
        lstm_best_model=ModelCheckpoint(f"{multivariate_grid_search}/lstm_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        lstm_fit=lstm_model.fit(X_train, train_y, epochs=lstm_epochs, batch_size=lstm_batch, validation_split=0.2, verbose=2, callbacks=[lstm_best_model])
        ##LOAD BEST MODEL##
        lstm_best_model=load_model(f"{multivariate_grid_search}/lstm_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        lstm_test_predict=lstm_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(test_y)
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
        scores = lstm_multivariate(config, df)
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
    with open(f"{multivariate_grid_search}/LSTM - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Multivariate Long Short-Term Memory (LSTM) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel Configuration: [Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##GATED RECURRENT UNIT##
def grid_search_gru_multivariate(df, parent_folder, series_name, output_variable, lags, nodes1, nodes2, optimizer, epochs, batch):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def gru_multivariate(config, df):
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
        ##DATA PREPARATION##
        features=df.shape[1]
        T=1
        ##TRANSFORM DATA TO TENSOR##
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
        data_train=data_transform(df_train, L,T)
        data_test=data_transform(df_test,L,T)
        train_y_data=data_train.iloc[: , -T*features:]
        train_y_data=train_y_data[output_variable+" (t)"].to_frame()
        train_x_data=data_train.iloc[: , :L*features]
        test_y_data=data_test.iloc[: , -T*features:]
        test_y_data=test_y_data[output_variable+" (t)"].to_frame()
        test_x_data=data_test.iloc[: , :L*features]
        ##SCALING##
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
        X_train = train_x.reshape((train_x.shape[0], L, features))
        X_test = test_x.reshape((test_x.shape[0], L, features))
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
        gru_best_model=ModelCheckpoint(f"{multivariate_grid_search}/gru_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        gru_fit=gru_model.fit(X_train, train_y, epochs=gru_epochs, batch_size=gru_batch, validation_split=0.2, verbose=2, callbacks=[gru_best_model])
        ##LOAD BEST MODEL##
        gru_best_model=load_model(f"{multivariate_grid_search}/gru_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        gru_test_predict=gru_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(test_y)
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
        scores = gru_multivariate(config, df)
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
    with open(f"{multivariate_grid_search}/GRU - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Multivariate Gated Recurrent Unit (GRU) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel Configuration: [Inputs, First Layer Nodes, Second Layer Nodes, Epochs, Batch Size, Optimizer]", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
##CONVOLUTIONAL NEURAL NETWORK##
def grid_search_cnn_multivariate(df, parent_folder, series_name, output_variable, lags, filters1, filters2, cnn_nodes, optimizer, epochs, batch):
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
    neural_network_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks'
    multivariate_folder = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate'
    multivariate_grid_search = pathlib.Path.home()/'Desktop/Time_Series_Prediction_Project' / parent_folder / series_name / 'Neural Networks/Multivariate/Grid Search'
    list_of_folders=[project_path,parent_folder,series_path,neural_network_folder,multivariate_folder,multivariate_grid_search]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    ##FIT MODEL FUNCTION##
    import pandas as pd
    from pandas import concat
    def cnn_multivariate(config, df):
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
        ##DATA PREPARATION##
        features=df.shape[1]
        T=1
        ##TRANSFORM DATA TO TENSOR##
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
        data_train=data_transform(df_train, L,T)
        data_test=data_transform(df_test,L,T)
        train_y_data=data_train.iloc[: , -T*features:]
        train_y_data=train_y_data[output_variable+" (t)"].to_frame()
        train_x_data=data_train.iloc[: , :L*features]
        test_y_data=data_test.iloc[: , -T*features:]
        test_y_data=test_y_data[output_variable+" (t)"].to_frame()
        test_x_data=data_test.iloc[: , :L*features]
        ##SCALING##
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y=scaler_y.fit_transform(train_y_data)
        train_x=scaler_x.fit_transform(train_x_data)
        test_y=scaler_y.fit_transform(test_y_data)
        test_x=scaler_x.fit_transform(test_x_data)
        ##RESHAPE INPUT DATA TO FIT KERAS REQUIREMENTS##
        X_train = train_x.reshape((train_x.shape[0], L, features))
        X_test = test_x.reshape((test_x.shape[0], L, features))
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
            cnn_model.add(Conv1D(filters=cnn_filters_1, kernel_size=2, activation='relu'))
            cnn_model.add(MaxPooling1D(pool_size=2))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(cnn_dense_nodes, activation='relu'))
            cnn_model.add(Dense(1))
            cnn_model.compile(loss='mse', optimizer=cnn_optimizer, metrics='mae')
        ##SAVE BEST MODEL##
        cnn_best_model=ModelCheckpoint(f"{multivariate_grid_search}/cnn_best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        ##FIT MODEL ON TRAINING DATA##
        cnn_fit=cnn_model.fit(X_train, train_y, epochs=cnn_epochs, batch_size=cnn_batch, validation_split=0.2, verbose=2, callbacks=[cnn_best_model])
        ##LOAD BEST MODEL##
        cnn_best_model=load_model(f"{multivariate_grid_search}/cnn_best_model.h5")
        ##FIT MODEL ON TEST DATA##
        cnn_test_predict=cnn_best_model.predict(X_test,verbose=2)
        ##INVERT SCALING AND RESHAPE VECTORS##
        y_test_inv = scaler_y.inverse_transform(test_y)
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
        scores = cnn_multivariate(config, df)
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
    with open(f"{multivariate_grid_search}/CNN - Grid Search Parameters.txt", "w") as f:
        print("Grid Search Algorithm on Multivariate Convolutional Neural Network (CNN) Model\nConfigurations Tried : "+str(len(cfg_list))+"\nExecution Time : "+str(execution_time)+"\nModel Configuration: [Inputs, First Layer Filters, Second Layer Filters, Third Layer Neurons, Epochs, Batch Size, Optimizer]", file=f)
        for cfg, error in scores:
            print("Model configuration:",cfg,"--> RMSE:",error,  file=f)
    print("Grid Search is complete")
#####################################################################################################################################
#####################################################################################################################################