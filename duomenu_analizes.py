from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
import pandas as pd
import read_data 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#PACF optimalaus zingsnio nustatymo anallize
def PCAF_analize(data_dir, tar_name):
    #tar_name = "AB50A30LRC01_PV"
    #data_dir = 'D:/projektas/test_1min_txt'
    df = read_data.read_data(data_dir)
    # PACF testas
    plt.figure(figsize=(10,5))
    plot_pacf(df[tar_name], lags=50, method="ols" )   # arba "ols" , "ld" , 'ywm'
    plt.title("PACF Testas - Lagų parinkimas")
    plt.show()

#Duomenu korealiacijos matrica
def data_matrica(data_dir):
    #direktorija = 'D:/projektas/test_1min_txt'
    df = read_data.read_data(data_dir)

    #duomenu korealiacijos matrica
    correlation_matrix = df.corr(method='spearman')
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Koreliacijos matrica')
    plt.show()

    # target_column = "AB50A30LRC01_PV"
    # df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    # df.dropna(inplace=True)