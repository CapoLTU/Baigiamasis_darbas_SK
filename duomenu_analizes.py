from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
import pandas as pd
import read_data 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def PCAF_analize(data_dir, tar_name):                                  #PACF optimalaus zingsnio nustatymo anallize
    df = read_data.read_data(data_dir)                                 # Iskvieciame funkcija "read_data", duomenu nuskaitimui ir apjungimui
    # PACF testas
    plt.figure(figsize=(10,5))                                         # Sukuriamas grafikas su nurodytu dydžiu
    plot_pacf(df[tar_name], lags=50, method="ols" )                    # Pasirenkame analizes metoda arba "ols" , "ld" , 'ywm'. Dalinė autokoreliacija - 50 lagų
    plt.title("PACF Testas - Lagų parinkimas")                         # Grafiko pavadinimas 
    plt.show()                                                         # Rezultato atvaizdavimas

def data_matrica(data_dir):                                            #Duomenu korealiacijos matrica
    df = read_data.read_data(data_dir)                                 # Iskvieciame funkcija "read_data", duomenu nuskaitimui ir apjungimui

    #duomenu korealiacijos matrica
    correlation_matrix = df.corr(method='spearman')                    # Skaičiuojama koreliacijos matrica, naudojant "Spearman" koreliacijos koeficienta
    plt.figure(figsize=(12, 8))                                        # Sukuriama matrica su nurodytu dydžiu
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',       # Nustatome matricos atvaizdavimo parametrus
                 fmt='.2f', linewidths=0.5)
    plt.title('Koreliacijos matrica')                                  # Matricos pavadinimas
    plt.show()                                                         # Rezultato atvaizdavimas