import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import read_data
import numpy as np

def norm_test(direktorija):
    df = read_data.read_data(direktorija)                                               # Iskvieciame funkcija "read_data", duomenu nuskaitimui ir apjungimui
   
    for column in df.select_dtypes(include=[np.number]).columns:                        # Histogramų analizės generavimas kiekvienam stulpeliui atskirai su KDE kreive
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], bins=30, kde=True, edgecolor='black')
        plt.title(f'Histogramų analizė: {column}')
        plt.xlabel(column)
        plt.ylabel('Dažnis')
        plt.show()