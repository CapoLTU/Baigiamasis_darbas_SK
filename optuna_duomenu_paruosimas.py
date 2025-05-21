import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from read_data import read_data 
import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm



#iskviecia funkcija read_data
# duom_dir is main perduoda i read_data() direktorija su failais 
# tikslas - is main perduodamas i feature_pv
def train_val_test_opt(duom_dir, tikslas):
    # Saugo originalias reikšmes
    df_original = read_data(duom_dir)  # Gauname duomenis
    df = df_original.copy()

    # Patikriname, ar yra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Naudojamas įrenginys: {device}")

    # Normalizavimas naudojant MinMaxScaler
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    sequence_length = 35  # Prognozuojame 70 laiko žingsnių į priekį

    # Funkcija duomenų sekų generavimui
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            if i + seq_length < len(X):  # Apsauga nuo indeksavimo klaidos
                X_seq.append(X.iloc[i:i+seq_length].values)
                y_seq.append(y.iloc[i+seq_length].values)
        return np.array(X_seq), np.array(y_seq)

    # Prognozuojame kintamąjį "AB50A30LRC01_PV"
    feature_pv = tikslas
    X_pv = df.drop(columns=[feature_pv])
    y_pv = df[[feature_pv]]

    X_pv_seq, y_pv_seq = create_sequences(X_pv, y_pv, sequence_length)

    # Daliname duomenis į treniruotę ir testą (90% train, 10% test)
    X_pv_train, X_pv_test, y_pv_train, y_pv_test = train_test_split(
        X_pv_seq, y_pv_seq, test_size=0.1, random_state=42
    )

    # Daliname treniruotę į train (85%) ir validaciją (15%)
    X_pv_train, X_pv_val, y_pv_train, y_pv_val = train_test_split(
        X_pv_train, y_pv_train, test_size=0.185, random_state=42
    )

    # Patikriname duomenų dydžius
    print(f"Train set: {X_pv_train.shape}, Validation set: {X_pv_val.shape}, Test set: {X_pv_test.shape}")

    # Konvertuojame į PyTorch tensorus
    X_pv_train = torch.tensor(X_pv_train, dtype=torch.float32).to(device)
    y_pv_train = torch.tensor(y_pv_train, dtype=torch.float32).to(device)
    X_pv_val = torch.tensor(X_pv_val, dtype=torch.float32).to(device)
    y_pv_val = torch.tensor(y_pv_val, dtype=torch.float32).to(device)
    X_pv_test = torch.tensor(X_pv_test, dtype=torch.float32).to(device)
    y_pv_test = torch.tensor(y_pv_test, dtype=torch.float32).to(device)

    # train_data = TensorDataset(X_pv_train, y_pv_train)
    # val_data = TensorDataset(X_pv_val, y_pv_val)
    # test_data = TensorDataset(X_pv_test, y_pv_test)

    # train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    print("Duomenys paruosti perdavimui i modeli")
    #return X_pv_train, X_pv_val, X_pv_test, y_pv_train, y_pv_val, y_pv_test, X_pv_seq, y_pv_seq, df_original 
    #return train_loader, val_loader, test_loader, X_pv_seq, y_pv_seq, df_original 
    return X_pv_train, y_pv_train, X_pv_val, y_pv_val 

def unscaler(normalized, original):
    return normalized * (original.max() - original.min()) + original.min()