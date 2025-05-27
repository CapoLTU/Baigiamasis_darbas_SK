import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from read_data import read_data 
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import joblib

def train_val_test(duom_dir, tikslas):                                      #Funkcija skirta duomenu normalizavimui, padalinimui į sekas ir train/val/test
    df_original = read_data(duom_dir)                                       # Iskvieciame funkcija "read_data", duomenu nuskaitimui ir apjungimui
    df = df_original.copy()                                                 # Nusikopijuojame Dataframe, tikslas nepakeisti originalu Dataframe

    # Patikriname, ar yra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Nusistatome ar galime naudoti GPU, ar CPU
    print(f"Naudojamas įrenginys: {device}")                                # Isvedame irenginio, su kuriuo dirbsime pavadinima

    # Pasiruošiam X ir y
    feature_pv = tikslas                                                    # Priskiriamas tikslinio stulpelio pavadinimas - Target
    X = df.drop(columns=[feature_pv])                                       # Ivesties duomenys – visi stulpeliai isskyrus Target
    y = df[[feature_pv]]                                                    # Isvestis - Target

    # Duomenu normalizacija
    scaler = MinMaxScaler()                                                 # Pasirenkamas MinMax skaleris
    X_scaled = scaler.fit_transform(X)                                      # Normalizuojame tik x reiksmes
    joblib.dump(scaler, "scaler_X.pkl")                                     # Issisaugojami scalerio parametrai vėlesniam naudojimui
    df[X.columns] = X_scaled                                                # Dataframeás atnaujinamas su normalizuotomis X reikšmėmis

    sequence_length = 35                                                    # Nustatomas duomenu sekos ilgis

    # Funkcija sekų generavimui
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            if i + seq_length < len(X):                                     # Apsauga nuo indeksavimo klaidų
                X_seq.append(X.iloc[i:i+seq_length].values)                 # Sekos ilgis - 35 eilutės
                y_seq.append(y.iloc[i+seq_length].values)                   # Tik viena reikšmė po kiekvieno lango pabaigos, tikslas po 35 zingsnio
        return np.array(X_seq), np.array(y_seq)

    X_pv = df.drop(columns=[feature_pv])                                    # Pasiimami ivesties  duomenys is normalizuoto Dataframe be Target
    y_pv = df[[feature_pv]]                                                 # Pasiimami Target duomenys

    X_pv_seq, y_pv_seq = create_sequences(X_pv, y_pv, sequence_length)      # Suformuojame sekas su nustatytu lango ilgiu

    # Dalijame į train / val / test
    X_pv_train, X_pv_test, y_pv_train, y_pv_test = train_test_split(        # Padaliname i train + val ir test (90% ir 10%).
        X_pv_seq, y_pv_seq, test_size=0.1, random_state=42
    )
    X_pv_train, X_pv_val, y_pv_train, y_pv_val = train_test_split(          # Padaliname train + val i train ir val (75% train, 15% val)
        X_pv_train, y_pv_train, test_size=0.185, random_state=42
    )
    print(f"Train: {X_pv_train.shape}, Val: {X_pv_val.shape}, Test: {X_pv_test.shape}") 

    # Konvertavimas į PyTorch tensorius
    X_pv_train = torch.tensor(X_pv_train, dtype=torch.float32).to(device)
    y_pv_train = torch.tensor(y_pv_train, dtype=torch.float32).to(device)
    X_pv_val = torch.tensor(X_pv_val, dtype=torch.float32).to(device)
    y_pv_val = torch.tensor(y_pv_val, dtype=torch.float32).to(device)
    X_pv_test = torch.tensor(X_pv_test, dtype=torch.float32).to(device)
    y_pv_test = torch.tensor(y_pv_test, dtype=torch.float32).to(device)

    # Sukuriame DataLoader. shuffle=True tik mokymui efektyvesniai treniruotei
    train_loader = DataLoader(TensorDataset(X_pv_train, y_pv_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_pv_val, y_pv_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_pv_test, y_pv_test), batch_size=32, shuffle=False)
    print("✅ Duomenys paruošti modeliui")

    return train_loader, val_loader, test_loader, X_pv_seq, y_pv_seq, df_original   # Funkcijos grazinamos reiksmes

def unscaler(normalized, original):
     return normalized * (original.max() - original.min()) + original.min()