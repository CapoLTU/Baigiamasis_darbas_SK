import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import duomenu_paruosimas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# LSTM Modelis
class LSTMModel(nn.Module):
    #def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.300000000000000004):  - 
    def __init__(self, input_size, hidden_sizes=[256, 96, 64], output_size=1, dropout_rate=0.300000000000000004):   # - modelis antras
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)           # 1 sluoksnis - is ivesties į pirma paslėpta sluoksni
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])                            # Batch normalizacija pirmo sluoksnio isvestims

        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)      # 2 sluoksnis - is pirmo paslėpto į antrą sluoksni
        self.batch_norm2 = nn.BatchNorm1d(hidden_sizes[1])                            # Antro sluoksnio normalizacija

        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)      # 3 sluoksnis - is antro paslėpto i trecia sluoksni
        self.dropout = nn.Dropout(dropout_rate)                                       # Persimokymo sumazinimui - atsitiktinai atmeta kai kurias reiksmes

        self.fc1 = nn.Linear(hidden_sizes[2], 32)                                     # 1 pilnai sujungtas sluoksnis - sumažina dimensiją iki 32
        self.relu = nn.ReLU()                                                         # ReLU aktyvacija 
        self.fc2 = nn.Linear(32, output_size)                                         # 2 pilnai sujungtas sluoksnis – išveda galutinę reikšmę

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.batch_norm1(lstm_out1.transpose(1, 2)).transpose(1, 2)       # Taikome normalizacija - transpose reikalingas, 
                                                                                      # nes BatchNorm1d tikisi (batch, features, time)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.batch_norm2(lstm_out2.transpose(1, 2)).transpose(1, 2)       # Taikome normalizacija - transpose reikalingas,
                                                                                      # nes BatchNorm1d tikisi (batch, features, time)
        lstm_out3, _ = self.lstm3(lstm_out2)                                          # Paskutinis - trecias LSTM sluoksnis
        lstm_out3 = self.dropout(lstm_out3[:, -1, :])                                 # Paimame paskutinio laiko zingsnio isvesti

        dense_out = self.fc1(lstm_out3)                                               # Praleidziame per pirmą dense sluoksni
        dense_out = self.relu(dense_out)                                              # ReLU aktyvacija
        output = self.fc2(dense_out)                                                  # Galutine prognoze

        return output

# Modelio treniravimo_Funkcija
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,                                                                         # Irenginys GPU / CPU - paimamas AUTO
    epochs = 6,                                                                       # Epochu skaicius
    best_model_path = "best_model_2.pth",                                             # Issaugomo geriausio modelio pavadinimas         
    best_val_loss = None,
    patience = 10                                                                     # Kiek epchų laukti be pagerėjimo
):
    if best_val_loss is None:                                   
        best_val_loss = float('inf')                                                # Nustatomas į begalybę, pirmo modelio issaugojimui
        
    # Automatinis learning rate mazinimas, jei val_loss nustoja gerėti
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.00001,                                                             # Nustatymas - kiek sumazinti learning rate
        patience = 3,                                                                 # Nustatymas - kiek epochų laukti be pagerejimo
    )
    train_losses = []
    val_losses = []
    epochs_no_improve = 0                                                           # Skaičiuojama kiek epochų be pagerėjimo

    # Mokymo ciklas
    for epoch in range(epochs):                                                     # Mokymo ciklas - eina per visas epochas               
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader,
                                      desc=f'Epoch {epoch+1}/{epochs}',
                                        leave=False):                               # Paimame duomenu partija (Batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()                                                   # Isvalomi gradientai
            y_pred = model(X_batch)                                                 # Atliekama prognoze
            loss = criterion(y_pred, y_batch)                                       # Apskaiciuojamas nuostolis
            loss.backward()                                                         # Atgalinis zingsnis
            optimizer.step()                                                        # Atnaujinami modelio svoriai
            train_loss += loss.detach().item()                                      # Issaugomas nuostolis, vėliasniam epochos vidurkio apskaiciavimui

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # VALIDACIJA
        model.eval()
        val_loss = 0
        with torch.no_grad():                                                       # Isjungiami gradientai - greitesne validacija
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()                       # Skaiciuojamas validacijos loss.

        #scheduler.step(val_loss)                                            
        val_loss /= len(val_loader)                                                 # Apskaiciuojamas vidutinis nuostolis per epocha
        val_losses.append(val_loss)
        scheduler.step(val_loss)                                                    # Tikriname ar yra pagerejimas. Jei nera 3 epochas - lr mazinamas

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Pasitikrinam ar pagerejo validacijos losas
        if val_loss < best_val_loss:                                                # Jei esamas val_loss geresnis uz buvusi - issaugome
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f" Geriausias modelis išsaugotas su nuostoliu: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Jokio pagerėjimo {epochs_no_improve}/{patience} epochas")       # Jei nera pagerejimo, skaiciuojame kiek epochu negereja

        # prieslaikinis stabdymas
        if epochs_no_improve >= patience:                                           # Jei negerėja ilgiau nei nustatyta - ankstyvas stabdymas
            print(f"Ankstyvas stabdymas po {epoch+1} epokų validacijos nuostolis nepagerėjo {patience} epokas.")
            break

    return train_losses, val_losses                                                 # Grazinami sarasai tolimesnei analizei.

def unscaler(normalized, original):
    return normalized * (original.max() - original.min()) + original.min()

#Testavimo Funkcija
def test_model(model_class, model_path, test_loader, device, criterion,
               original_df_column=None, save_csv_path=None):
  
    # 1. Sukuriam naują modelį ir įkeliame svorius is apmokyto ir issaugoto
    example_batch = next(iter(test_loader))[0]                                      # pasiimam pirma batcha iejimo duomenu dydziui
    input_size = example_batch.shape[2]
    model = model_class(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path))                                   # susikeliam apmokyto modelio svorius
    model.eval()

    all_preds = []                                                                  # Masyvas prognozuotoms reiksmems
    all_targets = []                                                                # Masyvas tikroms reiksmems

    with torch.no_grad():                                                           # Neskaiciuojame gradientu - mazesnis resursu poreikis
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu())                                          # Reikšmės perkeliamos i CPU pries konvertuojant i numpy
            all_targets.append(y_batch.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()                                     # Sujungiamos visos prognozes i numpy masyva
    targets = torch.cat(all_targets, dim=0).numpy()                                 # Sujungiamos visos tikros reiksmes i numpy masyva

    # 2. susiskaiciuojam metrikas ir atvaizduojam
    mse = mean_squared_error(targets, preds)                                        # Skaiciuojamas kvadratinis nuostolis,
    mae = mean_absolute_error(targets, preds)                                       # Skaiciuojamas vidutinis absoliutus skirtumas
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    
    preds_unscaled = preds
    targets_unscaled = targets

    # 4. Grafikas vizualiam modelio tikslumo ivertinimui
    plt.figure(figsize=(10, 5))
    plt.plot(targets_unscaled, label='Tikros reikšmės')
    plt.plot(preds_unscaled, label='Prognozės', alpha=0.7)
    plt.title("Tikros vs Prognozuotos reikšmės")
    plt.xlabel("Laiko žingsnis")
    plt.ylabel("Reikšmė")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. Eksportuojam i CSV tolimesniam naudojimui
    if save_csv_path is not None:
        df = pd.DataFrame({
            "Tikra": targets_unscaled.flatten(),
            "Prognozuota": preds_unscaled.flatten()
        })
        df.to_csv(save_csv_path, index=False)
        print(f"Rezultatai išsaugoti į: {save_csv_path}")

    return preds_unscaled, targets_unscaled

# LV_OUT_prognozes funkcija
def predict_is_duom_eilutes(x, model_path, device="cpu"):                               # pasiduodam tik viena seka ir grazinames prognoze
    # Modelio parametrai – tie patys kaip buvo treniravime
    input_size = x.shape[1]
    hidden_sizes = [256, 96, 64]
    output_size = 1

    model = LSTMModel(input_size=input_size, hidden_sizes=hidden_sizes,
                       output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))                  # Ikeliami išsaugoti svoriais is failo.
    model.to(device)
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)             # pridedame papildomą dimensiją,  
                                                                                        # „batch“ dimensijos simuliavimui (1, seq_len, input_size)
    with torch.no_grad():
        prediction = model(x_tensor)

    return prediction.cpu().numpy().flatten()[0]