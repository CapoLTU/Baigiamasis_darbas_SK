import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from optuna_duomenu_paruosimas import train_val_test_opt
from duomenu_paruosimas import train_val_test

# pasirenkam gpu cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Naudojamas įrenginys: {device}")

# parametrai analizei
duomenu_direktorija = 'D:/projektas/1min_txt_optuna'
prognozes_reiksme = 'AB50A30LRC01_PV'
sequence_length = 35

# Duomenys - taas pats kaip ir treniravimui tik biski modifikuotas
X_train, y_train, X_val, y_val = train_val_test_opt(duomenu_direktorija, prognozes_reiksme)
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

# LSTM modelis
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_sizes[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_sizes[2], 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.batch_norm1(lstm_out1.transpose(1, 2)).transpose(1, 2)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.batch_norm2(lstm_out2.transpose(1, 2)).transpose(1, 2)
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out3 = self.dropout(lstm_out3[:, -1, :])
        dense_out = self.fc1(lstm_out3)
        dense_out = self.relu(dense_out)
        output = self.fc2(dense_out)
        return output

# Optuna tikslinė funkcija
def objective(trial):
    hidden1 = trial.suggest_int("hidden1", 32, 256, step=32)
    hidden2 = trial.suggest_int("hidden2", 16, 128, step=16)
    hidden3 = trial.suggest_int("hidden3", 8, 64, step=8)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_sizes=[hidden1, hidden2, hidden3],
        dropout_rate=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epocha {epoch+1}/{num_epochs}", leave=False)
        for i, (xb, yb) in enumerate(train_loader_tqdm):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            percent_done = (i + 1) / len(train_loader)
            train_loader_tqdm.set_postfix({
                "Progresas": f"{percent_done * 100:.1f}%",
                "Loss": f"{loss.item():.4f}"
            })

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epocha {epoch+1}/{num_epochs} baigta. Vidutinis nuostolis: {avg_loss:.4f}")

    # Validacija su progresu
    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc="🔍 Validacija", leave=False)
    with torch.no_grad():
        for xb, yb in val_loader_tqdm:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = criterion(output, yb)
            val_loss += loss.item()
            val_loader_tqdm.set_postfix({
                "Loss": f"{loss.item():.4f}"
            })

    return val_loss / len(val_loader)

# Optimizavimas su Optuna
n_trials = 15
study = optuna.create_study(
    study_name="lstm_pytorch_optimization",
    direction="minimize",
    storage="sqlite:///optuna.db",
    load_if_exists=True
)

with tqdm(total=n_trials, desc="Optuna optimizacija") as pbar:
    def progress_callback(study, trial):
        pbar.update(1)
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

# Geriausi hiperparametrai
print(" Geriausi hiperparametrai:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")
print(f" Mažiausias validacijos nuostolis: {study.best_value:.6f}")
