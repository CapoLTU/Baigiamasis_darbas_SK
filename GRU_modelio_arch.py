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

#____________________________Modelis: GRU________________________________
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 96, 64], output_size=1, dropout_rate=0.3):
        super(GRUModel, self).__init__()

        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])

        self.gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_sizes[1])

        self.gru3 = nn.GRU(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_sizes[2], 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        gru_out1, _ = self.gru1(x)
        gru_out1 = self.batch_norm1(gru_out1.transpose(1, 2)).transpose(1, 2)

        gru_out2, _ = self.gru2(gru_out1)
        gru_out2 = self.batch_norm2(gru_out2.transpose(1, 2)).transpose(1, 2)

        gru_out3, _ = self.gru3(gru_out2)
        gru_out3 = self.dropout(gru_out3[:, -1, :])

        dense_out = self.fc1(gru_out3)
        dense_out = self.relu(dense_out)
        output = self.fc2(dense_out)

        return output

#________________________Modelio treniravimo funkcija________________________________
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=6,
    best_model_path="geriausias_GRU_modelis.pth",
    best_val_loss=None,
    patience=10
):
    if best_val_loss is None:
        best_val_loss = float('inf')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.00001, patience=3)
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        scheduler.step(val_loss)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f" Geriausias modelis išsaugotas su nuostoliu: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Jokio pagerėjimo {epochs_no_improve}/{patience} epochas")

        if epochs_no_improve >= patience:
            print(f"Ankstyvas stabdymas po {epoch+1} epokų validacijos nuostolis nepagerėjo {patience} epokas.")
            break

    return train_losses, val_losses

#______________________________Testavimo funkcija_______________________
def test_model(model_class, model_path, test_loader, device, criterion,
               original_df_column=None, save_csv_path=None):

    example_batch = next(iter(test_loader))[0]
    input_size = example_batch.shape[2]
    model = model_class(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    preds_unscaled = preds
    targets_unscaled = targets

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

    if save_csv_path is not None:
        df = pd.DataFrame({
            "Tikra": targets_unscaled.flatten(),
            "Prognozuota": preds_unscaled.flatten()
        })
        df.to_csv(save_csv_path, index=False)
        print(f"Rezultatai išsaugoti į: {save_csv_path}")

    return preds_unscaled, targets_unscaled

#_______________Predikcija vienai sekai________________________________
def predict_is_duom_eilutes(x, model_path, device="cpu"):
    input_size = x.shape[1]
    hidden_sizes = [256, 96, 64]
    output_size = 1

    model = GRUModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(x_tensor)

    return prediction.cpu().numpy().flatten()[0]