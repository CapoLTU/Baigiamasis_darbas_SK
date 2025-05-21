import GRU_modelio_arch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import duomenu_paruosimas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def GRU_train(duom_dir, tikslas):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, X_pv_seq, y_pv_seq, df_original = duomenu_paruosimas.train_val_test(duom_dir, tikslas)

    model = GRU_modelio_arch.GRUModel(input_size=X_pv_seq.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00020892936842993308)

    train_losses, val_losses = GRU_modelio_arch.train_model(
        model, train_loader, val_loader, optimizer, criterion, device, epochs=50
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Mokymo nuostolis')
    plt.plot(val_losses, label='Validacijos nuostolis')
    plt.xlabel('Epoch')
    plt.ylabel('Nuostolis (MSE)')
    plt.legend()
    plt.title('Mokymo ir validacijos nuostolių palyginimas')
    plt.grid(True)
    plt.tight_layout()
    plt.show()