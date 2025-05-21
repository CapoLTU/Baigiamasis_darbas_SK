from GRU_modelio_arch import test_model
import duomenu_paruosimas
import torch
import torch.nn as nn
import torch.optim as optim
from GRU_modelio_arch import GRUModel
import glob


#_____________________ateiciai____________________________
#susimesti projekto pavadinima ir kelia i kintamaji padavimui is main

def GRU_test(duom_dir, tikslas):

    # nusistatom su kuo dirbsim, CUDA  CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pasiimam duomenis, prezentuchai naudosim atskirus
    train_loader, val_loader, test_loader, X_pv_seq, y_pv_seq, df_original = duomenu_paruosimas.train_val_test(duom_dir, tikslas)

    # pasiimam issaugota modeli
    model_path_list = glob.glob("D:/projektas/LSTM_PyTorch/geriausias_GRU_modelis.pth") 
    if not model_path_list:
        raise FileNotFoundError("Modelio failas nerastas!")

    model_path = model_path_list[0]  # Pasiimam failo kelią (string)

    # Atliekam testavimą
    preds, targets = test_model(
        model_class=GRUModel,
        model_path=model_path,
        test_loader=test_loader,
        device=device,
        criterion=nn.MSELoss(),
        original_df_column=df_original[tikslas],  # unscalingui
        save_csv_path="rezultatai/GRU_testo_rezultatai.csv"
    )

    return preds, targets  # gražina prognozes jei reiks toliau analizuoti
