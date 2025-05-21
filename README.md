# Baigiamasis_darbas_SK
Baigiamasis darbas lygio palaikymas
Beveik viska galima paleisti is main.py, tereikia nuimti uzkomentinimus nuo norimu funkciju ir likusias uzkomentinti

#Voztuvo rekomenduojamo OUT skaiciavimas pasileidzia is LV_out_sp_skaiciavimas.py
galima paleisti kad automatiskai atliktu skaiciavima kas tam tikra laiko intervala, arba uzkomentinti automatini 
skaiciavima ir nuimti komentus nuo stepBystep 
Failu direktorijas ir geriausio modeli nusirodyti pagal save (kur issisaugota, puikiai matosi kode kur kas)

#Optuna testavimas pasileidzia is - Optuna_tik_analizei.py
konfiguruojasi:
# parametrai analizei
  duomenu_direktorija = 'D:/projektas/1min_txt_optuna'
  sequence_length = 35

# Optimizavimas su Optuna
n_trials = 15
study = optuna.create_study(
    study_name="lstm_pytorch_optimization",
    direction="minimize",
    storage="sqlite:///optuna.db",
    load_if_exists=True
)
