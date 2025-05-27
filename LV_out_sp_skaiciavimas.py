import pandas as pd
import generatorius_2
import read_data
from trigeris_pagal_laika import timed_trigger
import torch
import LSTM_modelio_arch
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings

warnings.filterwarnings("ignore", message="X neturi teisingu reiksmiu")             # Isjungiame perspėjimus (warnings), 
                                                                                    # jei yra blogu reiksmiu turi neteisingu reikšmių
duomenu_direktorija = 'D:/projektas/1min_txt_gen_test'                              # Nurodo katalogą, kuriame yra .txt failai.
prognozes_reiksme = 'AB50A30LRC01_PV'                                               # Prognozuojamu reiksmiu stulpelis (Y)
eilutes_ilgis = 35                                                                  # Sekos ilgis
LRC_SP = 'AB50A30LRC01_SP'                                                          # Uzduota reaktoriaus lygio reiksme-is duomenu
LTC_OUT = 'AB50A30LRC01_OUT'
#originalus df unscaleriui
df_originall=read_data.read_data(duomenu_direktorija)                               # Iskvieciama funkcija "read_data" duomenu nuskaitymui


Eilutes_cl = generatorius_2.SequenceStreamer(df=df_originall,                       # Sukuriame eiluciu generavima ir modifikavimo klases objekta
                                             target_column = prognozes_reiksme , 
                                             seq_length = eilutes_ilgis )


#FC eilutes generavimui
def generuok_eil():                                                                 # Funkcija, kuri grazina kita seka, jei dar yra duomenu
    if Eilutes_cl.has_next():
        x = Eilutes_cl.next_sequence()
        return x
    else:
        print("✅ Sekų pabaiga nieko daugiau negeneruojama.")

#pasiimam LRS_SP reiksme is sugeneruotos eilutes
def get_LRC_SP(x):                                                                  # Funkcija, kuri grąžina vertes is paskutinės sekos eilutes
    val_LRC_SP = Eilutes_cl.get_feature_value_from_last_row(x, LRC_SP)              #Paimama lygio uzduotis - SP
    val_LRC_OUT = Eilutes_cl.get_feature_value_from_last_row(x, LTC_OUT)            # Paimama voztuvo OUT verte
    return val_LRC_SP, val_LRC_OUT

# # Modelio prognozė busimai lygio PV uz 35min - arba koks suformuotas mokymo zingsnis 1zingsnis = 1min
def predict_LR_PV_val(x):                                                           # Prognozuoja LRC_PV reikšme (35 zingsniu)
    if x is not None:
        device_pr = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Nusistatome ar galime naudoti GPU, ar CPU
        #pasiimam issaugota geriausiai apmokyta modeli - LSTM
        model_path = "D:/projektas/LSTM_PyTorch/best_model_2.pth"                   # Nurodome kelia i issaugota modeli
        scaler = joblib.load("scaler_X.pkl")                                        # Normalizuojame seka naudodami ta pati scaleri,
                                                                                    # kuris buvo mokymo metu
        x_scaled = scaler.transform(x)  
        pred = LSTM_modelio_arch.predict_is_duom_eilutes(x_scaled, model_path, device = device_pr)  # Atliekame prognoze
        return pred
    
 # Ciklinė funkcija, kuri keicia vožtuvo OUT reikšme tol, kol prognozuota PV priarteja prie SP  
def get_pred_LV_OUT():
   eilute = generuok_eil()
   LRC_SP_act, LRC_OUT_act = get_LRC_SP(eilute)                                     # Paimamos SP ir OUT reikšmes iš paskutinės sekos eilutės
   LRC_PV_pred = predict_LR_PV_val(eilute)                                          # Kartojame, kol prognozuota reiksme pakankamai priarteja prie SP (±0.7)
   while True:  
    if LRC_PV_pred > LRC_SP_act+0.7:                                                    # Jei prognoze didesne nei SP + 0.7
        print(f"🔁 Didinam LRC OUT reiksme, Prediktinta reiksme 🔁 {LRC_PV_pred}")     # Padidiname vožtuvo OUT reikšmę visoje sekoje +1
        eilute = Eilutes_cl.modify_sequence(eilute, feature_name=LTC_OUT, new_value = LRC_OUT_act+1)
        LRC_PV_pred = predict_LR_PV_val(eilute)
    elif LRC_PV_pred < LRC_SP_act-0.7:                                                  # Jei prognoze mazesne nei SP - 0.7
        print(f" 🔁 Mazinam LRC OUT reiksme, Prediktinta reiksme 🔁 {LRC_PV_pred}")    # Sumaziname vožtuvo OUT reikšmę visoje sekoje -1
        eilute = Eilutes_cl.modify_sequence(eilute, feature_name=LTC_OUT, new_value = LRC_OUT_act-1)
        LRC_PV_pred = predict_LR_PV_val(eilute)
    else: 
        print("Is pirmo karto")                                                     # Jei prognoze pakankamai artima SP – isvedame rezultata
        LRC_SP_act, LRC_OUT_rek = get_LRC_SP(eilute) 
        print(f" 🎯 Apskaiciuota LRC voztuvo SP reiksme 🎯 = {LRC_OUT_rek}")
        break
    
   return None
#Naudojam kai norim generuoti stepBystep -- Privaloma uzkomentuoti paskutine eilute
# print("Bandom nuspeti vostuvo sp")
# komanda = input("🚀 Ar pradeti 🚀 y/n  :")
# if komanda == "y":
#     get_pred_LV_OUT()
#     while True:
#         kartoti = input("🔁 Ar kartoti 🔁 y/n  :")
#         if kartoti=="y":
#             get_pred_LV_OUT()
#         else:
#             print("🏁 Baigiam darba 🏁")
#             break


#Naudojama automatiniam seku generavimui su uzduotu uzlaikymu --Ciklo sustabdymui - ctrl+C

timed_trigger(interval_seconds=15, action = get_pred_LV_OUT)
