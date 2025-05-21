import pandas as pd
import generatorius_2
import read_data
from trigeris_pagal_laika import timed_trigger
import torch
import LSTM_modelio_arch
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings

warnings.filterwarnings("ignore", message="X neturi teisingu reiksmiu")

duomenu_direktorija = 'D:/projektas/1min_txt_gen_test'
prognozes_reiksme = 'AB50A30LRC01_PV'
eilutes_ilgis = 35
LRC_SP = 'AB50A30LRC01_SP'
LTC_OUT = 'AB50A30LRC01_OUT'
#originalus df unscaleriui
df_originall=read_data.read_data(duomenu_direktorija)

# susikuriam eiluciu generavima ir modifikavimo klases objekta
Eilutes_cl = generatorius_2.SequenceStreamer(df=df_originall,
                                             target_column = prognozes_reiksme , 
                                             seq_length = eilutes_ilgis )


#FC eilutes generavimui
def generuok_eil():
    if Eilutes_cl.has_next():
        x = Eilutes_cl.next_sequence()
        return x
    else:
        print("✅ Sekų pabaiga nieko daugiau negeneruojama.")

#pasiimam LRS_SP reiksme is sugeneruotos eilutes
def get_LRC_SP(x):
    #pasiimam lygio SP
    val_LRC_SP = Eilutes_cl.get_feature_value_from_last_row(x, LRC_SP) 
    #pasiimam voztuvo OUT
    val_LRC_OUT = Eilutes_cl.get_feature_value_from_last_row(x, LTC_OUT) 
    return val_LRC_SP, val_LRC_OUT

# # Modelio prognozė busimai lygio PV uz 35min - arba koks suformuotas mokymo zingsnis 1zingsnis = 1min
def predict_LR_PV_val(x):
   # x = generuok_eil()  # gaunam naują seką

    if x is not None:
        # nusistatom su kuo dirbsim, CUDA  CPU
        device_pr = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #pasiimam issaugota geriausiai apmokyta modeli - LSTM
        model_path = "D:/projektas/LSTM_PyTorch/best_model_2.pth"
        
        #susiskaliuojam paduodamos eilutes reiksmes
        scaler = joblib.load("scaler_X.pkl")  # naudojam ta pati scaleri
        x_scaled = scaler.transform(x)  
        pred = LSTM_modelio_arch.predict_is_duom_eilutes(x_scaled, model_path, device = device_pr)
        #unscalinam atvaizdavimui
        #real = LSTM_modelio_arch.unscaler(pred, df_originall[prognozes_reiksme])
        #print(f"🔮 Modelio prognozė: {pred}")
        #print(f"🔮 Modelio prognozė: {real}")
        return pred
    
 #skaiciuojam reikiama voztuvo LV   
def get_pred_LV_OUT():
   #pasiimam abi aktualias reiksmes
   eilute = generuok_eil()
   LRC_SP_act, LRC_OUT_act = get_LRC_SP(eilute) 
#pradedam prediktinima per cikla kol gausim reikiama reiksme - Jonas sake nelabai efektyvu
   LRC_PV_pred = predict_LR_PV_val(eilute)
   while True:  
    if LRC_PV_pred > LRC_SP_act+0.7:
        #pakeiciam voztuvo out reiksmes visoje eiluteje - buvusia reiksme padidinam +1
        print(f"🔁 Didinam LRC OUT reiksme, Prediktinta reiksme 🔁 {LRC_PV_pred}")
        eilute = Eilutes_cl.modify_sequence(eilute, feature_name=LTC_OUT, new_value = LRC_OUT_act+1)
        LRC_PV_pred = predict_LR_PV_val(eilute)
    elif LRC_PV_pred < LRC_SP_act-0.7:
         #pakeiciam voztuvo out reiksmes visoje eiluteje - buvusia reiksme padidinam +1
        print(f" 🔁 Mazinam LRC OUT reiksme, Prediktinta reiksme 🔁 {LRC_PV_pred}")
        eilute = Eilutes_cl.modify_sequence(eilute, feature_name=LTC_OUT, new_value = LRC_OUT_act-1)
        LRC_PV_pred = predict_LR_PV_val(eilute)
    else: 
        print("Is pirmo karto")
        LRC_SP_act, LRC_OUT_rek = get_LRC_SP(eilute) 
        print(f" 🎯 Apskaiciuota LRC voztuvo SP reiksme 🎯 = {LRC_OUT_rek}")
        break
    
   return None
#_____________________Naudojam kai norim generuoti stepBystep -- uzkomentinam Auto generacija______________________________
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

#_____________________Naudojam kai norim generuoti auto per uzduota laika --stabdymas ctrl+C--______________________________
timed_trigger(interval_seconds=15, action = get_pred_LV_OUT)
