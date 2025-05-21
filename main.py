
import LSTM_modelio_mokymas
import LSTM_testavimas
import normalizavimo_poreikis
import duomenu_analizes
import modelio_testavimo_analize
import duomenu_paruosimas
import GRU_modelio_mokyma
import GRU_testavimas

duomenu_direktorija = 'D:/projektas/test_1min_txt'
prognozes_reiksme = 'AB50A30LRC01_PV'
rolling_window = 35 # duomenų sekos dydis, toks pats kaip ir duomenu paruosime nori keisti -- pasikeisk ir duomenu paruosimo FC parametra--

#______________________Paruostu duomenu vertinimas ir kiti parametrai__________________________________________
    # normalizavimo_poreikis.norm_test(duomenu_direktorija)
    # duomenu_analizes.data_matrica(duomenu_direktorija )
    # duomenu_analizes.PCAF_analize(duomenu_direktorija , prognozes_reiksme)



#______________LSTM modelis___________________________________________________________
##________________________LSTM modelio mokymas_________________________________________
    #LSTM_modelio_mokymas.LSTM_train(duomenu_direktorija, prognozes_reiksme)
##________________________LSTM modelio testavimas + testavimo analize__!!!jei nori abu iskart nuimti abu komentus!!!____________
    #pred, targ = LSTM_testavimas.LSTM_test(duomenu_direktorija, prognozes_reiksme)
    #modelio_testavimo_analize.analyze_predictions(pred, targ, rolling_window)
##________________________LSTM modelio tik testavimas____________________________________________________
    #LSTM_testavimas.LSTM_test(duomenu_direktorija, prognozes_reiksme)


#_________________________________________GRU_modelis_____________________________
##____________________________GRU modelio mokymas__________________________________
    #GRU_modelio_mokyma.GRU_train(duomenu_direktorija, prognozes_reiksme)
##___________________________GRU modelio testavimas + testavimo analize___!!!jei nori abu iskart nuimti abu komentus!!!____________
# pred_GRU, targ_GRU = GRU_testavimas.GRU_test(duomenu_direktorija, prognozes_reiksme)
# modelio_testavimo_analize.analyze_predictions(pred_GRU, targ_GRU, rolling_window)



#________________________duomenu paruosimo FC testavimas________________________________
    #duomenu_paruosimas.train_val_test(duomenu_direktorija , prognozes_reiksme)