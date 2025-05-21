import pandas as pd
import glob
import os

direktorija = 'D:/projektas/test_1min_txt'
prognozes_reiksme = 'AB50A30LRC01_PV'
def read_data(direktorija): #perduodam direktorija kuriame yra txt failai

# Randame visus .txt failus kataloge ir pasidarom failu lista iteracijoms
    txt_failai = glob.glob(os.path.join(direktorija, "*.txt"))
    df_listas = []

# Nuskaitymas į pandas DataFrame ir sujungimas į vieną
    for failas in txt_failai:
        try:
            df = pd.read_csv(failas, sep=',', low_memory=False, dtype=str)
            df.columns = df.columns.str.strip()

            if 'Date stamp' in df.columns:
                df['Date stamp'] = pd.to_datetime(df['Date stamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
            else:
                print(f"Įspėjimas: Faile {failas} nėra 'Date stamp' stulpelio!")

            df_listas.append(df)
            print(f"Failas {failas} sėkmingai nuskaitytas ir suformatuotas!")
        except Exception as e:
            print(f"Klaida skaitant failą {failas}: {e}")

# Sujungiame visus duomenis į vieną DataFrame
    if df_listas:
        df_apjungtas = pd.concat(df_listas, ignore_index=True)

    # Automatinė stulpelių konversija į skaitines reikšmes po sujungimo
        for stulp in df_apjungtas.columns:
            if stulp != 'Date stamp':  # Neperkonvertuojame datos stulpelio
                df_apjungtas[stulp] = pd.to_numeric(df_apjungtas[stulp], errors='coerce')
        df_apjungtas.set_index('Date stamp', inplace=True)
        return df_apjungtas
        print("Visi failai sėkmingai sujungti į vieną DF su suvienodintu 'Date stamp' formatu!")
        print("Visi skaitiniai stulpeliai konvertuoti iš 'str' į tinkamus tipus.")
    else:
        print("Nerasta tinkamų failų sujungimui.")
        return None
