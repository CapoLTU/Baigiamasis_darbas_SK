
# Reaktoriaus Lygio Prognozavimo Projektas (LSTM/GRU)

Šis projektas skirtas prognozuoti cheminio reaktoriaus lygį pasitelkiant laiko eilučių analizę bei pažangias rekursines neuroninių tinklų architektūras (LSTM, GRU). Naudojami duomenys su stipriomis laikinėmis priklausomybėmis, todėl klasikiniai neuroniniai tinklai netinka – vietoje to taikomi specializuoti sekų modeliai.

---

## 📁 Projekto struktūra

###  1. Duomenų analizė ir apdorojimas
| Failas                      | Funkcija |
|----------------------------|----------|
| `read_data.py`             | Sujungia .txt failus į vieną `DataFrame`. |
| `duomenu_analizes.py`      | Koreliacija, PACF analizė, kintamųjų atranka. |
| `normalizavimo_poreikis.py`| Histogramų analizė, pagrindžianti normalizavimą. |  

###  2. Duomenų paruošimas mokymui
| Failas                          | Funkcija |
|--------------------------------|----------|
| `duomenu_paruosimas.py`        | transformuoja laiko eilučių duomenis į modelio mokymui tinkamą formatą, sukuriant normalizuotas slenkančių langų sekas ir padalijant jas į mokymo, validacijos ir testavimo rinkinius. |
| `optuna_duomenu_paruosimas.py` | Paruošia duomenis modelio hiperparametrų optimizacijai su Optuna |
| `generatorius_2.py`            | Sekų generavimas pavieniam prognozavimui. |

###  3. Modelių architektūra
| Failas                  | Aprašymas |
|------------------------|-----------|
| `LSTM_modelio_arch.py` | Trijų sluoksnių LSTM su treniravimo, testavimo funkcijomis. |
| `GRU_modelio_arch.py`  | GRU versija – palyginimui.                                  |

###  4. Modelio apmokymas
| Failas                     | Funkcija |
|---------------------------|----------|
| `LSTM_modelio_mokymas.py` | LSTM treniravimas su vizualizacija.   |
| `GRU_modelio_mokyma.py`   | GRU treniravimas.                     |
| `Optuna_tik_analize.py`   | Optuna optimizacija hiperparametrams. |

###  5. Testavimas ir analizė
| Failas                     | Funkcija |
|---------------------------|----------|
| `LSTM_testavimas.py`      | LSTM testavimas, metrikos, vaizdavimas.                           |
| `GRU_testavimas.py`       | GRU testavimas, metrikų skaičiavimas ir rezultatų vizualizavimas. |
| `modelio_testavimo_analize.py` | Metrikų analizė: MAE, RMSE, R², klaidos histogramos.         |

###  6. OUT/Setpoint skaičiavimas
| Failas                        | Funkcija |
|------------------------------|----------|
| `LV_out_sp_skaiciavimas.py`  | OUT valdymo užduoties (SP) skaičiavimas pagal LSTM prognozę, leidžiant pasirinkti tarp automatinio režimo (su trigeriu) arba vienkartinio vykdymo. |

###  7. Paleidimui pagal laiką
| Failas                     | Funkcija |
|---------------------------|----------|
| `trigeris_pagal_laika.py` | Paleidžia modelio prognozę periodiškai kas N sekundžių (`threading.Timer`). |

###  8. Paleidimo failas
| Failas      | Funkcija |
|-------------|----------|
| `main.py`   | Centralizuotas pasirinkimas ką vykdyti: treniruoti, testuoti, reguliuoti, naudoti GRU ar LSTM. |

---

## 🔂 Loginės priklausomybės schema

```
read_data.py
     |
     v
duomenu_paruosimas.py     optuna_duomenu_paruosimas.py
     |                              |
     |                              v
     |                     Optuna_tik_analize.py
     v
LSTM_modelio_mokymas.py ─────> LSTM_modelio_arch.py
GRU_modelio_mokyma.py  ─────> GRU_modelio_arch.py
     |                              |
     v                              v
LSTM_testavimas.py            GRU_testavimas.py
     |                              |
     +────> modelio_testavimo_analize.py

LV_out_sp_skaiciavimas.py ─────> generatorius_2.py
                                └────> LSTM_modelio_arch.py

trigeris_pagal_laika.py → periodinis `predict()` paleidimas
main.py → jungia visus komponentus
```

---

## 📦 Reikalavimai (minimalūs)

```
pandas
numpy
matplotlib
scikit-learn
torch
optuna
tqdm
seaborn
```

> Diegimas: `pip install -r requirements.txt`

---

##  Paleidimas

```bash
python main.py
```

Pasirinkus `main.py`, vartotojas gali pasirinkti modelį (LSTM/GRU), treniruoti, testuoti arba naudoti OUT/SP skaičiavimą.

---

## 🔖 Licencija

Projektas mokslinis, sukurtas baigiamajam darbui. Naudoti galima edukaciniais tikslais.

