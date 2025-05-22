
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
| `GRU_modelio_mokyma.py`   | GRU treniravimas su vizualizacija.                    |
| `Optuna_tik_analize.py`   | Optuna optimizacija hiperparametrams su vizualizacija. (!!!Savarankiškas skriptas, nesileidžia iš `main.py`!!!) |

###  5. Testavimas ir analizė
| Failas                     | Funkcija |
|---------------------------|----------|
| `LSTM_testavimas.py`      | LSTM testavimas, metrikų skaičiavimas ir rezultatų vizualizavimas.|
| `GRU_testavimas.py`       | GRU testavimas, metrikų skaičiavimas ir rezultatų vizualizavimas. |
| `modelio_testavimo_analize.py` | Metrikų analizė: MAE, RMSE, R², klaidos histogramos.         |

###  6. OUT/Setpoint skaičiavimas
| Failas                        | Funkcija |
|------------------------------|----------|
| `LV_out_sp_skaiciavimas.py`  | OUT valdymo užduoties (SP) skaičiavimas pagal LSTM prognozę, leidžiant pasirinkti tarp automatinio režimo (su trigeriu) arba vienkartinio vykdymo. (!!!Savarankiškas skriptas, nesileidžia iš `main.py`!!!). |

###  7. Paleidimui pagal laiką
| Failas                     | Funkcija |
|---------------------------|----------|
| `trigeris_pagal_laika.py` | Paleidžia modelio prognozę periodiškai kas N sekundžių . |

###  8. Paleidimo failas
| Failas      | Funkcija |
|-------------|----------|
| `main.py`   | centralizuotas valdymo skriptas, leidžiantis vartotojui pasirinkti modelio tipą (LSTM arba GRU), paleisti jų treniravimą ar testavimą; neskirtas vykdyti savarankiškų modulių, tokių kaip LV_out_sp_skaiciavimas.py ar Optuna_tik_analize.py |

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
     v                              |
LSTM_testavimas.py                  v
                             GRU_testavimas.py
     |                              |
     | <────────────────────────────+
     |
     +────> modelio_testavimo_analize.py

LV_out_sp_skaiciavimas.py ─────> generatorius_2.py
                                └────> LSTM_modelio_arch.py

trigeris_pagal_laika.py → periodinis `predict()` paleidimas
main.py → jungia visus komponentus, išskyrus LV_out_sp_skaiciavimas.py ir Optuna_tik_analize.py - savarankiški skriptai.
```

---

##  Reikalavimai 

```
alembic==1.15.2
colorama==0.4.6
colorlog==6.9.0
contourpy==1.3.1
cycler==0.12.1
filelock==3.16.1
fonttools==4.57.0
fsspec==2024.10.0
greenlet==3.1.1
Jinja2==3.1.4
joblib==1.4.2
kiwisolver==1.4.8
Mako==1.3.9
MarkupSafe==2.1.5
matplotlib==3.10.1
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.4
optuna==4.2.1
packaging==24.2
pandas==2.2.3
patsy==1.0.1
pillow==11.0.0
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
scikit-learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
setuptools==70.2.0
six==1.17.0
SQLAlchemy==2.0.40
statsmodels==0.14.4
sympy==1.13.3
threadpoolctl==3.6.0
torch==2.8.0.dev20250402+cu128
torchaudio==2.6.0.dev20250403+cu128
torchvision==0.22.0.dev20250403+cu128
tqdm==4.67.1
typing_extensions==4.12.2
tzdata==2025.2
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

