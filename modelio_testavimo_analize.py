import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_predictions(preds, targets, rolling_window = 10):
    """
      Atliekama prognoziu analize:
    - Skaičiuoja metrikas (MSE, MAE, RMSE, R², MAPE, SMAPE)
    - Braižo klaidų histogramą
    - Braižo scatter grafiką (tikros vs prognozuotos)
    - Braižo slenkančią (rolling) vidutinę klaidą
    """
    
    # Užtikrinam, kad būtų vienmaciai numpy masyvai
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    # Metriku skaiciavimas
    mse = mean_squared_error(targets, preds)                                        # Vidutine kvadratine klaida
    mae = mean_absolute_error(targets, preds)                                       # Vidutine absoliuti klaida
    rmse = np.sqrt(mse)                                                             # Saknis iš MSE
    r2 = r2_score(targets, preds)                                                   # Determinacijos koeficientas

    # MAPE ir SMAPE 
    epsilon = 1e-8                                                                  # Apsauga nuo dalybos iš nulio
    mape = np.mean(np.abs((targets - preds) / (targets + epsilon))) * 100           # MAPE – vidutine procentine klaida
    smape = 100 / len(targets) * np.sum(                                            # SMAPE – simetrinė MAPE versija
        2 * np.abs(preds - targets) / (np.abs(targets) + np.abs(preds) + epsilon)
    )

    # Metriku atvaizdavimas
    print("===== METRIKOS =====")
    print(f"MSE :  {mse:.6f}")
    print(f"MAE :  {mae:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"R²   :  {r2:.6f}")
    print(f"MAPE :  {mape:.2f}%")
    print(f"SMAPE:  {smape:.2f}%")

    # Klaidu analize
    errors = preds - targets                                                        # Skirtumas tarp prognozes ir tikros reiksmes

    # Histogramos braizymas – klaidų pasiskirstymas
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, edgecolor='k')
    plt.title("Prognozių klaidų pasiskirstymas")
    plt.xlabel("Klaida")
    plt.ylabel("Dažnis")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tikros vs prognozės 
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Tikros reikšmės")
    plt.ylabel("Prognozės")
    plt.title("Tikros vs Prognozuotos reikšmės")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # # Apskaiciuojame slenkancia (rolling MAE) vidutine absoliutine klaida
    rolling_error = pd.Series(np.abs(errors)).rolling(window=rolling_window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_error)
    plt.title(f"Rolling MAE (langas={rolling_window})")
    plt.xlabel("Laiko žingsnis")
    plt.ylabel("Vidutinė absoliuti klaida")
    plt.grid(True)
    plt.tight_layout()
    plt.show()