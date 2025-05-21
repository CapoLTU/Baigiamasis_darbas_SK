import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_predictions(preds, targets, rolling_window = 10):
    """
    Atliekama detali analizė tarp prognozuotų ir tikrų reikšmių.
    """
    # Užtikrinam, kad būtų numpy array
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    # Metrikos
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    # MAPE ir SMAPE 
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - preds) / (targets + epsilon))) * 100
    smape = 100 / len(targets) * np.sum(
        2 * np.abs(preds - targets) / (np.abs(targets) + np.abs(preds) + epsilon)
    )

    # Spausdinam metrikas
    print("===== METRIKOS =====")
    print(f"MSE :  {mse:.6f}")
    print(f"MAE :  {mae:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"R²   :  {r2:.6f}")
    print(f"MAPE :  {mape:.2f}%")
    print(f"SMAPE:  {smape:.2f}%")

    # Klaidos
    errors = preds - targets

    # Klaidų histogramą
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

    # Rolling klaida
    rolling_error = pd.Series(np.abs(errors)).rolling(window=rolling_window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_error)
    plt.title(f"Rolling MAE (langas={rolling_window})")
    plt.xlabel("Laiko žingsnis")
    plt.ylabel("Vidutinė absoliuti klaida")
    plt.grid(True)
    plt.tight_layout()
    plt.show()