import threading
import time

def timed_trigger(interval_seconds, action, repeat=True):
    """
    Triggeris, kuris kas `interval_seconds` sekundžių paleidžia `action()` funkciją.

    Parametrai:
        interval_seconds - kiek sekundžių laukti tarp trigerių
        action            - funkcija, kuri bus paleidžiama
        repeat            - jei True, veiks nuolat; jei False, vieną kartą
    """
    try:
        while True:
            print(f"⏱️ Laukiama {interval_seconds} sek...")
            time.sleep(interval_seconds)                                            # Sustabdomas programos vykdymas nustatytam laikui
            print("🚨 Trigeris suveikė!")
            action()                                                                # Vykdom nurodyta funkcija
            if not repeat:                                                          # Jei `repeat` yra False – vykdoma 1 karta
                break
    except KeyboardInterrupt:
        print("Sustabdyta rankiniu būdu.")                                          # Sustabdymas rankiniu budu - Ctrl+C