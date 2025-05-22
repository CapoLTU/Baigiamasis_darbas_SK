import threading
import time

#____________________trigerio klase naudojimui GUI_______________________________________
class GuiTimerTrigger:
    def __init__(self, interval_seconds, action):
        self.interval = interval_seconds
        self.action = action
        self._timer = None
        self._running = False

    def _run(self):
        if self._running:
            self.action()  # čia tavo funkcija
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.start()

    def start(self):
        if not self._running:
            self._running = True
            self._run()

    def stop(self):
        self._running = False
        if self._timer is not None:
            self._timer.cancel()


#________________________Naudojimas_______________________________
# def mano_gui_funkcija():
#     print("🔔 Trigeris: čia gali generuoti seką ar kviesti modelį.")

# trigger = GuiTimerTrigger(interval_seconds=5, action=mano_gui_funkcija)

# # Paleidimas (pvz. paspaudus mygtuką GUI lange)
# trigger.start()

# # Sustabdymas (pvz. paspaudus STOP mygtuką)
# # trigger.stop()


#______________________________________trigeris pagal laika_naudoti_Pythone_________________
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
            time.sleep(interval_seconds)
            print("🚨 Trigeris suveikė!")
            action()
            if not repeat:
                break
    except KeyboardInterrupt:
        print("Sustabdyta rankiniu būdu.")

