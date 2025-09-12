import time
import sys
import os
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rotwing_mon.rotwing_viewer import RWStatusMessage, EscMessage, POWERDEVICEMessage, AIRDATAMessage

# Paparazzi environment
PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface
from pprzlink.message import PprzMessage

# Import EnergyModel
from energy_model_dummy import EnergyModel, DataLogger


LOG_HISTORY = deque(maxlen=100)

def log_and_print(ac_id, text):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {text}"
    LOG_HISTORY.append(line)
    print(line)



def main(ac_id="37"):
    energy = EnergyModel(use_model_only=True)
    logger = DataLogger()

    ivy = IvyMessagesInterface("EnergyModel")

    # Subscriptions
    def on_RWStatus(sender, msg):
        rw = RWStatusMessage(msg)
        state_str = rw.get_state() if hasattr(rw, "get_state") else str(rw.state)
        if "HOVER" in state_str:
            energy.flight_mode = "hover"
        elif "FW" in state_str:
            energy.flight_mode = "cruise"
        else:
            energy.flight_mode = "transition"

    def on_AIRDATA(sender, msg):
        air = AIRDATAMessage(msg)
        energy.update_from_AIRDATA(air)

    ivy.subscribe(on_RWStatus, PprzMessage("telemetry", "ROTATING_WING_STATE", ac_id))
    ivy.subscribe(on_AIRDATA, PprzMessage("telemetry", "AIR_DATA", ac_id))

    print(f"Subscribed to messages for AC {ac_id}")

    try:
        while True:
            energy.update_energy_budget()
            logger.log(energy)
            print(f"[Energy] mode={energy.flight_mode}, "
                  f"P_est={energy.estimate_power():.1f} W, "
                  f"Remaining={energy.get_remaining_fraction():.2f}, "
                  f"SafeTime={energy.get_safe_flight_time():.1f} s")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("Shutting down...")
        ivy.shutdown()
        logger.plot()


if __name__ == "__main__":
    main()