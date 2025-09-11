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
from energy_model_dummy import EnergyModel


LOG_HISTORY = deque(maxlen=100)

def log_and_print(ac_id, text):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {text}"
    LOG_HISTORY.append(line)
    print(line)


def main(ac_id="37"):
    energy = EnergyModel()

    # Ivy interface
    ivy = IvyMessagesInterface("EnergyModel")

    # --- Subscriptions ---
    def on_ESC(sender, msg):
        esc = EscMessage(msg)
        energy.update_from_ESC(esc)
        log_and_print(ac_id,
            f"[ESC] motor={esc.id}, rpm={esc.rpm:.0f}, "
            f"current={esc.amp:.2f}A, voltage={esc.volt_b:.2f}V, "
            f"temp={esc.temperature:.1f}C, energy={esc.energy:.1f} J"
        )

    def on_POWER(sender, msg):
        power = POWERDEVICEMessage(msg)
        energy.update_from_POWERDEVICE(power)
        log_and_print(ac_id,
            f"[Power] node={power.node_id}, circuit={power.circuit}, "
            f"voltage={power.voltage:.2f}V, current={power.current:.2f}A, "
            f"power={power.voltage * power.current:.1f} W"
        )

    def on_RWStatus(sender, msg):
        rw = RWStatusMessage(msg)

        # Map state → flight mode
        state_str = rw.get_state() if hasattr(rw, "get_state") else str(rw.state)
        if "HOVER" in state_str:
            energy.flight_mode = "hover"
        elif "FW" in state_str:
            energy.flight_mode = "cruise"
        else:
            energy.flight_mode = "transition"

        log_and_print(ac_id,
            f"[RWStatus] skew={rw.meas_skew_angle:.1f}° (sp={rw.sp_skew_angle:.1f}°), "
            f"airspeed={rw.nav_airspeed:.2f} m/s "
            f"[min={rw.min_airspeed:.2f}, max={rw.max_airspeed:.2f}]"
        )

    def on_AIRDATA(sender, msg):
        air = AIRDATAMessage(msg)
        energy.update_from_AIRDATA(air)
        log_and_print(ac_id,
            f"[AirData] airspeed={air.airspeed:.2f} m/s, tas={air.tas:.2f} m/s"
        )

    # Subscribe only to the messages we care about
    ivy.subscribe(on_ESC, PprzMessage("telemetry", "ESC", ac_id))
    ivy.subscribe(on_POWER, PprzMessage("telemetry", "POWER_DEVICE", ac_id))
    ivy.subscribe(on_RWStatus, PprzMessage("telemetry", "ROTATING_WING_STATE", ac_id))
    ivy.subscribe(on_AIRDATA, PprzMessage("telemetry", "AIR_DATA", ac_id))

    print(f"Subscribed to messages for AC {ac_id}")

    # --- Main loop ---
    try:
        while True:
            energy.update_energy_budget()
            print(f"[Energy] Remaining fraction: {energy.get_remaining_fraction():.2f}, "
                  f"Safe flight time: {energy.get_safe_flight_time():.1f} s")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("Shutting down...")
        ivy.shutdown()

        # Optional: dump the last N messages
        print("\n--- Last Messages ---")
        for line in LOG_HISTORY:
            print(line)


if __name__ == "__main__":
    main()
