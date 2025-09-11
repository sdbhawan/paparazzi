import time
from rotwing_mon.rotwing_viewer import RWStatusMessage, EscMessage, POWERDEVICEMessage, AIRDATAMessage

class EnergyModel:
    def __init__(self, battery_capacity_Wh=222):  
        """
        Hybrid energy model for VSQP.
        battery_capacity_Wh : nominal capacity of battery (Wh)
        """
        self.capacity_Wh = battery_capacity_Wh
        self.capacity_J = battery_capacity_Wh * 3600
        self.remaining_J = self.capacity_J

        # Static baseline values (from measurements)
        self.baseline_power = {
            "hover": 920,       # W
            "cruise": 300,      # W @ 19 m/s
            "transition": 1500  # W for ~5s
        }
        self.transition_duration = 5.0 #s
        self.transition_cost = 7500  # J

        # Real-time telemetry values (updated by callbacks)
        self.last_update_time = time.time()
        self.voltage = None
        self.current = None
        self.power_measured = None
        self.flight_mode = "hover"  # default
        self.airspeed = None

    # ------------------------------
    # Update from Paparazzi messages
    # ------------------------------
    def update_from_ESC(self, msg):
        """ESCMessage: gives motor rpm, current, voltage, energy"""
        self.current = msg.current
        self.voltage = msg.voltage
        self.power_measured = self.voltage * self.current

    def update_from_POWERDEVICE(self, msg):
        """POWERDEVICEMessage: battery telemetry"""
        self.voltage = msg.voltage
        self.current = msg.current
        self.power_measured = self.voltage * self.current

    def update_from_RWStatus(self, msg):
        """RWStatusMessage: detect mode"""
        self.flight_mode = msg.mode  # e.g. 'hover', 'cruise', 'transition'

    def update_from_AIRDATA(self, msg):
        """AIRDATAMessage: airspeed"""
        self.airspeed = msg.airspeed

    # ------------------------------
    # Hybrid energy estimation
    # ------------------------------
    def estimate_power(self):
        """
        Returns hybrid power estimate [W].
        If real measurements available → blend with static baseline.
        """
        static_est = self.baseline_power.get(self.flight_mode, 500)

        if self.power_measured is not None:
            # Blend measured vs baseline
            alpha = 0.3  # trust 70% measurement, 30% static
            return alpha * static_est + (1 - alpha) * self.power_measured
        else:
            return static_est

    def update_energy_budget(self):
        """Integrate consumed energy over time."""
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        P = self.estimate_power()
        E_used = P * dt
        self.remaining_J = max(0, self.remaining_J - E_used)

    def get_remaining_energy(self):
        return self.remaining_J

    def get_remaining_fraction(self):
        return self.remaining_J / self.capacity_J

    def get_safe_flight_time(self):
        """Estimate time left [s] at current consumption rate."""
        P = self.estimate_power()
        if P > 0:
            return self.remaining_J / P
        return float('inf')
