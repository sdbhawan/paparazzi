import time
import math
import matplotlib.pyplot as plt



class EnergyModel:
    def __init__(self, battery_capacity_Wh=222, use_model_only=True):
        """
        Hybrid energy model for VSQP.
        - battery_capacity_Wh : nominal capacity of battery (Wh)
        - use_model_only: if True, ignore ESC/POWER_DEVICE and use physics-based model (good for NPS)
        """
        self.capacity_Wh = battery_capacity_Wh
        self.capacity_J = battery_capacity_Wh * 3600
        self.remaining_J = self.capacity_J

        # Nominal power values (from measurements)
        self.baseline_power = {
            "hover": 920,       # W
            "cruise": 300,      # W @ 19 m/s
            "transition": 1500  # W for ~5s
        }
        self.transition_duration = 5.0  # s
        self.transition_start_time = None
        self.transition_prev_mode = None
        self.transition_next_mode = None

        # State variables
        self.last_update_time = time.time()
        self.flight_mode = "hover"
        self.airspeed = 0.0
        self.vertical_speed = 0.0
        self.rotor_speeds = []  # from ROTATING_WING_STATE if available

        # Telemetry values (if available)
        self.voltage = None
        self.current = None
        self.power_measured = None

        # Control flag
        self.use_model_only = use_model_only

    # ------------------------------
    # Update from telemetry messages
    # ------------------------------
    def update_from_ESC(self, msg):
        """ESCMessage: motor telemetry (not available in NPS)"""
        if self.use_model_only:
            return
        self.current = msg.amp
        self.voltage = msg.volt_b
        self.power_measured = self.voltage * self.current

    def update_from_POWERDEVICE(self, msg):
        """POWERDEVICEMessage: battery telemetry (not available in NPS)"""
        if self.use_model_only:
            return
        self.voltage = msg.voltage
        self.current = msg.current
        self.power_measured = self.voltage * self.current

    def update_from_RWStatus(self, msg):
        """RWStatusMessage: contains state info for VTOL"""
        state_str = msg.get_state()

        if "HOVER" in state_str:
            new_mode = "hover"
        elif "FW" in state_str:
            new_mode = "cruise"
        else:
            new_mode = "transition"

        # Transition handling
        if new_mode == "transition" and self.flight_mode != "transition":
            # entering transition
            self.transition_start_time = time.time()
            self.transition_prev_mode = self.flight_mode
            self.transition_next_mode = None
            self.flight_mode = "transition"
        elif new_mode != "transition" and self.flight_mode == "transition":
            # destination mode seen during transition
            self.transition_next_mode = new_mode
        else:
            self.flight_mode = new_mode

        # Store rotor speeds if available
        if hasattr(msg, "rotor_speeds"):
            self.rotor_speeds = msg.rotor_speeds

    def update_from_AIRDATA(self, msg):
        """AIRDATAMessage: airspeed + climb rate"""
        self.airspeed = msg.airspeed
        if hasattr(msg, "vertical_speed"):
            self.vertical_speed = msg.vertical_speed

    # ------------------------------
    # Model-based estimation (for NPS)
    # ------------------------------
    def _estimate_hover_power(self):
        P_hover = self.baseline_power["hover"]

        # Adjust for climb/descend
        if self.vertical_speed > 0:  # climbing
            P_hover *= (1 + 0.5 * self.vertical_speed / 2.0)  # max climb = 2 m/s
        elif self.vertical_speed < 0:  # descending
            P_hover *= (1 - 0.3 * abs(self.vertical_speed) / 1.0)  # max descend = 1 m/s

        # Adjust for rotor speeds if available
        if self.rotor_speeds:
            omega_nom = 1.0  # normalize
            factor = sum([w**3 for w in self.rotor_speeds]) / (len(self.rotor_speeds) * omega_nom**3)
            P_hover *= factor

        return max(P_hover, 50.0)

    def _estimate_cruise_power(self):
        P_cruise = self.baseline_power["cruise"]

        # Penalize deviations from nominal cruise speed (19 m/s)
        if self.airspeed > 0:
            v_cruise = 19.0
            k_v = 0.2
            P_cruise *= (1 + k_v * abs(self.airspeed - v_cruise) / v_cruise)

        # Adjust for climb/descend
        if self.vertical_speed > 0:
            P_cruise *= (1 + 0.5 * self.vertical_speed / 2.0)
        elif self.vertical_speed < 0:
            P_cruise *= (1 - 0.3 * abs(self.vertical_speed) / 3.0)

        return max(P_cruise, 50.0)

    def _estimate_transition_power(self):
        return self.baseline_power["transition"]

    # ------------------------------
    # Hybrid estimation
    # ------------------------------
    def estimate_power(self):
        # Handle timed transition
        if self.flight_mode == "transition":
            if self.transition_start_time is not None:
                elapsed = time.time() - self.transition_start_time
                if elapsed < self.transition_duration:
                    return self.baseline_power["transition"]
                else:
                    # End of transition
                    if self.transition_next_mode:
                        self.flight_mode = self.transition_next_mode
                    else:
                        self.flight_mode = self.transition_prev_mode or "cruise"
                    return self.estimate_power()

        # Blend measured with static baseline
        if not self.use_model_only and self.power_measured is not None:
            static_est = self.baseline_power.get(self.flight_mode, 500)
            alpha = 0.3
            return alpha * static_est + (1 - alpha) * self.power_measured

        # Model-based fallback
        if self.flight_mode == "hover":
            return self._estimate_hover_power()
        elif self.flight_mode == "cruise":
            return self._estimate_cruise_power()
        else:
            return 500.0

    def update_energy_budget(self):
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
        return float("inf")



# class EnergyModel:
#     def __init__(self, battery_capacity_Wh=222, use_model_only=True):  
#         """
#         Hybrid energy model for VSQP.
#         - battery_capacity_Wh : nominal capacity of battery (Wh)
#         - use_model_only: if True, ignore ESC/POWER_DEVICE and use physics-based model (good for NPS)
#         """
#         self.capacity_Wh = battery_capacity_Wh
#         self.capacity_J = battery_capacity_Wh * 3600
#         self.remaining_J = self.capacity_J

#         # Nominal power values (from measurements)
#         self.baseline_power = {
#             "hover": 920,       # W
#             "cruise": 300,      # W @ 19 m/s
#             "transition": 1500  # W for ~5s
#         }
#         self.transition_duration = 5.0 #s
#         self.transition_cost = 7500  # J

#         # State variables
#         self.last_update_time = time.time()
#         self.flight_mode = "hover"
#         self.airspeed = 0.0
#         self.vertical_speed = 0.0
#         self.rotor_speeds = []  # from ROTATING_WING_STATE if available

#         # Telemetry values (if available)
#         self.voltage = None
#         self.current = None
#         self.power_measured = None

#         # Control flag
#         self.use_model_only = use_model_only

#     # ------------------------------
#     # Update from telemetry messages
#     # ------------------------------
#     def update_from_ESC(self, msg):
#         """ESCMessage: motor telemetry (not available in NPS)"""
#         if self.use_model_only:
#             return
#         self.current = msg.amp
#         self.voltage = msg.volt_b
#         self.power_measured = self.voltage * self.current

#     def update_from_POWERDEVICE(self, msg):
#         """POWERDEVICEMessage: battery telemetry (not available in NPS)"""
#         if self.use_model_only:
#             return
#         self.voltage = msg.voltage
#         self.current = msg.current
#         self.power_measured = self.voltage * self.current

#     def update_from_RWStatus(self, msg):
#         """RWStatusMessage: contains state info for VTOL"""
#         # Map to flight mode
#         state_str = msg.get_state()
#         if "HOVER" in state_str:
#             self.flight_mode = "hover"
#         elif "FW" in state_str:
#             self.flight_mode = "cruise"
#         else:
#             self.flight_mode = "transition"

#         # Store rotor speeds if available
#         if hasattr(msg, "rotor_speeds"):
#             self.rotor_speeds = msg.rotor_speeds

#     def update_from_AIRDATA(self, msg):
#         """AIRDATAMessage: airspeed + climb rate"""
#         self.airspeed = msg.airspeed
#         if hasattr(msg, "vertical_speed"):
#             self.vertical_speed = msg.vertical_speed

#     # ------------------------------
#     # Model-based estimation (for NPS)
#     # ------------------------------
#     def _estimate_hover_power(self):
#         P_hover = self.baseline_power["hover"]

#         # Adjust for climb/descend
#         if self.vertical_speed > 0:  # climbing
#             P_hover *= (1 + 0.5 * self.vertical_speed / 2.0)  # max climb = 2 m/s
#         elif self.vertical_speed < 0:  # descending
#             P_hover *= (1 - 0.3 * abs(self.vertical_speed) / 1.0)  # max descend = 1 m/s

#         # Adjust for rotor speeds if available
#         if self.rotor_speeds:
#             omega_nom = 1.0  # normalize to hover baseline
#             factor = sum([w**3 for w in self.rotor_speeds]) / (len(self.rotor_speeds) * omega_nom**3)
#             P_hover *= factor

#         return max(P_hover, 50.0)

#     def _estimate_cruise_power(self):
#         P_cruise = self.baseline_power["cruise"]

#         # Penalize deviations from nominal cruise speed (19 m/s)
#         if self.airspeed > 0:
#             v_cruise = 19.0
#             k_v = 0.2
#             P_cruise *= (1 + k_v * abs(self.airspeed - v_cruise) / v_cruise)

#         # Adjust for climb
#         if self.vertical_speed > 0:  # climbing
#             P_cruise *= (1 + 0.5 * self.vertical_speed / 2.0)
#         elif self.vertical_speed < 0:  # descending
#             P_cruise *= (1 - 0.3 * abs(self.vertical_speed) / 3.0)

#         return max(P_cruise, 50.0)

#     def _estimate_transition_power(self):
#         return self.baseline_power["transition"]

#     # ------------------------------
#     # Hybrid estimation
#     # ------------------------------
#     def estimate_power(self):
#         if not self.use_model_only and self.power_measured is not None:
#             # Blend real telemetry with static baseline
#             static_est = self.baseline_power.get(self.flight_mode, 500)
#             alpha = 0.3  # trust 70% telemetry, 30% baseline
#             return alpha * static_est + (1 - alpha) * self.power_measured

#         # Model-based estimation (NPS or fallback)
#         if self.flight_mode == "hover":
#             return self._estimate_hover_power()
#         elif self.flight_mode == "cruise":
#             return self._estimate_cruise_power()
#         elif self.flight_mode == "transition":
#             return self._estimate_transition_power()
#         else:
#             return 500.0

#     def update_energy_budget(self):
#         now = time.time()
#         dt = now - self.last_update_time
#         self.last_update_time = now

#         P = self.estimate_power()
#         E_used = P * dt
#         self.remaining_J = max(0, self.remaining_J - E_used)

#     def get_remaining_energy(self):
#         return self.remaining_J

#     def get_remaining_fraction(self):
#         return self.remaining_J / self.capacity_J

#     def get_safe_flight_time(self):
#         """Estimate time left [s] at current consumption rate."""
#         P = self.estimate_power()
#         if P > 0:
#             return self.remaining_J / P
#         return float("inf")
    





class DataLogger:
    def __init__(self):
        self.times = []
        self.remaining = []
        self.power_est = []
        self.safe_time = []
        self.flight_mode = []
        self.airspeed = []

        self.start_time = time.time()

    def log(self, energy):
        t = time.time() - self.start_time
        self.times.append(t)
        self.remaining.append(energy.get_remaining_fraction())
        self.power_est.append(energy.estimate_power())
        self.safe_time.append(energy.get_safe_flight_time())
        self.flight_mode.append(energy.flight_mode)
        self.airspeed.append(energy.airspeed if energy.airspeed else 0.0)

    def plot(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Remaining energy
        axs[0].plot(self.times, self.remaining, label="Remaining fraction")
        axs[0].set_ylabel("Energy fraction")
        axs[0].set_ylim(0, 1.05)
        axs[0].legend()
        axs[0].grid(True)

        # Power
        axs[1].plot(self.times, self.power_est, label="Estimated Power [W]")
        axs[1].set_ylabel("Power [W]")
        axs[1].legend()
        axs[1].grid(True)

        # Safe time
        axs[2].plot(self.times, self.safe_time, label="Safe flight time [s]")
        axs[2].set_ylabel("Time left [s]")
        axs[2].legend()
        axs[2].grid(True)

        # Flight mode timeline
        mode_map = {"hover": 0, "transition": 1, "cruise": 2}
        mode_vals = [mode_map.get(m, -1) for m in self.flight_mode]
        axs[3].step(self.times, mode_vals, where="post", label="Flight Mode")
        axs[3].set_yticks([0, 1, 2])
        axs[3].set_yticklabels(["hover", "transition", "cruise"])
        axs[3].set_ylabel("Mode")
        axs[3].set_xlabel("Time [s]")
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()

        # Airspeed vs Power (scatter)
        plt.figure(figsize=(6, 5))
        plt.scatter(self.airspeed, self.power_est, c=self.times, cmap="viridis", s=10)
        plt.colorbar(label="Time [s]")
        plt.xlabel("Airspeed [m/s]")
        plt.ylabel("Power [W]")
        plt.title("Airspeed vs Power")
        plt.grid(True)
        plt.show()