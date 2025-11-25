# multi_uav_new_runtime.py
"""
Thin runtime wrapper for multi_uav_new.py.

Goal:
- reuse your exact planner code
- prevent simulation loop from running on import
- expose a clean SwarmPlanner.step(uav_states) interface
"""

import numpy as np

# Import EVERYTHING you already defined (functions + globals)
# IMPORTANT: multi_uav_new.py must NOT auto-run on import.
# So we do a safe import by telling you to add a small guard.
import multi_uav_paparazzi as core


class SwarmPlanner:
    def __init__(self):
        # copy the planner state from core
        # (belief map, track list, parameters, etc.)
        # We rely on your existing globals.
        self.core = core

        # Override any sim-only initializations you want here.
        # Example: dynamic number of drones:
        self.num_drones = 0
        self.initialized = False

    def _ensure_initialized(self, ac_ids):
        """Initialize only once, dynamically from Paparazzi AC_IDs."""
        if self.initialized:
            return

        self.num_drones = len(ac_ids)

        # Replace hardcoded sim init with real-time structures
        self.core.num_drones = self.num_drones
        self.core.drone_positions = [np.zeros(3) for _ in ac_ids]
        self.core.drone_vels = [np.zeros(3) for _ in ac_ids]
        self.core.drone_modes = ["explore" for _ in ac_ids]

        # Reset tracks / cooldown arrays to match count
        self.core.track = []
        for _ in range(self.num_drones):
            self.core.track.append({
                "active": False,
                "confirmed": False,
                "pos": None,
                "time": None,
                "phase": None,
                "lock_start": None,
                "cone_start": None,
                "hover_start": None,
                "fail_timer": 0.0,
                "pos_at_detection": None
            })

        self.core.cooldown_until = [0.0]*self.num_drones

        self.initialized = True
        print(f"[PLANNER] Initialized for {self.num_drones} UAVs: {ac_ids}")

    def step(self, uav_states, t_now):
        """
        One planner step.

        uav_states: dict {ac_id: {"x","y","z","vx","vy","vz","heading"} } ENU
        t_now: float simulation time (seconds)

        Returns:
            cmd_dict: {ac_id: (vx_enu, vy_enu, vz_enu)}
        """
        ac_ids = sorted(uav_states.keys())
        self._ensure_initialized(ac_ids)

        # Map Paparazzi states into planner arrays
        for d_idx, ac_id in enumerate(ac_ids):
            st = uav_states[ac_id]
            self.core.drone_positions[d_idx] = np.array([st["x"], st["y"], st["z"]], dtype=float)
            self.core.drone_vels[d_idx] = np.array([st["vx"], st["vy"], st["vz"]], dtype=float)

        # ---------------------------
        # RUN YOUR EXACT LOOP BODY ONCE
        # ---------------------------
        # Paste the contents of ONE iteration of your simulation while-loop here.
        # Specifically: everything inside the loop that updates belief, tracking,
        # and produces vx_des, vy_des, vz_des for each drone.
        #
        # At the end of that pasted block you should have:
        #   vx_des_list, vy_des_list, vz_des_list or directly drone_vels
        #
        # Wrap it as a function if you want, but DON'T change logic.
        # ---------------------------
        self._step_internal(t_now)

        # Build output command dict from updated desired velocities
        cmd_dict = {}
        for d_idx, ac_id in enumerate(ac_ids):
            vx, vy, vz = self.core.drone_vels[d_idx]
            cmd_dict[ac_id] = (float(vx), float(vy), float(vz))
        return cmd_dict

    def _step_internal(self, t_now):
        """
        IMPORTANT:
        Paste one full loop iteration from multi_uav_new.py here
        (no plotting, no plt.pause, no sim position integration).
        Just the planner logic.
        """
        # ---- YOUR CODE GOES HERE ----
        # Example minimal placeholder:
        # for d_idx in range(self.num_drones):
        #     vx, vy, vz, *_ = self.core.plan_velocity_ipp_3D(...)
        #     self.core.drone_vels[d_idx][:] = [vx, vy, vz]
        pass
