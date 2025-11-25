#!/usr/bin/env python3
"""
paparazzi_bridge.py

- Dynamic multi-UAV discovery (ac_id)
- Reads INS + GPS_INT from Paparazzi (PPRZLink)
- Feeds states to SwarmPlanner (multi_uav_new_runtime.py)
- Sends GUIDED_SETPOINT_NED velocity setpoints
- MAVLink-style smoothing
- Per-UAV logs
"""

import os, sys, time, math, csv, threading
from collections import defaultdict
import numpy as np
import pymap3d as pm

# ---------------- Paparazzi / PPRZLink ----------------
PPRZ_HOME = os.getenv(
    "PAPARAZZI_HOME",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
)
sys.path.append(os.path.join(PPRZ_HOME, "sw/ext/pprzlink/lib/v1.0/python"))

from pprzlink.ivy import IvyMessagesInterface
from pprzlink.message import PprzMessage

# ---------------- Planner wrapper ----------------
from multi_uav_new_runtime import SwarmPlanner

# ---------------- ENU reference ----------------
LAT0, LON0, ALT0 = 52.1681551, 4.4126468, 0.0

# ---------------- INS scales ----------------
POS_SCALE = 0.0039063
VEL_SCALE = 0.0000019

# ---------------- Parameters ----------------
PLANNER_DT = 0.2       # 5 Hz
SMOOTHING_ON = True
MAX_ACCEL = 3.0        # m/s^2 slew limit
ALPHA_LP = 0.3         # low-pass filter
MAX_CMD_SPEED = 15.0

LOG_DIR = "logs_bridge"
os.makedirs(LOG_DIR, exist_ok=True)

UAVS = {}
LOCK = threading.Lock()

def ensure_uav(ac_id):
    with LOCK:
        if ac_id in UAVS:
            return UAVS[ac_id]

        st = {"x": None,"y": None,"z": None,"vx":0.0,"vy":0.0,"vz":0.0,"heading":None}
        cmd_prev = np.zeros(3)

        lf = open(os.path.join(LOG_DIR, f"uav_{ac_id:02d}.csv"), "w", newline="")
        w = csv.writer(lf)
        w.writerow(["t","x","y","z","vx","vy","vz","vx_raw","vy_raw","vz_raw","vx_cmd","vy_cmd","vz_cmd"])
        lf.flush()

        UAVS[ac_id] = {"state": st, "cmd_prev": cmd_prev, "log_f": lf, "log_w": w}
        print(f"[BRIDGE] Registered AC{ac_id}")
        return UAVS[ac_id]

def on_gps_int(ac_id, msg):
    if msg.name != "GPS_INT": return
    ac_id = int(ac_id)
    uav = ensure_uav(ac_id)

    lat = float(msg["lat"]) * 1e-7
    lon = float(msg["lon"]) * 1e-7
    alt = float(msg["alt"]) / 100.0

    x, y, z = pm.geodetic2enu(lat, lon, alt, LAT0, LON0, ALT0)
    with LOCK:
        uav["state"]["x"] = x
        uav["state"]["y"] = y
        uav["state"]["z"] = z

def on_ins(ac_id, msg):
    if msg.name != "INS":
        return

    ac_id = int(ac_id)
    uav = ensure_uav(ac_id)

    ins_x  = float(msg["ins_x"])
    ins_y  = float(msg["ins_y"])
    ins_z  = float(msg["ins_z"])
    ins_xd = float(msg["ins_xd"])
    ins_yd = float(msg["ins_yd"])
    ins_zd = float(msg["ins_zd"])

    north_m = ins_x * POS_SCALE
    east_m  = ins_y * POS_SCALE
    up_m    = -ins_z * POS_SCALE

    north_v = ins_xd * VEL_SCALE
    east_v  = ins_yd * VEL_SCALE
    up_v    = -ins_zd * VEL_SCALE

    heading = math.atan2(east_v, north_v) if (abs(east_v)>1e-6 or abs(north_v)>1e-6) else None
    heading_deg = math.degrees(heading) if heading is not None else float('nan')

    # --- store ---
    with LOCK:
        st = uav["state"]
        st["x"], st["y"], st["z"] = east_m, north_m, up_m
        st["vx"], st["vy"], st["vz"] = east_v, north_v, up_v
        st["heading"] = heading

    # --- DEBUG PRINT ---
    print(f"INS -> AC{ac_id}: x={east_m:.2f}, y={north_m:.2f}, z={up_m:.2f}, "
          f"vx={east_v:.2f}, vy={north_v:.2f}, vz={up_v:.2f}, "
          f"heading={heading_deg:.1f}°")

def ned_from_enu(vx_enu, vy_enu, vz_enu):
    return vy_enu, vx_enu, -vz_enu

def smooth(cmd_prev, cmd_des, dt):
    if dt <= 0: return cmd_des
    max_delta = MAX_ACCEL * dt
    delta = np.clip(cmd_des - cmd_prev, -max_delta, max_delta)
    v_slew = cmd_prev + delta
    return ALPHA_LP * cmd_des + (1-ALPHA_LP) * v_slew

def send_guided(interface, ac_id, vx_enu, vy_enu, vz_enu=0.0):
    """
    Multi-UAV safe wrapper to send GUIDED_SETPOINT_NED velocity commands
    using ENU input (vx, vy, vz) from the planner.
    """

    # Convert ENU → NED (same convention used in example script)
    vx_ned = vy_enu
    vy_ned = vx_enu
    vz_ned = -vz_enu

    msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
    msg['ac_id'] = int(ac_id)
    msg['flags'] = np.packbits([0,0,0,0,0,1,0,0], bitorder='little')[0].astype(np.uint8)

    msg['x'] = float(vx_ned)
    msg['y'] = float(vy_ned)
    msg['z'] = float(vz_ned)
    msg['yaw'] = 0.0

    interface.send(msg, ac_id=int(ac_id))
    print(f"[SEND] AC{ac_id}: vx_enu={vx_enu:.2f}, vy_enu={vy_enu:.2f}, vx_ned={vx_ned:.2f}, vy_ned={vy_ned:.2f}")


def ivy_callback(agent, *args):
    print("IVY RAW:", args)

def on_pprz_msg(ac_id, msg):

    # Only print message names for messages we care about
    if msg.name in ["INS", "INS_EKF2", "GPS_INT"]:
        print(f"[RECV] AC{ac_id} -> {msg.name}")

    # Forward messages to the proper handlers:
    if msg.name == "GPS_INT":
        on_gps_int(ac_id, msg)

    elif msg.name == "INS":
        on_ins(ac_id, msg)

    elif msg.name == "INS_EKF2":
        # Many setups use INS_EKF2 as better replacement → treat like INS
        on_ins(ac_id, msg)


def planner_thread(interface):
    planner = SwarmPlanner()
    t0 = time.time()
    last = t0

    while True:
        now = time.time()
        dt = now - last
        if dt < PLANNER_DT:
            time.sleep(PLANNER_DT - dt)
            now = time.time()
            dt = now - last
        last = now
        t_sim = now - t0

        with LOCK:
            ac_ids = list(UAVS.keys())
            states = {ac: UAVS[ac]["state"].copy() for ac in ac_ids}

        if not ac_ids:
            continue

        # --- planner step ---
        cmd_raw = planner.step(states, t_sim)

        # --- send commands per UAV ---
        for ac_id in ac_ids:
            uav = UAVS[ac_id]
            st = states[ac_id]
            vx_raw, vy_raw, vz_raw = cmd_raw[ac_id]

            # clamp speed
            sp = math.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)
            if sp > MAX_CMD_SPEED:
                s = MAX_CMD_SPEED/(sp+1e-6)
                vx_raw, vy_raw, vz_raw = vx_raw*s, vy_raw*s, vz_raw*s

            des = np.array([vx_raw, vy_raw, vz_raw])
            prev = uav["cmd_prev"]
            cmd = smooth(prev, des, dt) if SMOOTHING_ON else des
            uav["cmd_prev"] = cmd.copy()

            vx_cmd, vy_cmd, vz_cmd = cmd.tolist()
            send_guided(interface, ac_id, vx_cmd, vy_cmd, vz_cmd)

            # logging
            uav["log_w"].writerow([
                t_sim, st["x"], st["y"], st["z"], st["vx"], st["vy"], st["vz"],
                vx_raw, vy_raw, vz_raw,
                vx_cmd, vy_cmd, vz_cmd
            ])
            uav["log_f"].flush()



def manual_test(interface):
    """
    Simple test: send different velocity commands to two UAVs.
    AC 44 moves (vx=1, vy=5), AC 46 moves (vx=-1, vy=-5)
    """

    while True:
        send_guided(interface, 44,  1.0,  5.0, 0.0)
        send_guided(interface, 46, -1.0, -5.0, 0.0)

        print("Sent test commands: AC44 -> (1,5), AC46 -> (-1,-5)")
        time.sleep(0.2)   # 5 Hz


def main():
    ivy_bus = os.getenv("PPRZ_IVY_BUS", "127.255.255.255:2010")
    interface = IvyMessagesInterface("multi_uav_bridge", ivy_bus=ivy_bus)
    interface.subscribe(on_pprz_msg)     # debug
    interface.subscribe(on_gps_int)
    interface.subscribe(on_ins)
    interface.subscribe(lambda ac_id, msg: on_ins(ac_id, msg) if msg.name=="INS_EKF2" else None)


    interface.start()

    # th = threading.Thread(target=planner_thread, args=(interface,), daemon=True)
    # th.start()
    print("[BRIDGE] Running manual velocity test.")
    manual_test(interface)


    print("[BRIDGE] Running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        interface.stop()
        with LOCK:
            for uav in UAVS.values():
                uav["log_f"].close()

if __name__ == "__main__":
    main()
