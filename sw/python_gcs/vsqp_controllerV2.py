#!/usr/bin/env python3
import random
import time
from ivy.std_api import *

# --- SETTINGS ---
DRONE_AC_ID = 1  # usually 1 in NPS sim
WAYPOINTS = {
    "p1": (65, 75),
    "p2": (120, 50),
    "p3": (30, 140),
    "p4": (90, 110)
}
HOVER_RADIUS = 15  # meters

# --- Dummy probability map (replace with real one later) ---
prob_map = {wp: random.random() for wp in WAYPOINTS}

# --- Ivy Callbacks ---
def on_cx_change(agent, connected):
    print(f"[Ivy] {agent} {'joined' if connected else 'left'}")

def on_die(agent, id):
    print(f"[Ivy] {agent} is dead")

def on_msg(agent, *args):
    # Example: listen for telemetry if needed
    pass

# --- Helper functions ---
def send_block(block_name):
    msg = f"{DRONE_AC_ID} NAV BLOCK '{block_name}'"
    print(f"[SEND] {msg}")
    IvySendMsg(msg)

def distance(wp1, wp2):
    return ((wp1[0]-wp2[0])**2 + (wp1[1]-wp2[1])**2) ** 0.5

# --- Main logic ---
def pick_and_send():
    # Pick highest probability waypoint
    best_wp = max(prob_map, key=prob_map.get)
    print(f"[LOGIC] Best WP: {best_wp} with prob {prob_map[best_wp]:.2f}")

    # Get current pos (dummy here — replace with telemetry later)
    current_pos = (0, 0)  # HOME
    dist = distance(current_pos, WAYPOINTS[best_wp])

    if dist <= HOVER_RADIUS:
        send_block(f"stay_{best_wp}")
    else:
        send_block(f"route_{best_wp}")

# --- Ivy Setup ---
IvyInit("victim_search_controller", "Victim Search Ready", 0, on_cx_change, on_die)
IvyStart("127.255.255.255:2010")
IvyBindMsg(on_msg, "(.*)")

# --- Main loop ---
try:
    while True:
        pick_and_send()
        time.sleep(5)  # every 5 seconds
except KeyboardInterrupt:
    print("Stopping controller...")
    IvyStop()
