# A test to see if communication works

#!/usr/bin/env python3

from ivy.std_api import *
import time

# === Configuration ===
agent_name = "VSQP_GCS"
ready_message = "Hello from VSQP_GCS"
broadcast_address = "127.255.255.255:2010"  # default for Paparazzi
drone_ac_id = 1  # replace with your actual aircraft ID

# === Callback when Ivy is ready ===
def on_ready():
    print("Ivy is now ready. Listening and sending messages...")

# === Callback for all messages (you can filter here) ===
def on_msg(agent, *args):
    print(f"[{agent}] --> {args[0]}")

# === Start Ivy Bus ===
IvyInit(agent_name, ready_message, 0, on_msg, on_ready)
IvyStart(broadcast_address)

# === Example: send a command after some time ===
time.sleep(2)

# This triggers a block named "Search" on your UAV
IvySendMsg(f"{drone_ac_id} NAV BLOCK 'Search'")

# You can also try: IvySendMsg(f"{drone_ac_id} NAV GOTO WP wp1") to go to a waypoint

# === Keep script running ===
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down Ivy...")
    IvyStop()
