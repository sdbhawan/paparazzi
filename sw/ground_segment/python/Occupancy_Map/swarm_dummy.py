import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pymap3d as pm
import time
import threading

# Paparazzi paths
PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')))
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface
from pprzlink.message import PprzMessage

# --- ENU origin (from flight plan) ---
lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0

# --- Parse waypoints from XML (handles lat/lon and x/y) ---
xml_file = os.path.expanduser("~/paparazzi2/paparazzi/conf/flight_plans/tudelft/rotwing_EHVB_Damian.xml")
tree = ET.parse(xml_file)
root = tree.getroot()

waypoints = {}
for wp in root.findall(".//waypoint"):
    name = wp.attrib.get("name")
    if "lat" in wp.attrib and "lon" in wp.attrib:
        lat, lon = float(wp.attrib["lat"]), float(wp.attrib["lon"])
        alt = float(wp.attrib.get("alt", 0.0))
        x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        waypoints[name] = (x, y, z)
    elif "x" in wp.attrib and "y" in wp.attrib:
        x, y = float(wp.attrib["x"]), float(wp.attrib["y"])
        z = float(wp.attrib.get("z", 0.0))
        waypoints[name] = (x, y, z)

# Build polygons safely (if keys missing, produce empty arrays)
def safe_poly(names):
    pts = []
    for n in names:
        if n in waypoints:
            pts.append(waypoints[n][:2])
    return np.array(pts) if pts else np.empty((0,2))

ehvb_xy = safe_poly(["C1","C2","C3","C4","C5","C6","C7","C8","C9"])
softgeo_xy = safe_poly(["S1","S2","S3","S4","S5","S6","S7","S8","S9"])

# --- Victims (random inside soft geofence) ---
soft_poly = Polygon(softgeo_xy)
victims = []
while len(victims) < 10:
    x = np.random.uniform(soft_poly.bounds[0], soft_poly.bounds[2])
    y = np.random.uniform(soft_poly.bounds[1], soft_poly.bounds[3])
    if soft_poly.contains(Point(x, y)):
        victims.append([x, y])
victims = np.array(victims)
victim_found = np.zeros(len(victims), dtype=bool)

# --- Probability heatmap (coarse) ---
all_x = np.concatenate([[p[0] for p in waypoints.values()] , ehvb_xy[:,0] if ehvb_xy.size else np.array([]), softgeo_xy[:,0] if softgeo_xy.size else np.array([]), victims[:,0]])
all_y = np.concatenate([[p[1] for p in waypoints.values()] , ehvb_xy[:,1] if ehvb_xy.size else np.array([]), softgeo_xy[:,1] if softgeo_xy.size else np.array([]), victims[:,1]])
margin = 50
min_x, max_x = np.min(all_x)-margin, np.max(all_x)+margin
min_y, max_y = np.min(all_y)-margin, np.max(all_y)+margin

grid_res = 20
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res), np.arange(min_y, max_y, grid_res))
prob_map = np.ones_like(xv, dtype=float) * 0.1
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map += np.exp(-(dist/50)**2)
prob_map /= np.sum(prob_map)

# --- Plot setup (static layers) ---
plt.ion()
fig, ax = plt.subplots(figsize=(16,12))
ax.set_title("UAV Mission Area (ENU)")
ax.set_aspect('equal', adjustable='datalim')
heat = ax.pcolormesh(xv, yv, prob_map, cmap='coolwarm', alpha=0.6, shading='auto')

# draw waypoints
for name, (x, y, _) in waypoints.items():
    ax.scatter(x, y, c='blue', marker='o')
    ax.text(x+5, y+5, name, fontsize=8, color='white')

# victims and polygons
ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')
if ehvb_xy.size:
    ax.plot(np.append(ehvb_xy[:,0], ehvb_xy[0,0]), np.append(ehvb_xy[:,1], ehvb_xy[0,1]), 'orange', linewidth=2, label='EHVB / Flyzone')
if softgeo_xy.size:
    ax.plot(np.append(softgeo_xy[:,0], softgeo_xy[0,0]), np.append(softgeo_xy[:,1], softgeo_xy[0,1]), 'purple', linewidth=2, label='SoftGeofence')

uav_dot, = ax.plot([], [], 'ro', markersize=12, label='UAV')
ax.legend()
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# --- Messaging interface ---
interface = IvyMessagesInterface("single_uav_swarm", ivy_bus="127.255.255.255:2010")

# shared UAV state (single UAV)
uav_state = {"lat": None, "lon": None, "alt": None, "x": None, "y": None, "z": None}
uav_target_idx = -1

# GPS callback expects signature (ac_id, parsed_msg)
def on_gps_int(ac_id, msg):
    # check msg object (pprzlink produces a message-like dict)
    try:
        if msg.name == "GPS_INT":
            lat = float(msg["lat"]) * 1e-7
            lon = float(msg["lon"]) * 1e-7
            alt = float(msg["alt"]) / 1000.0   # mm->m in your definition earlier; if cm adjust
            x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
            uav_state.update({"lat": lat, "lon": lon, "alt": alt, "x": x, "y": y, "z": z})
    except Exception as e:
        # defensive: log and ignore malformed messages
        print("on_gps_int error:", e)

# IMPORTANT: subscribe without regex/pattern so pprzlink passes parsed msg objects
interface.subscribe(on_gps_int)

# start Ivy in background thread so the main thread can run plotting loop
# def run_ivy_loop():
#     try:
#         interface.loop()
#     except Exception as e:
#         print("Ivy loop ended:", e)

t = threading.Thread(target=run_ivy_loop, daemon=True)
t.start()

# --- send MOVE_WP helper (use PprzMessage with datalink class) ---
def send_move_wp(ac_id, lat, lon, alt):
    msg = PprzMessage("datalink", "MOVE_WP")
    msg['wp_id'] = int(12)       # ghost waypoint
    msg['ac_id'] = int(ac_id)    # which UAV to move
    msg['lat'] = int(lat*1e7)    # ENU→LLH conversion already done before
    msg['lon'] = int(lon*1e7)
    msg['alt'] = int(alt*1000)
    interface.send(msg)
    print(f"Sent MOVE_WP to AC {ac_id}: lat={lat}, lon={lon}, alt={alt}")


# --- Example: map-driven target selection (highest-prob cell) ---
def pick_highest_prob_target():
    # find argmax in prob_map, convert grid cell center to ENU coords
    idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
    gx = xv[idx]
    gy = yv[idx]
    return np.array([gx, gy])

# --- Background command thread: send a MOVE_WP toward highest-prob every 10s ---
def command_loop():
    global uav_target_idx 
    time.sleep(3)
    while True:
        if uav_state["x"] is not None:
            target_enu = pick_highest_prob_target()
            # convert target ENU -> LLH (choose 50 m as alt)
            lat, lon, alt = pm.enu2geodetic(float(target_enu[0]), float(target_enu[1]), 50.0, lat0, lon0, alt0)
            send_move_wp(37, lat, lon, alt)  # ghost WP = 12
        time.sleep(10)

threading.Thread(target=command_loop, daemon=True).start()

# --- Main plotting loop (non-blocking Ivy runs in background) ---
try:
    while True:
        if uav_state["x"] is not None:
            uav_dot.set_data(uav_state["x"], uav_state["y"])
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Exiting...")
    interface.shutdown()
