# waypoints
# drone id? ---> not yet implemented
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
import pymap3d as pm
import time
import threading

################## Communication example part ###########################################################

PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

PPRZ_SRC = os.getenv("PAPARAZZI_SRC", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

# sys.path.append(PPRZ_HOME + "/var/lib/python")
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface

# --- ENU reference origin (from your flight plan) ---


# --- ENU origin (from flight plan) ---
lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0

# --- Parse waypoints from XML (same as you had) ---
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

ehvb_xy = np.array([waypoints[wp][:2] for wp in ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]])
softgeo_xy = np.array([waypoints[wp][:2] for wp in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]])

# --- Victims ---
soft_poly = Polygon(softgeo_xy)
victims = []
while len(victims) < 10:
    x = np.random.uniform(soft_poly.bounds[0], soft_poly.bounds[2])
    y = np.random.uniform(soft_poly.bounds[1], soft_poly.bounds[3])
    if soft_poly.contains(Point(x, y)):
        victims.append([x, y])
victims = np.array(victims)

# --- Probability heatmap ---
all_x = np.concatenate([[p[0] for p in waypoints.values()], ehvb_xy[:,0], softgeo_xy[:,0], victims[:,0]])
all_y = np.concatenate([[p[1] for p in waypoints.values()], ehvb_xy[:,1], softgeo_xy[:,1], victims[:,1]])
margin = 50
min_x, max_x = np.min(all_x)-margin, np.max(all_x)+margin
min_y, max_y = np.min(all_y)-margin, np.max(all_y)+margin

grid_res = 2
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res), np.arange(min_y, max_y, grid_res))
prob_map = np.zeros_like(xv, dtype=float)
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map += np.exp(-(dist/20)**2)
prob_map /= np.max(prob_map)

# --- Build static map ---
plt.ion()
fig, ax = plt.subplots(figsize=(16,12))
ax.set_title("UAV Mission Area (ENU)")
ax.set_aspect('equal', adjustable='datalim')

ax.pcolormesh(xv, yv, prob_map, cmap='coolwarm', alpha=0.6, shading='auto')

for name, (x, y, _) in waypoints.items():
    ax.scatter(x, y, c='blue', marker='o')
    ax.text(x+5, y+5, name, fontsize=8, color='white')

ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')

ax.plot(np.append(ehvb_xy[:,0], ehvb_xy[0,0]),
        np.append(ehvb_xy[:,1], ehvb_xy[0,1]),
        'orange', linewidth=2, label='EHVB / Flyzone')

ax.plot(np.append(softgeo_xy[:,0], softgeo_xy[0,0]),
        np.append(softgeo_xy[:,1], softgeo_xy[0,1]),
        'purple', linewidth=2, label='SoftGeofence')

# UAV dot (dynamic)
uav_dot, = ax.plot([], [], 'ro', markersize=12, label='UAV')

ax.legend()
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)


lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0

# --- Shared UAV state ---
uav_state = {"x": None, "y": None, "z": None}

# --- Callback for messages ---
def on_gps_int(ac_id, msg):
    if msg.name == "GPS_INT":
        lat = float(msg["lat"]) * 1e-7
        lon = float(msg["lon"]) * 1e-7
        alt = float(msg["alt"]) / 100.0  # cm → m
        x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        uav_state["x"], uav_state["y"], uav_state["z"] = x, y, z

    # Custom message for waypoints:

    # elif msg.name == "WP_LIST_ENU":
    #     n = int(msg['nb_wp'])   # convert to int
    #     print(f"\n[WP_LIST_ENU] received {n} waypoints from AC {ac_id}")
    #     for i in range(n):
    #         east = int(msg['east'][i])
    #         north = int(msg['north'][i])
    #         up = int(msg['up'][i])
    #         print(f" WP {i}: east={east}, north={north}, up={up}")

# --- Start Ivy interface ---
interface = IvyMessagesInterface("rotwingframe", ivy_bus="127.255.255.255:2010")
interface.subscribe(on_gps_int)

# # Keep Ivy running
# interface.loop()


# --- Live update loop ---
while True:
    if uav_state["x"] is not None:
        uav_dot.set_data(uav_state["x"], uav_state["y"])
        fig.canvas.draw()
        fig.canvas.flush_events()
    time.sleep(0.1)

# --- Setup map (simplified example, add your waypoints/polygons/victims here) ---
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.set_title("UAV Mission Area (ENU)")
# ax.set_aspect("equal", adjustable="datalim")
# ax.set_xlim(-500, 500)
# ax.set_ylim(-500, 500)

# # UAV dot
# uav_dot, = ax.plot([], [], "ro", markersize=10, label="UAV")
# ax.legend()

# plt.ion()
# plt.show()

# # --- Live update loop ---
# while True:
#     if uav_state["x"] is not None:
#         uav_dot.set_data(uav_state["x"], uav_state["y"])
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#     time.sleep(0.1)

##################################################################################################################



# --- Load XML ---
xml_file = os.path.expanduser("~/paparazzi2/paparazzi/conf/flight_plans/tudelft/rotwing_EHVB_Damian.xml")
tree = ET.parse(xml_file)
root = tree.getroot()

# --- Flight plan reference origin ---
lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0  # origin for ENU conversion (from the flight plan)

# --- Parse waypoints ---
waypoints = {}
for wp in root.findall(".//waypoint"):
    name = wp.attrib.get("name")
    if "lat" in wp.attrib and "lon" in wp.attrib:
        lat, lon = float(wp.attrib["lat"]), float(wp.attrib["lon"])
        alt = float(wp.attrib.get("alt", 0.0))
        # Convert to local ENU relative to lat0/lon0
        x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        waypoints[name] = (x, y, z)
    elif "x" in wp.attrib and "y" in wp.attrib:
        x, y = float(wp.attrib["x"]), float(wp.attrib["y"])
        z = float(wp.attrib.get("z", 0.0))
        waypoints[name] = (x, y, z)

# --- Build EHVB and SoftGeofence polygons in XY ---
ehvb_xy = np.array([waypoints[wp][:2] for wp in ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]])
softgeo_xy = np.array([waypoints[wp][:2] for wp in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]])

print("EHVB XY (meters):\n", ehvb_xy)
print("SoftGeofence XY (meters):\n", softgeo_xy)

# --- Victim placement inside SoftGeofence ---
soft_poly = Polygon(softgeo_xy)
victims = []
while len(victims) < 10:
    x = np.random.uniform(soft_poly.bounds[0], soft_poly.bounds[2])
    y = np.random.uniform(soft_poly.bounds[1], soft_poly.bounds[3])
    if soft_poly.contains(Point(x, y)):
        victims.append([x, y])
victims = np.array(victims)

# --- Plot limits ---
all_x = np.concatenate([
    [p[0] for p in waypoints.values()],
    ehvb_xy[:,0], softgeo_xy[:,0], victims[:,0]
])
all_y = np.concatenate([
    [p[1] for p in waypoints.values()],
    ehvb_xy[:,1], softgeo_xy[:,1], victims[:,1]
])
margin = 50
min_x, max_x = np.min(all_x)-margin, np.max(all_x)+margin
min_y, max_y = np.min(all_y)-margin, np.max(all_y)+margin

# --- Probability heatmap ---
grid_res = 2
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res),
                     np.arange(min_y, max_y, grid_res))
prob_map = np.zeros_like(xv, dtype=float)
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map += np.exp(-(dist/20)**2)
prob_map /= np.max(prob_map)

# --- Plot ---
fig, ax = plt.subplots(figsize=(16,12))
ax.set_title("UAV Mission Area (local ENU)")
ax.set_aspect('equal', adjustable='datalim')

# Heatmap using pcolormesh for correct alignment
ax.pcolormesh(xv, yv, prob_map, cmap='coolwarm', alpha=0.6, shading='auto')

# Waypoints
for name, (x, y, _) in waypoints.items():
    ax.scatter(x, y, c='blue', marker='o')
    ax.text(x+5, y+5, name, fontsize=8, color='white')

# Victims
ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')

# Polygons
if len(ehvb_xy) > 0:
    ax.plot(np.append(ehvb_xy[:,0], ehvb_xy[0,0]),
            np.append(ehvb_xy[:,1], ehvb_xy[0,1]),
            'orange', linewidth=2, label='EHVB / Flyzone')
if len(softgeo_xy) > 0:
    ax.plot(np.append(softgeo_xy[:,0], softgeo_xy[0,0]),
            np.append(softgeo_xy[:,1], softgeo_xy[0,1]),
            'purple', linewidth=2, label='SoftGeofence')

ax.legend()
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
plt.show()



