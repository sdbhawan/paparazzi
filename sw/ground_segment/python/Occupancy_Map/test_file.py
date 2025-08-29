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

################## Communication example part ###########################################################

PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

PPRZ_SRC = os.getenv("PAPARAZZI_SRC", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

# sys.path.append(PPRZ_HOME + "/var/lib/python")
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface

interface = IvyMessagesInterface("rotwingframe", ivy_bus= "127.255.255.255:2010")


# def message_recv(ac_id, msg):
#     if msg.name == "WP_LIST_ENU":
#         n = int(msg['nb_wp'])   # convert to int
#         print(f"\n[WP_LIST_ENU] received {n} waypoints from AC {ac_id}")
#         for i in range(n):
#             east = int(msg['east'][i])
#             north = int(msg['north'][i])
#             up = int(msg['up'][i])
#             print(f" WP {i}: east={east}, north={north}, up={up}")

# # Subscribe to all messages
# interface.subscribe(message_recv)

# # Keep Ivy running
# interface.loop()

###################################################################################################################


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



while True: 
    pass

################## Old model ########################################

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation settings ---
grid_size = (100, 100)
num_uavs = 3
num_victims = 5
step_size = 1.5  # UAV movement per timestep
detect_thresh = 0.8  # Probability threshold to "detect" a victim
np.random.seed(42)

# --- Victim positions (static) ---
victim_positions = np.random.randint(10, 90, size=(num_victims, 2))
victim_found = np.zeros(num_victims, dtype=bool)

# --- UAV initial positions ---
uav_positions = np.random.randint(0, 100, size=(num_uavs, 2)).astype(float)
uav_paths = [ [pos.copy()] for pos in uav_positions ]
uav_targets = [-1]*num_uavs  # assigned victim indices
ghost_waypoints = uav_positions.copy()

# --- Static probability map ---
x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
prob_map = np.zeros(grid_size)
for pos in victim_positions:
    # dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
    dist = np.sqrt((x - pos[1])**2 + (y - pos[0])**2)

    prob_map += np.exp(-(dist/10)**2)  # Gaussian around each victim
prob_map /= np.max(prob_map)  # normalize to [0,1]

# --- Simulation loop ---
timesteps = 50
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

for t in range(timesteps):
    ax.clear()
    ax.imshow(prob_map.T, origin='lower', cmap='coolwarm', alpha=0.5)
    
    # Assign UAVs to closest un-found victim
    for i in range(num_uavs):
        if uav_targets[i] == -1 or victim_found[uav_targets[i]]:
            # Choose nearest victim not yet found
            assigned = set()
            for i in range(num_uavs):
                if uav_targets[i] == -1 or victim_found[uav_targets[i]]:
                    un_found_idx = [idx for idx in range(num_victims) if not victim_found[idx] and idx not in assigned]
                    if len(un_found_idx) > 0:
                        distances = np.linalg.norm(uav_positions[i] - victim_positions[un_found_idx], axis=1)
                        chosen = un_found_idx[np.argmin(distances)]
                        uav_targets[i] = chosen
                        assigned.add(chosen)
                    else:
                        uav_targets[i] = -1


    # Update UAV positions towards ghost waypoints
    for i in range(num_uavs):
        target_idx = uav_targets[i]
        if target_idx == -1:
            continue
        # Ghost waypoint is victim position
        ghost_waypoints[i] = victim_positions[target_idx]
        direction = ghost_waypoints[i] - uav_positions[i]
        dist = np.linalg.norm(direction)
        if dist < step_size:
            uav_positions[i] = ghost_waypoints[i].copy()
        else:
            uav_positions[i] += (direction / dist) * step_size

        # Check if victim found
        if np.linalg.norm(uav_positions[i] - victim_positions[target_idx]) < 2.0:
            victim_found[target_idx] = True
            uav_targets[i] = -1

        # Record path
        uav_paths[i].append(uav_positions[i].copy())

    # --- Visualization ---
    ax.imshow(prob_map.T, origin='lower', cmap='coolwarm', alpha=0.5)
    ax.scatter(victim_positions[:,0], victim_positions[:,1], c='red', marker='X', s=100, label='Victims')
    ax.scatter(uav_positions[:,0], uav_positions[:,1], c='blue', label='UAVs')
    # Draw ghost waypoints arrows
    for i in range(num_uavs):
        if uav_targets[i] != -1:
            ax.arrow(uav_positions[i][0], uav_positions[i][1],
                     ghost_waypoints[i][0]-uav_positions[i][0],
                     ghost_waypoints[i][1]-uav_positions[i][1],
                     color='green', head_width=1.5, length_includes_head=True)
        path = np.array(uav_paths[i])
        ax.plot(path[:,0], path[:,1], 'b--', alpha=0.5)
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_title(f"Timestep {t}")
    plt.pause(0.1)

plt.ioff()
plt.show()

