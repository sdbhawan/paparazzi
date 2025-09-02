import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import xml.etree.ElementTree as ET
import pymap3d as pm
import sys

# --- Paparazzi home ---

################## Communication example part ###########################################################

PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

PPRZ_SRC = os.getenv("PAPARAZZI_SRC", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

# sys.path.append(PPRZ_HOME + "/var/lib/python")
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface
from pprzlink.message import PprzMessage

# --- Flight plan origin ---
lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0

# --- Waypoints and SoftGeofence ---
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


# --- Build polygons ---
ehvb_xy = np.array([waypoints[wp][:2] for wp in ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]])
softgeo_xy = np.array([waypoints[wp][:2] for wp in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]])
soft_poly = Polygon(softgeo_xy)

# --- Victims ---
victims = []
while len(victims) < 5:
    x = np.random.uniform(soft_poly.bounds[0], soft_poly.bounds[2])
    y = np.random.uniform(soft_poly.bounds[1], soft_poly.bounds[3])
    if soft_poly.contains(Point(x, y)):
        victims.append([x, y])
victims = np.array(victims)
victim_found = np.zeros(len(victims), dtype=bool)

# --- Probability map ---
grid_res = 2
min_x, max_x = np.min(softgeo_xy[:,0])-50, np.max(softgeo_xy[:,0])+50
min_y, max_y = np.min(softgeo_xy[:,1])-50, np.max(softgeo_xy[:,1])+50
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res), np.arange(min_y, max_y, grid_res))
prob_map = np.zeros_like(xv, dtype=float)
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map += np.exp(-(dist/20)**2)
prob_map /= np.max(prob_map)

# --- Ivy interface ---
interface = IvyMessagesInterface("single_uav_swarm", ivy_bus="127.255.255.255:2010")

# --- UAV state ---
uav_pos = np.array([0.0, 0.0])
uav_target_idx = -1
ghost_wp = uav_pos.copy()

# --- Map setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(12,10))
ax.set_aspect('equal')
heatmap = ax.pcolormesh(xv, yv, prob_map, cmap='coolwarm', alpha=0.6, shading='auto')
ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')
ax.plot(np.append(ehvb_xy[:,0], ehvb_xy[0,0]), np.append(ehvb_xy[:,1], ehvb_xy[0,1]), 'orange', linewidth=2, label='EHVB')
ax.plot(np.append(softgeo_xy[:,0], softgeo_xy[0,0]), np.append(softgeo_xy[:,1], softgeo_xy[0,1]), 'purple', linewidth=2, label='SoftGeofence')
uav_dot, = ax.plot([], [], 'go', markersize=10, label='UAV')
ax.legend()
plt.show()

# --- Helper to send MOVE_WP ---
#define WP_Ghost 12
def send_move_wp(ac_id, lat, lon, alt):
    # Build message using message class
    msg = PprzMessage("datalink", "MOVE_WP")
    msg['wp_id'] = int(12)  # this can be the ghost waypoint
    msg['ac_id'] = int(ac_id)
    msg['lat'] = int(lat*1e7)
    msg['lon'] = int(lon*1e7)
    msg['alt'] = int(alt*1000)
    interface.send(msg)
    print(f"Sent MOVE_WP to AC {ac_id}: lat={lat}, lon={lon}, alt={alt}")

# --- Swarm step for single UAV ---
def swarm_step(ac_id):
    global uav_target_idx, ghost_wp
    # Assign closest un-found victim
    if uav_target_idx == -1 or victim_found[uav_target_idx]:
        un_found = [i for i, v in enumerate(victim_found) if not v]
        if un_found:
            dists = np.linalg.norm(victims[un_found] - uav_pos, axis=1)
            uav_target_idx = un_found[np.argmin(dists)]
        else:
            uav_target_idx = -1

    # If target exists, send MOVE_WP
    if uav_target_idx != -1:
        ghost_wp = victims[uav_target_idx]
        # ENU → LLH
        lat, lon, alt = pm.enu2geodetic(ghost_wp[0], ghost_wp[1], 50, lat0, lon0, alt0)
        send_move_wp(ac_id, lat, lon, alt)

# --- GPS callback ---
def message_recv(ac_id, msg):
    global uav_pos, uav_target_idx 
    if msg.name == "GPS_INT":
        lat, lon, alt = float(msg['lat'])/1e7, float(msg['lon'])/1e7, float(msg['alt'])/1000.0
        x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        uav_pos = np.array([x, y])

        # Check if victim reached
        if uav_target_idx != -1 and np.linalg.norm(uav_pos - victims[uav_target_idx]) < 2.0:
            victim_found[uav_target_idx] = True
            print(f"Victim {uav_target_idx} found!")
            uav_target_idx = -1

        # Run swarm step and update map
        swarm_step(ac_id)
        uav_dot.set_data([uav_pos[0]], [uav_pos[1]])
        

interface.subscribe(message_recv)

while True:

    fig.canvas.draw() #I cannot do this in the resceive message. 
    fig.canvas.flush_events()

    
# interface.loop()
