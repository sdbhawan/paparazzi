import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from math import hypot
import os
import sys
import xml.etree.ElementTree as ET
from pyproj import Proj, transform
import pymap3d as pm
import time
import threading
from math import atan2, hypot, degrees, cos, sin
import random


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
margin = 200
min_x, max_x = np.min(all_x)-margin, np.max(all_x)+margin
min_y, max_y = np.min(all_y)-margin, np.max(all_y)+margin

grid_res = 20   # coarse grid cells
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res),
                     np.arange(min_y, max_y, grid_res))

# start with uniform background
prob_map = np.ones_like(xv, dtype=float) * 0.1  

# add victim hotspots
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map += np.exp(-(dist/100)**2)   # wider spread for coarse grid

# normalize to proper probability distribution
prob_map /= np.sum(prob_map)


# -------------------------------
# Assuming prob_map, xv, yv, softgeo_xy, ehvb_xy, victims already defined
grid_res_x = xv.shape[1]
grid_res_y = yv.shape[0]

# -------------------------------
# Simulation settings
# -------------------------------

# -------------------------------
# Assumes your real map is already defined:
# prob_map, xv, yv, victims, ehvb_xy, softgeo_xy
# -------------------------------

# -------------------------------
# Simulation settings
# -------------------------------
num_uavs = 1
timesteps = 300
max_speed = 5.0
detection_radius = 5
base_prob = 0.1
remove_radius = 5
exploration_bonus = 0.01
alpha = 1.0
beta = 0.01
local_window = 3
local_thresh = 0.05  # threshold for switching to boustrophedon

# -------------------------------
# Initialize UAV
# -------------------------------
ehvb_center = np.mean(ehvb_xy, axis=0)
uav_positions = np.array([ehvb_center])
uav_paths = [[ehvb_center.copy()]]
uav_target = None
mode = "exploit"  # "exploit" or "coverage"

# -------------------------------
# Track detected victims
# -------------------------------
victim_found = np.zeros(len(victims), dtype=bool)

# -------------------------------
# Helper functions
# -------------------------------
def recompute_prob_map(victims, victim_found, xv, yv, base_prob=0.1, remove_radius=5, exploration_bonus=0.01):
    prob_map = np.ones_like(xv, dtype=float) * base_prob
    for idx, victim_pos in enumerate(victims):
        if not victim_found[idx]:
            dist = np.sqrt((xv - victim_pos[0])**2 + (yv - victim_pos[1])**2)
            prob_map += np.exp(-(dist/50)**2)
        else:
            dist = np.sqrt((xv - victim_pos[0])**2 + (yv - victim_pos[1])**2)
            prob_map[dist <= remove_radius] = base_prob
    prob_map += exploration_bonus
    prob_map /= np.sum(prob_map)
    return prob_map

def select_target_weighted(prob_map, uav_pos, xv, yv, alpha=1.0, beta=0.01, base_prob=0.1):
    masked_prob = np.ma.masked_where(prob_map <= base_prob, prob_map)
    if masked_prob.count() == 0:
        return None
    xx, yy = np.meshgrid(xv[0,:], yv[:,0])
    dist = np.sqrt((xx - uav_pos[0])**2 + (yy - uav_pos[1])**2)
    score = alpha * masked_prob - beta * dist
    ty, tx = np.unravel_index(np.argmax(score), prob_map.shape)
    return tx, ty

def grid_to_coords(tx, ty, xv, yv):
    return np.array([xv[0, tx], yv[ty, 0]])

def move_toward(current_pos, target_pos, max_speed):
    direction = target_pos - current_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return current_pos.copy(), np.array([0.0,0.0]), 0.0
    speed = min(max_speed, dist)
    velocity = (direction / dist) * speed
    new_pos = current_pos + velocity
    heading = atan2(velocity[1], velocity[0])
    return new_pos, velocity, heading

def local_contrast(prob_map, uav_pos, xv, yv, window=3):
    ix = np.abs(xv[0,:] - uav_pos[0]).argmin()
    iy = np.abs(yv[:,0] - uav_pos[1]).argmin()
    half = window // 2
    y_min = max(0, iy-half)
    y_max = min(prob_map.shape[0], iy+half+1)
    x_min = max(0, ix-half)
    x_max = min(prob_map.shape[1], ix+half+1)
    neighborhood = prob_map[y_min:y_max, x_min:x_max]
    contrast = (np.max(neighborhood) - np.min(neighborhood)) / (np.max(neighborhood)+1e-6)
    return contrast

def boustrophedon_step(uav_pos, step_size=5.0, direction=1, xv=None, yv=None):
    # Move horizontally in the grid direction, then step vertically
    ix = np.abs(xv[0,:] - uav_pos[0]).argmin()
    iy = np.abs(yv[:,0] - uav_pos[1]).argmin()
    nx = ix + direction
    if nx >= xv.shape[1] or nx < 0:
        nx = ix
        ny = iy + 1
        if ny >= yv.shape[0]:
            ny = 0
        tx, ty = nx, ny
    else:
        tx, ty = nx, iy
    target = np.array([xv[0, tx], yv[ty,0]])
    new_pos, velocity, heading = move_toward(uav_pos, target, step_size)
    return new_pos, velocity, heading, direction

boustrophedon_dir = 1  # 1: right, -1: left

# -------------------------------
# Simulation loop
# -------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10,10))

for t in range(timesteps):
    ax.clear()
    ax.set_title(f"Timestep {t}")
    ax.set_aspect('equal', adjustable='datalim')

    # --- Recompute probability map ---
    prob_map = recompute_prob_map(victims, victim_found, xv, yv, base_prob, remove_radius, exploration_bonus)

    # --- Mode switching based on local contrast ---
    contrast = local_contrast(prob_map, uav_positions[0], xv, yv, window=local_window)
    if mode=="exploit" and contrast < local_thresh:
        mode="coverage"
        print(f"Switching to boustrophedon at t={t} (local contrast {contrast:.3f})")
    elif mode=="coverage" and contrast >= local_thresh:
        mode="exploit"
        print(f"Switching back to IPP at t={t} (local contrast {contrast:.3f})")

    # --- Determine next target ---
    if mode=="exploit":
        target_cell = select_target_weighted(prob_map, uav_positions[0], xv, yv, alpha, beta, base_prob)
        if target_cell is not None:
            tx, ty = target_cell
            target_pos = grid_to_coords(tx, ty, xv, yv)
        else:
            target_pos = None
        if target_pos is not None:
            new_pos, velocity, heading = move_toward(uav_positions[0], target_pos, max_speed)
        else:
            new_pos, velocity, heading, _ = boustrophedon_step(uav_positions[0], max_speed, boustrophedon_dir, xv, yv)
    else:  # coverage
        new_pos, velocity, heading, boustrophedon_dir = boustrophedon_step(uav_positions[0], max_speed, boustrophedon_dir, xv, yv)

    uav_positions[0] = new_pos
    uav_paths[0].append(new_pos.copy())

    # --- Print velocity and heading ---
    print(f"Time {t}: pos=({new_pos[0]:.1f},{new_pos[1]:.1f}) "
          f"vx={velocity[0]:.2f}, vy={velocity[1]:.2f}, heading={degrees(heading):.1f}°")

    # --- Check for victim detection ---
    for idx, victim_pos in enumerate(victims):
        if not victim_found[idx] and np.linalg.norm(new_pos - victim_pos) <= detection_radius:
            victim_found[idx] = True
            print(f"  Victim at ({victim_pos[0]:.1f},{victim_pos[1]:.1f}) detected.")

    # --- Visualization ---
    ax.pcolormesh(xv, yv, prob_map, cmap='coolwarm', alpha=0.6, shading='auto')
    ax.scatter(uav_positions[:,0], uav_positions[:,1], c='blue', s=80, label='UAV')
    path_arr = np.array(uav_paths[0])
    ax.plot(path_arr[:,0], path_arr[:,1], 'b--', alpha=0.5)
    ax.plot(np.append(ehvb_xy[:,0], ehvb_xy[0,0]),
            np.append(ehvb_xy[:,1], ehvb_xy[0,1]),
            'orange', linewidth=2, label='EHVB / Flyzone')
    ax.plot(np.append(softgeo_xy[:,0], softgeo_xy[0,0]),
            np.append(softgeo_xy[:,1], softgeo_xy[0,1]),
            'purple', linewidth=2, label='SoftGeofence')
    ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')
    ax.legend()
    plt.pause(0.05)

plt.ioff()
plt.show()