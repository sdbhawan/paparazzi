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
amount_of_victims = 8
while len(victims) < amount_of_victims:
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
# --- Parameters ---
grid_res_fine = 10    # meters per cell for fine grid (used in planning)
vis_factor    = 4     # downsampling factor for visualization (e.g. 4 → 4x4 fine cells averaged into 1)

# --- Build fine grid ---
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res_fine),
                     np.arange(min_y, max_y, grid_res_fine))

prob_map_fine = np.ones_like(xv, dtype=float) * 0.1  

spread = 150
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map_fine += np.exp(-(dist/spread)**2)

prob_map_fine /= np.sum(prob_map_fine)  # normalize

# --- Downsample for visualization ---
def downsample_grid(grid, factor):
    h, w = grid.shape
    h2 = h - (h % factor)
    w2 = w - (w % factor)
    grid_cropped = grid[:h2, :w2]
    grid_ds = grid_cropped.reshape(h2//factor, factor, w2//factor, factor).mean(axis=(1,3))
    return grid_ds

prob_map_vis = downsample_grid(prob_map_fine, vis_factor)

grid_res_fine = xv[0,1] - xv[0,0]  # compute from your grid
vis_factor = 2  # downsampling factor for visualization

# # --- Initialize state ---

# victim state
detected_victims = np.zeros(len(victims), dtype=bool)

# UAV state
# --- UAV setup ---
uav_pos = np.array([min_x + (max_x-min_x)/2, min_y + (max_y-min_y)/2])
dt = 0.5
visited_decay = 0.5
detection_radius = 20
target_speed = 19.0  # max cruise speed (m/s)
min_speed = 14.0     # min speed in corners (m/s)
alpha_speed = 0.2    # low-pass smoothing factor for speed
max_turn_rate = np.deg2rad(10)  # max heading change per timestep
look_ahead_distance = 20.0  # meters

# --- Initialize state ---
visited_map = np.zeros_like(prob_map_fine, dtype=bool)
prev_speed = target_speed
prev_heading = 0.0

# --- Path recording ---
smoothed_path = [uav_pos.copy()]
raw_path = [uav_pos.copy()]
look_ahead_points = []

# --- Helper functions ---
def world_to_cell(x, y, min_x, min_y, grid_res):
    i = int((y - min_y) // grid_res)
    j = int((x - min_x) // grid_res)
    i = np.clip(i, 0, prob_map_fine.shape[0]-1)
    j = np.clip(j, 0, prob_map_fine.shape[1]-1)
    return i, j

def get_velocity_gradient(uav_pos, prob_map, min_x, min_y, grid_res, look_ahead=1):
    i, j = world_to_cell(uav_pos[0], uav_pos[1], min_x, min_y, grid_res)
    i_min, i_max = max(0,i-look_ahead), min(prob_map.shape[0], i+look_ahead+1)
    j_min, j_max = max(0,j-look_ahead), min(prob_map.shape[1], j+look_ahead+1)
    
    patch = prob_map[i_min:i_max, j_min:j_max]
    yi, xi = np.meshgrid(np.arange(i_min,i_max), np.arange(j_min,j_max), indexing='ij')
    dx = (xi - j) * grid_res
    dy = (yi - i) * grid_res
    vx = np.sum(patch * dx)
    vy = np.sum(patch * dy)
    return vx, vy, (i,j)

def smooth_heading(current, desired, max_rate):
    delta = (desired - current + np.pi) % (2*np.pi) - np.pi
    delta = np.clip(delta, -max_rate, max_rate)
    return (current + delta) % (2*np.pi)

def downsample_grid(grid, factor):
    h, w = grid.shape
    h2 = h - (h % factor)
    w2 = w - (w % factor)
    grid_cropped = grid[:h2, :w2]
    return grid_cropped.reshape(h2//factor, factor, w2//factor, factor).mean(axis=(1,3))

def recompute_prob_map(victims, detected_victims, xv, yv, base_prob=0.1, spread=150):
    """
    Recompute the smooth probability map based on undetected victims.
    """
    prob_map_new = np.ones_like(xv) * base_prob
    for i, pos in enumerate(victims):
        if not detected_victims[i]:
            dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
            prob_map_new += np.exp(-(dist/spread)**2)
    prob_map_new /= np.sum(prob_map_new)
    return prob_map_new


# --- Simulation loop ---
plt.ion()
fig, ax = plt.subplots(figsize=(10,10))
timesteps = 400

geofence_buffer = 40.0
bounce_steps = 10
bounce_counter = 0


for t in range(timesteps):
    # --- 1️⃣ Raw velocity from probability gradient ---
    vx_raw, vy_raw, cell = get_velocity_gradient(uav_pos, prob_map_fine, min_x, min_y, grid_res_fine, look_ahead=1)
    
    # --- 2️⃣ Look-ahead target ---
    grad_norm = np.hypot(vx_raw, vy_raw)
    if grad_norm > 0:
        dir_x = vx_raw / grad_norm
        dir_y = vy_raw / grad_norm
    else:
        dir_x = dir_y = 0.0
    target_point = uav_pos + look_ahead_distance * np.array([dir_x, dir_y])
    
    delta = target_point - uav_pos
    dist = np.hypot(*delta)
    if dist > 0:
        vx_des = (delta[0] / dist) * target_speed
        vy_des = (delta[1] / dist) * target_speed
    else:
        vx_des = vy_des = 0.0
    
    # --- Record raw path ---
    raw_pos = raw_path[-1] + np.array([vx_des, vy_des]) * dt
    raw_path.append(raw_pos)
    
    # --- 3️⃣ Desired heading ---
    heading_des = np.arctan2(vy_des, vx_des)
    
    # --- 4️⃣ Turn-aware speed scaling ---
    delta_heading = (heading_des - prev_heading + np.pi) % (2*np.pi) - np.pi
    turn_factor = np.clip(1 - abs(delta_heading)/(max_turn_rate*2), 0.0, 1.0)
    speed_des = min_speed + (target_speed - min_speed) * turn_factor
    
    # --- 5️⃣ Heading smoothing ---
    smoothed_heading = smooth_heading(prev_heading, heading_des, max_turn_rate)
    
    # --- 6️⃣ Speed smoothing ---
    smoothed_speed = alpha_speed * speed_des + (1-alpha_speed) * prev_speed
    
    # --- 7️⃣ Smoothed velocity ---
    vx_smoothed = smoothed_speed * np.cos(smoothed_heading)
    vy_smoothed = smoothed_speed * np.sin(smoothed_heading)
    
    # --- 8️⃣ Softgeofence check ---
    next_pos = uav_pos + np.array([vx_smoothed, vy_smoothed]) * dt
    dist_to_boundary = soft_poly.exterior.distance(Point(next_pos))
    
    if bounce_counter > 0:
        # Still in bounce: reverse heading and reduce speed
        vx_smoothed = -vx_smoothed
        vy_smoothed = -vy_smoothed
        smoothed_speed *= 0.5
        bounce_counter -= 1
    elif dist_to_boundary < geofence_buffer:
        print(f"Timestep {t}: Approaching softgeofence, bouncing back!")
        vx_smoothed = -vx_smoothed
        vy_smoothed = -vy_smoothed
        smoothed_speed *= 0.5
        bounce_counter = bounce_steps
    else:
        # Smooth acceleration back to target speed
        smoothed_speed = alpha_speed * target_speed + (1-alpha_speed) * smoothed_speed
    
    # Update smoothed velocity after adjustments
    vx_smoothed = smoothed_speed * np.cos(smoothed_heading)
    vy_smoothed = smoothed_speed * np.sin(smoothed_heading)
    
    # --- 9️⃣ Update UAV position ---
    uav_pos[0] += vx_smoothed * dt
    uav_pos[1] += vy_smoothed * dt
    
    # --- 10️⃣ Victim detection & probability map update ---
    for i, pos in enumerate(victims):
        if not detected_victims[i]:
            if np.linalg.norm(uav_pos - pos) < detection_radius:
                detected_victims[i] = True
                print(f"Timestep {t}: Victim at {pos} detected!")
                prob_map_fine = recompute_prob_map(victims, detected_victims, xv, yv)
    
    # --- 11️⃣ Update visited map ---
    prob_map_fine[cell[0], cell[1]] *= visited_decay
    visited_map[cell[0], cell[1]] = True
    
    # --- 12️⃣ Save state ---
    prev_speed = smoothed_speed
    prev_heading = smoothed_heading
    smoothed_path.append(uav_pos.copy())
    look_ahead_points.append(target_point.copy())
    
    # --- 13️⃣ Print velocities ---
    print(f"Timestep {t}: raw vx = {vx_des:.2f}, vy = {vy_des:.2f}, |v| = {np.hypot(vx_des, vy_des):.2f} | "
          f"smoothed vx = {vx_smoothed:.2f}, vy = {vy_smoothed:.2f}, |v| = {np.hypot(vx_smoothed, vy_smoothed):.2f}")
    
    # --- 14️⃣ Visualization ---
    prob_map_vis = downsample_grid(prob_map_fine, 2)
    ax.clear()
    ax.imshow(prob_map_vis, origin='lower', cmap='coolwarm',
              extent=[min_x, max_x, min_y, max_y], alpha=0.85)
    
    ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')
    smoothed_path_np = np.array(smoothed_path)
    raw_path_np = np.array(raw_path)
    look_ahead_points_np = np.array(look_ahead_points)
    
    ax.plot(smoothed_path_np[:,0], smoothed_path_np[:,1], 'b-', lw=2, label='Smoothed path')
    ax.plot(raw_path_np[:,0], raw_path_np[:,1], 'r--', lw=2, label='Raw path')
    if look_ahead_points_np.size > 0:
        ax.scatter(look_ahead_points_np[:,0], look_ahead_points_np[:,1], c='green', s=20, alpha=0.5, label='Look-ahead targets')
    
    ax.plot(np.append(ehvb_xy[:, 0], ehvb_xy[0, 0]),
            np.append(ehvb_xy[:, 1], ehvb_xy[0, 1]),
            'orange', linewidth=2, label='EHVB / Flyzone')
    ax.plot(np.append(softgeo_xy[:, 0], softgeo_xy[0, 0]),
            np.append(softgeo_xy[:, 1], softgeo_xy[0, 1]),
            'purple', linewidth=2, label='SoftGeofence')
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.set_title(f"Timestep {t}")
    ax.legend()
    plt.pause(0.05)

plt.ioff()
plt.show()

# for t in range(timesteps):
#     # --- 1️⃣ Raw velocity from probability gradient ---
#     vx_raw, vy_raw, cell = get_velocity_gradient(uav_pos, prob_map_fine, min_x, min_y, grid_res_fine, look_ahead=1)
    
#     # --- 2️⃣ Look-ahead target ---
#     grad_norm = np.hypot(vx_raw, vy_raw)
#     if grad_norm > 0:
#         dir_x = vx_raw / grad_norm
#         dir_y = vy_raw / grad_norm
#     else:
#         dir_x = dir_y = 0.0
#     target_point = uav_pos + look_ahead_distance * np.array([dir_x, dir_y])
    
#     delta = target_point - uav_pos
#     dist = np.hypot(*delta)
#     if dist > 0:
#         vx_des = (delta[0] / dist) * target_speed
#         vy_des = (delta[1] / dist) * target_speed
#     else:
#         vx_des = vy_des = 0.0
    
#     # --- Record raw path ---
#     raw_pos = raw_path[-1] + np.array([vx_des, vy_des]) * dt
#     raw_path.append(raw_pos)
    
#     # --- 3️⃣ Desired heading ---
#     heading_des = np.arctan2(vy_des, vx_des)
    
#     # --- 4️⃣ Turn-aware speed scaling ---
#     delta_heading = (heading_des - prev_heading + np.pi) % (2*np.pi) - np.pi
#     turn_factor = np.clip(1 - abs(delta_heading)/(max_turn_rate*2), 0.0, 1.0)
#     speed_des = min_speed + (target_speed - min_speed) * turn_factor
    
#     # --- 5️⃣ Heading smoothing ---
#     smoothed_heading = smooth_heading(prev_heading, heading_des, max_turn_rate)
    
#     # --- 6️⃣ Speed smoothing ---
#     smoothed_speed = alpha_speed * speed_des + (1-alpha_speed) * prev_speed
    
#     # --- 7️⃣ Smoothed velocity ---
#     vx_smoothed = smoothed_speed * np.cos(smoothed_heading)
#     vy_smoothed = smoothed_speed * np.sin(smoothed_heading)
    
#     # --- 8️⃣ Update UAV position ---
#     uav_pos[0] += vx_smoothed * dt
#     uav_pos[1] += vy_smoothed * dt
    
#     # --- 9️⃣ Update probability map for visited cell ---
#     for i, pos in enumerate(victims):
#         if not detected_victims[i]:
#             if np.linalg.norm(uav_pos - pos) < detection_radius:
#                 detected_victims[i] = True
#                 print(f"Timestep {t}: Victim at {pos} detected!")

#                 # recompute full probability map using only undetected victims
#                 prob_map_fine = recompute_prob_map(victims, detected_victims, xv, yv)




#     prob_map_fine[cell[0], cell[1]] *= visited_decay
#     visited_map[cell[0], cell[1]] = True
    
#     # --- 10️⃣ Save state ---
#     prev_speed = smoothed_speed
#     prev_heading = smoothed_heading
#     smoothed_path.append(uav_pos.copy())
#     look_ahead_points.append(target_point.copy())
    
#     # --- 11️⃣ Print velocities ---
#     print(f"Timestep {t}: raw vx = {vx_des:.2f}, vy = {vy_des:.2f}, |v| = {np.hypot(vx_des, vy_des):.2f} | "
#           f"smoothed vx = {vx_smoothed:.2f}, vy = {vy_smoothed:.2f}, |v| = {np.hypot(vx_smoothed, vy_smoothed):.2f}")
    
#     # --- 12️⃣ Visualization ---
#     prob_map_vis = downsample_grid(prob_map_fine, 2)
#     ax.clear()
#     ax.imshow(prob_map_vis, origin='lower', cmap='coolwarm',
#               extent=[min_x, max_x, min_y, max_y], alpha=0.85)
    
#     colors = ['red' if not d else 'gray' for d in detected_victims]
#     ax.scatter(victims[:,0], victims[:,1], c=colors, marker='x', s=80, label='Victims')

    
#     # ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')
#     smoothed_path_np = np.array(smoothed_path)
#     raw_path_np = np.array(raw_path)
#     look_ahead_points_np = np.array(look_ahead_points)
    
#     # Plot smoothed path (actual UAV)
#     ax.plot(smoothed_path_np[:,0], smoothed_path_np[:,1], 'b-', lw=2, label='Smoothed path')
    
#     # Plot raw path (if UAV blindly followed gradient)
#     ax.plot(raw_path_np[:,0], raw_path_np[:,1], 'r--', lw=2, label='Raw path')
    
#     # Plot look-ahead targets
#     ax.scatter(look_ahead_points_np[:,0], look_ahead_points_np[:,1], c='green', s=20, alpha=0.5, label='Look-ahead targets')
    
#     # Areas
#     ax.plot(np.append(ehvb_xy[:, 0], ehvb_xy[0, 0]),
#             np.append(ehvb_xy[:, 1], ehvb_xy[0, 1]),
#             'orange', linewidth=2, label='EHVB / Flyzone')
#     ax.plot(np.append(softgeo_xy[:, 0], softgeo_xy[0, 0]),
#             np.append(softgeo_xy[:, 1], softgeo_xy[0, 1]),
#             'purple', linewidth=2, label='SoftGeofence')
    
#     ax.set_xlim(min_x, max_x)
#     ax.set_ylim(min_y, max_y)
#     ax.set_aspect('equal')
#     ax.set_title(f"Timestep {t}")
#     ax.legend()
#     plt.pause(0.05)

# plt.ioff()
# plt.show()
