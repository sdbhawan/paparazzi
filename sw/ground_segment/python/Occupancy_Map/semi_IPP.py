import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from scipy.special import expit  # stable sigmoid
import pymap3d as pm
import math
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


# --- SETTINGS ---
USE_PPRZ = False
RANDOM_SEED = 112
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Paparazzi setup ---
PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')))
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0
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

softgeo_xy = np.array([waypoints[wp][:2] for wp in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]])
soft_poly = Polygon(softgeo_xy)


# --- Parameters ---
alpha_d = 0.5
dt_step = 1.0          # simulation timestep [s]
FOV_radius = 50.0      # UAV FOV [m]
I_scale = 100.0
E_scale = 100.0

# -----------------------------
# ENERGY MODEL
# -----------------------------
# POWER_HOVER = 920
# POWER_FIXED = 300
# POWER_TRANSITION = 1500

# -----------------------------
# POLYGON + GRID
# -----------------------------

softgeo_xy  = softgeo_xy
soft_poly = Polygon(softgeo_xy)

x_min, y_min, x_max, y_max = soft_poly.bounds
grid_resolution = 10.0
grid_x = np.arange(x_min, x_max, grid_resolution)
grid_y = np.arange(y_min, y_max, grid_resolution)
grid_points = np.array([[x,y] for x in grid_x for y in grid_y])
inside_idx = [i for i,p in enumerate(grid_points) if soft_poly.contains(Point(p))]
belief = 0.5*np.ones(len(grid_points))

# -----------------------------
# VICTIMS
# -----------------------------

victims = np.array([[50,-200],[150,-350]], dtype=float)
for v in victims:
    d2 = np.sum((grid_points - v)**2, axis=1)
    belief[d2 < 20**2] = 0.9  # high occupancy around victims

# -----------------------------
# DRONE INITIAL
# -----------------------------
drone_pos = np.array([10.0,-300.0,50.0])
drone_vel = np.zeros(3)

# -----------------------------
# ENTROPY
# -----------------------------
def cell_entropy_map(belief):
    p = np.clip(belief, 1e-9, 1-1e-9)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

# -----------------------------
# VISIBLE CELLS
# -----------------------------
def visible_cells_at(pos_xy, altitude, grid_points, fov_radius):
    dx = grid_points[:,0]-pos_xy[0]
    dy = grid_points[:,1]-pos_xy[1]
    d2 = dx**2+dy**2
    return d2 <= fov_radius**2


def expected_info_gain(path, belief, grid_points, fov_radius, pred_depth=3):
    """
    Semi-predictive expected information gain along path.
    Computes true entropy reduction ΔH = H_before - H_after,
    only within the UAV's FOV at each step.
    """
    pred_belief = belief.copy()
    total_IG = 0.0
    vis_mask_total = np.zeros(len(grid_points), dtype=bool)

    for i in range(len(path) - 1):
        if i >= pred_depth:
            break

        p0, p1 = np.array(path[i]), np.array(path[i + 1])
        seg_len = np.linalg.norm(p1[:2] - p0[:2])
        n_samples = max(2, int(np.ceil(seg_len / 20.0)))

        xs = np.linspace(p0[0], p1[0], n_samples)
        ys = np.linspace(p0[1], p1[1], n_samples)
        zs = np.linspace(p0[2], p1[2], n_samples)

        for x, y, z in zip(xs, ys, zs):
            vis_mask = visible_cells_at(np.array([x, y]), z, grid_points, fov_radius)

            # --- Entropy before and after (only within FOV) ---
            H_before = cell_entropy_map(pred_belief[vis_mask])

            # Simulate an observation update toward 0.5 baseline
            new_belief = pred_belief.copy()
            new_belief[vis_mask] = new_belief[vis_mask] * 0.7 + 0.3 * 0.5

            H_after = cell_entropy_map(new_belief[vis_mask])

            # --- True information gain in this FOV ---
            IG_step = np.sum(H_before - H_after)
            total_IG += IG_step

            # --- Update predictive belief for next step ---
            pred_belief = new_belief
            vis_mask_total |= vis_mask

    return total_IG, vis_mask_total


# -----------------------------
# ENERGY (simple proxy)
# -----------------------------
def energy_of_path(path):
    """
    Compute energy along path segment (hover/fixed transition)
    """
    p0, p1 = path[0], path[-1]
    dist = np.linalg.norm(p1[:2]-p0[:2])
    dz = abs(p1[2]-p0[2])
    speed = dist / 1.0  # dt_step
    if speed < 12: mode = "hover"
    elif speed < 17: mode = "transition"
    else: mode = "fixedwing"
    if mode=="hover": E = 920.0
    elif mode=="transition": E = 1500.0
    else: E = 300.0
    E += 5.0 * dz
    return E

# -----------------------------
# PLANNER
# -----------------------------
def plan_velocity_ipp(drone_pos, belief, grid_points, soft_poly, fov_radius,
                      v_max=12.0, n_directions=16, step_length=40.0,
                      I_scale=100.0, E_scale=100.0, alpha_d=0.5,
                      buffer=0.0, pred_depth=3):
    """
    Semi-predictive planner: returns desired vx,vy,vz along best path.
    """
    # shrink polygon by buffer
    if buffer>0.0:
        poly = Polygon(soft_poly).buffer(-buffer)
    else:
        poly = Polygon(soft_poly)

    # generate candidate directions
    cx, cy, cz = drone_pos
    angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    candidates = []
    for a in angles:
        dx, dy = np.cos(a), np.sin(a)
        end = np.array([cx + dx*step_length, cy + dy*step_length, cz])
        if poly.contains(Point(end[0], end[1])):
            candidates.append([np.array([cx,cy,cz]), end])

    # evaluate semi-predictive objective
    best_J, best_path, best_mask = -np.inf, None, None
    for path in candidates:
        I_p, vis_mask = expected_info_gain(path, belief, grid_points, fov_radius, pred_depth)
        E_p = energy_of_path(path)
        J = I_p/I_scale - alpha_d*E_p/E_scale
        if J > best_J:
            best_J = J
            best_path = path
            best_mask = vis_mask

    if best_path is None:
        return 0.0,0.0,0.0,None,None,0.0

    # compute velocity along path
    p0, p1 = best_path[0], best_path[-1]
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    if dist<1e-3: 
        return 0.0,0.0,0.0,best_path,best_mask,best_J
    
    travel_fraction = min(1.0, 1.0 / max(dist/v_max,1e-6))  # dt_step = 1.0
    step_vec = vec * travel_fraction
    vx, vy, vz = step_vec / 1.0
    return float(vx), float(vy), float(vz), best_path, best_mask, best_J


grid_points_all = np.array([[x, y] for x in grid_x for y in grid_y])
inside_idx = np.array([soft_poly.contains(Point(p)) for p in grid_points_all])  # bool array
grid_points = grid_points_all[inside_idx]  # points inside polygon
belief = 0.5 * np.ones(np.sum(inside_idx))  # occupancy only for softgeofence


# Precompute victim signal on full grid
victim_signal_full = np.zeros(len(grid_points_all))
for v in victims:
    d2 = np.sum((grid_points_all - v[:2])**2, axis=1)
    victim_signal_full[d2 <= FOV_radius**2] = 1.0

# -----------------------------
# SIMULATION PARAMETERS
# -----------------------------
v_max = 20.0        # max velocity [m/s]
a_max = 2.0         # max acceleration [m/s²]

# -----------------------------
# PLOT SETUP
# -----------------------------
fig, ax = plt.subplots(figsize=(8,8))
cmap = plt.cm.get_cmap('RdYlBu_r')

# Raster belief placeholder (full grid)
raster_belief = np.full_like(grid_points_all[:,0], np.nan, dtype=float)
raster_belief[inside_idx] = belief

# Scatter plot
sc = ax.scatter(
    grid_points_all[:,0], grid_points_all[:,1],
    c=raster_belief, cmap=cmap, s=20,
    vmin=0.0, vmax=1.0
)

# UAV
drone_plot, = ax.plot(drone_pos[0], drone_pos[1], 'ko', markersize=8)

# Victims
if len(victims) > 0:
    ax.plot(victims[:,0], victims[:,1], 'rx', markersize=8)

# Colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Occupancy probability')

plt.xlim(x_min-10, x_max+10)
plt.ylim(y_min-10, y_max+10)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Occupancy Map with UAV and Victims')
plt.ion()
plt.show()

# -----------------------------
# SIMULATION LOOP
# -----------------------------
max_dheading = np.deg2rad(10)  # max heading change per second
buffer =  FOV_radius 

# Create a buffered polygon for safe navigation: ensures that if we detect that there are way less visible cells, 
# the UAV will turn such that it can sense more cells again

if buffer > 0:
    safe_poly = soft_poly.buffer(-buffer)
else:
    safe_poly = soft_poly


#for loop single UAV without drift:

# for t in range(300):
#     # -----------------------------
#     # 1. Update observed cells
#     # -----------------------------
#     vis_mask_full = visible_cells_at(drone_pos[:2], drone_pos[2], grid_points_all, FOV_radius)
#     vis_mask = vis_mask_full[inside_idx]
#     victim_signal = victim_signal_full[inside_idx]

#     belief[vis_mask] += 0.3 * (victim_signal[vis_mask] - belief[vis_mask])
#     belief = np.clip(belief, 0, 1)

#     # -----------------------------
#     # 2. Plan next velocity
#     # -----------------------------
#     vx_des, vy_des, vz_des, best_path, best_mask, Jval = plan_velocity_ipp(
#         drone_pos, belief, grid_points, soft_poly, FOV_radius,
#         v_max=v_max, n_directions=16, step_length=40.0,
#         I_scale=I_scale, E_scale=E_scale, alpha_d=alpha_d,
#         buffer=buffer
#     )

#     # -----------------------------
#     # 3. Boundary correction
#     # -----------------------------
#     next_pos = drone_pos + np.array([vx_des, vy_des, vz_des]) * dt_step
#     point_next = Point(next_pos[0], next_pos[1])
#     if not safe_poly.contains(point_next):
#         # Project movement along the closest point on the safe polygon
#         nearest = np.array(safe_poly.exterior.interpolate(safe_poly.exterior.project(point_next)).coords[0])
#         direction = nearest - drone_pos[:2]
#         norm = np.linalg.norm(direction)
#         if norm > 1e-3:
#             vx_des, vy_des = direction / norm * min(norm/dt_step, v_max)
#         else:
#             vx_des, vy_des = 0.0, 0.0

#     # -----------------------------
#     # 4. Smooth UAV heading
#     # -----------------------------
#     speed = np.linalg.norm([vx_des, vy_des])
#     if speed < 1e-3:
#         vx_smooth, vy_smooth = 0.0, 0.0
#     else:
#         current_heading = np.arctan2(drone_vel[1], drone_vel[0])
#         desired_heading = np.arctan2(vy_des, vx_des)
#         delta_heading = (desired_heading - current_heading + np.pi) % (2*np.pi) - np.pi
#         delta_heading = np.clip(delta_heading, -max_dheading*dt_step, max_dheading*dt_step)
#         new_heading = current_heading + delta_heading
#         vx_smooth = speed * np.cos(new_heading)
#         vy_smooth = speed * np.sin(new_heading)

#     drone_vel[:2] = [vx_smooth, vy_smooth]
#     drone_vel[2] = vz_des
#     drone_pos += drone_vel * dt_step

#     # -----------------------------
#     # 5. Update visualization
#     # -----------------------------
#     raster_belief[inside_idx] = belief
#     sc.set_array(raster_belief)
#     drone_plot.set_data(drone_pos[0], drone_pos[1])
#     plt.pause(0.01)

#     # -----------------------------
#     # 6. Print status
#     # -----------------------------
#     mean_H = np.mean(cell_entropy_map(belief))
#     heading_deg = math.degrees(math.atan2(vy_smooth, vx_smooth))
#     print(f"t={t:03d}s pos=({drone_pos[0]:.1f},{drone_pos[1]:.1f},{drone_pos[2]:.1f}) "
#           f"vx={vx_smooth:.2f} vy={vy_smooth:.2f} vz={vz_des:.2f} heading={heading_deg:.2f} "
#           f"vis_cells={np.sum(vis_mask)} mean_H={mean_H:.3f} J={Jval:.3f}")


# plt.ioff()
# plt.show()

# -----------------------------
# SIMULATION LOOP WITH MOVING VICTIMS
# -----------------------------
v_drift = np.array([0.5, 0.2])  # drift velocity of victims [m/s] in XY


for t in range(300):
    # -----------------------------
    # 1. Update victim positions (drift)
    # -----------------------------
    v_drift = np.array([0.5, 0.2])  # simple linear drift per second
    victims[:, :2] += v_drift * dt_step

    # -----------------------------
    # 2. Update observed cells
    # -----------------------------
    vis_mask_full = visible_cells_at(drone_pos[:2], drone_pos[2], grid_points_all, FOV_radius)
    vis_mask = vis_mask_full[inside_idx]

    # recompute victim signal
    victim_signal_full = np.zeros(len(grid_points_all))
    for v in victims:
        d2 = np.sum((grid_points_all - v[:2])**2, axis=1)
        victim_signal_full[d2 <= FOV_radius**2] = 1.0
    victim_signal = victim_signal_full[inside_idx]

    # belief update
    belief[vis_mask] += 0.3 * (victim_signal[vis_mask] - belief[vis_mask])
    belief = np.clip(belief, 0, 1)

    # -----------------------------
    # 3. Plan next velocity
    # -----------------------------
    vx_des, vy_des, vz_des, best_path, best_mask, Jval = plan_velocity_ipp(
        drone_pos, belief, grid_points, soft_poly, FOV_radius,
        v_max=v_max, n_directions=16, step_length=40.0,
        I_scale=I_scale, E_scale=E_scale, alpha_d=alpha_d,
        buffer=buffer
    )

    # fallback if no path found
    if best_path is None:
        vx_des, vy_des, vz_des = 0.5, 0.0, 0.0  # small default motion

    # -----------------------------
    # 4. Smooth UAV heading
    # -----------------------------
    speed = np.linalg.norm([vx_des, vy_des])
    if speed < 1e-3:
        vx_smooth, vy_smooth = 0.0, 0.0
    else:
        current_heading = np.arctan2(drone_vel[1], drone_vel[0])
        desired_heading = np.arctan2(vy_des, vx_des)
        delta_heading = (desired_heading - current_heading + np.pi) % (2*np.pi) - np.pi
        delta_heading = np.clip(delta_heading, -max_dheading*dt_step, max_dheading*dt_step)
        new_heading = current_heading + delta_heading
        vx_smooth = speed * np.cos(new_heading)
        vy_smooth = speed * np.sin(new_heading)

    drone_vel[:2] = [vx_smooth, vy_smooth]
    drone_vel[2] = vz_des

    # -----------------------------
    # 5. Boundary correction
    # -----------------------------
    next_pos = drone_pos + drone_vel * dt_step
    point_next = Point(next_pos[0], next_pos[1])
    if not safe_poly.contains(point_next):
        # project onto closest point on safe polygon
        nearest = np.array(safe_poly.exterior.interpolate(safe_poly.exterior.project(point_next)).coords[0])
        direction = nearest - drone_pos[:2]
        norm = np.linalg.norm(direction)
        if norm > 1e-3:
            vx_smooth, vy_smooth = direction / norm * min(norm/dt_step, v_max)
        else:
            vx_smooth, vy_smooth = 0.0, 0.0
        drone_vel[:2] = [vx_smooth, vy_smooth]

    # -----------------------------
    # 6. Advance UAV
    # -----------------------------
    drone_pos += drone_vel * dt_step

    # -----------------------------
    # 7. Update visualization
    # -----------------------------
    raster_belief[inside_idx] = belief
    sc.set_array(raster_belief)
    drone_plot.set_data(drone_pos[0], drone_pos[1])
    if len(victims) > 0:
        ax.plot(victims[:,0], victims[:,1], 'rx', markersize=8)
    plt.pause(0.01)

    # -----------------------------
    # 8. Print status
    # -----------------------------
    mean_H = np.mean(cell_entropy_map(belief))
    heading_deg = math.degrees(np.arctan2(vy_smooth, vx_smooth))
    print(f"t={t:03d}s pos=({drone_pos[0]:.1f},{drone_pos[1]:.1f},{drone_pos[2]:.1f}) "
          f"vx={vx_smooth:.2f} vy={vy_smooth:.2f} vz={vz_des:.2f} heading={heading_deg:.2f} "
          f"vis_cells={np.sum(vis_mask)} mean_H={mean_H:.3f} J={Jval:.3f}")

plt.ioff()
plt.show()