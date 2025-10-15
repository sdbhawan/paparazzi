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


# add a constant z for 3D computations
altitude = 50.0  # flight altitude
grid_points_3d = np.hstack([grid_points, np.full((grid_points.shape[0], 1), altitude)])
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


# def plan_velocity_ipp(drone_pos, belief, grid_points, soft_poly, fov_radius,
#                       v_max=12.0, n_directions=16, step_length=40.0,
#                       I_scale=100.0, E_scale=100.0, alpha_d=0.5,
#                       buffer=0.0, horizon_depth=3):
#     """
#     Receding-horizon IPP planner: returns desired vx,vy,vz along the best path.
#     Computes expected entropy reduction (ΔH) over the horizon.
#     """
#     # shrink polygon by buffer
#     if buffer > 0.0:
#         poly = Polygon(soft_poly).buffer(-buffer)
#     else:
#         poly = Polygon(soft_poly)

#     # generate candidate directions
#     cx, cy, cz = drone_pos
#     angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
#     candidates = []
#     for a in angles:
#         dx, dy = np.cos(a), np.sin(a)
#         end = np.array([cx + dx*step_length, cy + dy*step_length, cz])
#         if poly.contains(Point(end[0], end[1])):
#             candidates.append([np.array([cx, cy, cz]), end])

#     # evaluate horizon objective
#     best_J, best_path, best_mask = -np.inf, None, None
#     for path in candidates:
#         I_p, vis_mask = expected_info_gain(path, belief, grid_points, fov_radius, horizon_depth)
#         E_p = energy_of_path(path)
#         J = I_p/I_scale - alpha_d*E_p/E_scale
#         if J > best_J:
#             best_J = J
#             best_path = path
#             best_mask = vis_mask

#     if best_path is None:
#         return 0.0, 0.0, 0.0, None, None, 0.0

#     # compute velocity along first step of horizon
#     p0, p1 = best_path[0], best_path[-1]
#     vec = p1 - p0
#     dist = np.linalg.norm(vec)
#     if dist < 1e-3:
#         return 0.0, 0.0, 0.0, best_path, best_mask, best_J

#     travel_fraction = min(1.0, 1.0 / max(dist/v_max, 1e-6))  # dt_step = 1.0
#     step_vec = vec * travel_fraction
#     vx, vy, vz = step_vec / 1.0
#     return float(vx), float(vy), float(vz), best_path, best_mask, best_J


def propagate_belief_vectorized(belief, grid_points, v_drift, dt, sigma=5.0):
    """
    Vectorized belief propagation with drift + diffusion.

    Parameters
    ----------
    belief : np.array, shape (N,)
        Current occupancy probabilities.
    grid_points : np.array, shape (N,2)
        XY coordinates of grid cells.
    v_drift : np.array, shape (2,)
        Drift velocity (vx, vy) in m/s.
    dt : float
        Time step in seconds.
    sigma : float
        Standard deviation for diffusion kernel (in grid units).

    Returns
    -------
    new_belief : np.array, shape (N,)
        Predicted belief after drift and diffusion.
    """
    # Step 1: Compute previous locations of each grid cell
    # (move current grid points backward along drift)
    prev_positions = grid_points - v_drift * dt

    # Step 2: Find nearest neighbor indices in original grid
    tree = cKDTree(grid_points)
    _, idx = tree.query(prev_positions)

    # Step 3: Assign belief from nearest neighbor
    new_belief = belief[idx]

    # Step 4: Optional: smooth / diffuse
    # Map belief into a 2D grid for Gaussian filter
    x_coords = np.unique(grid_points[:,0])
    y_coords = np.unique(grid_points[:,1])
    x_idx = np.searchsorted(x_coords, grid_points[:,0])
    y_idx = np.searchsorted(y_coords, grid_points[:,1])

    grid_2d = np.full((len(x_coords), len(y_coords)), np.nan)
    grid_2d[x_idx, y_idx] = new_belief

    # Apply Gaussian filter; nan handling
    mask = np.isnan(grid_2d)
    grid_2d[mask] = 0.5  # neutral probability for NaNs
    grid_2d = gaussian_filter(grid_2d, sigma=sigma)
    grid_2d = np.clip(grid_2d, 0.0, 1.0)

    # Step 5: Map back to 1D belief array
    new_belief = grid_2d[x_idx, y_idx]

    return new_belief

def velocity_to_follow_victim(drone_pos, victim_pos, victim_vel, max_speed=12.0, close_thresh=1.0):
    """
    Robust 3D follow velocity. Accepts 2D or 3D victim_pos.
    """
    drone = np.array(drone_pos, dtype=float)
    victim = np.array(victim_pos, dtype=float)
    if victim.shape[0] == 2:
        victim = np.array([victim[0], victim[1], drone[2]])  # assume same altitude

    vec = victim - drone
    horiz = vec[:2]
    dist = np.linalg.norm(horiz)
    if dist < close_thresh:
        vx, vy = float(victim_vel[0]), float(victim_vel[1])
        vz = 0.0
        s = np.linalg.norm([vx, vy])
        if s > max_speed:
            vx, vy = (np.array([vx, vy]) / s * max_speed).tolist()
        return float(vx), float(vy), float(vz)

    dir_xy = horiz / (dist + 1e-9)
    target_speed = min(max_speed, dist)
    vx = dir_xy[0] * target_speed + victim_vel[0]
    vy = dir_xy[1] * target_speed + victim_vel[1]
    vz = 0.0
    return float(vx), float(vy), float(vz)


# --- Sensor parameters ---
# -----------------------------
# SENSOR MODEL PARAMETERS
# -----------------------------
theta_FOV = np.deg2rad(60)   # sensor opening angle [rad]
h_ref = 50.0                 # reference altitude for full resolution [m]
P0 = 1.0                     # base detection probability

def f_res(h):
    """Resolution factor as a function of UAV altitude."""
    return min(1.0, h_ref / h)

def r_FOV(h):
    """Ground-projected sensor footprint radius."""
    return h * np.tan(theta_FOV)

def P_hit(h):
    """
    Detection probability as a function of altitude h
    """
    f_res = min(1.0, h_ref / h)
    return P0 * f_res


# -----------------------------
# VISIBLE CELLS WITH ALTITUDE-DEPENDENT DETECTION
# -----------------------------
def visible_cells_at(pos_xyz, grid_points, fov_angle=theta_FOV):
    """
    Returns boolean mask of cells inside FOV and their detection probabilities.
    pos_xyz: UAV position [x,y,z]
    grid_points: array of grid points [[x,y], ...]
    """
    x, y, z = pos_xyz
    r = r_FOV(z)
    dx = grid_points[:,0] - x
    dy = grid_points[:,1] - y
    d2 = dx**2 + dy**2
    mask = d2 <= r**2
    p_hit = np.zeros(len(grid_points))
    p_hit[mask] = P_hit(z)
    return mask, p_hit

# -----------------------------
# EXPECTED INFO GAIN (ALTITUDE-AWARE)
# -----------------------------
def expected_info_gain(path, belief, grid_points, fov_angle=theta_FOV, pred_depth=3):
    """
    Compute altitude-aware expected information gain along path.
    """
    pred_belief = belief.copy()
    total_IG = 0.0
    vis_mask_total = np.zeros(len(grid_points), dtype=bool)
    
    ent = cell_entropy_map(pred_belief)
    
    for i in range(len(path)-1):
        if i >= pred_depth:
            break
        p0, p1 = path[i], path[i+1]
        seg_len = np.linalg.norm(p1[:2]-p0[:2])
        n_samples = max(2, int(np.ceil(seg_len / 20.0)))
        xs = np.linspace(p0[0], p1[0], n_samples)
        ys = np.linspace(p0[1], p1[1], n_samples)
        zs = np.linspace(p0[2], p1[2], n_samples)
        
        for x, y, z in zip(xs, ys, zs):
            mask, p_hit = visible_cells_at(np.array([x,y,z]), grid_points)
            # ΔH weighted by detection probability
            delta_H = ent[mask] * p_hit[mask]
            total_IG += np.sum(delta_H)
            
            # simulate belief update
            pred_belief[mask] = pred_belief[mask] * (1 - p_hit[mask]) + 0.5 * p_hit[mask]
            vis_mask_total |= mask
        
        ent = cell_entropy_map(pred_belief)
    
    return total_IG, vis_mask_total

def plan_velocity_ipp_3D(drone_pos, belief, grid_points, soft_poly, 
                         fov_angle=theta_FOV, v_max=12.0, n_directions=16, 
                         step_length=40.0, altitude_candidates=[30,50,70],
                         I_scale=100.0, E_scale=100.0, alpha_d=0.5, buffer=0.0,
                         pred_depth=3):
    """
    Receding-horizon 3D IPP planner.
    Evaluates candidate paths in XY directions and multiple altitudes.
    Returns desired velocity (vx,vy,vz) along best path.
    """
    # shrink polygon by buffer
    if buffer>0.0:
        poly = Polygon(soft_poly).buffer(-buffer)
    else:
        poly = Polygon(soft_poly)
    
    cx, cy, cz = drone_pos
    angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    candidates = []
    
    # generate candidate paths with altitudes
    for a in angles:
        dx, dy = np.cos(a), np.sin(a)
        for alt in altitude_candidates:
            end = np.array([cx + dx*step_length, cy + dy*step_length, alt])
            if poly.contains(Point(end[0], end[1])):
                candidates.append([np.array([cx, cy, cz]), end])
    
    best_J, best_path, best_mask = -np.inf, None, None
    
    for path in candidates:
        I_p, vis_mask = expected_info_gain(path, belief, grid_points, fov_angle=fov_angle, pred_depth=pred_depth)
        E_p = energy_of_path(path)  # energy already works in 3D
        J = I_p/I_scale - alpha_d*E_p/E_scale
        if J > best_J:
            best_J = J
            best_path = path
            best_mask = vis_mask
    
    if best_path is None:
        return 0.0,0.0,0.0,None,None,0.0
    
    # compute velocity along best path
    p0, p1 = best_path[0], best_path[-1]
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    if dist < 1e-3:
        return 0.0, 0.0, 0.0, best_path, best_mask, best_J
    
    travel_fraction = min(1.0, 1.0 / max(dist/v_max,1e-6))
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
# MULTI-UAV SETUP
# -----------------------------
num_drones = 2
drone_positions = [np.array([400.0, 0.0, 50.0]),
                   np.array([50.0, -300.0, 50.0])]
drone_vels = [np.zeros(3) for _ in range(num_drones)]
drone_modes = ['explore' for _ in range(num_drones)]
tracking_timers = [0.0 for _ in range(num_drones)]
tracking_duration = 20.0  # seconds for tracking a victim
pred_depth = 3
victim_thresh = 0.7
assigned_victim_idx = [None] * num_drones
assigned_victim_pos = [None] * num_drones
assigned_victim_vel = [None] * num_drones


# -----------------------------
# PLOT SETUP
# -----------------------------
fig, ax = plt.subplots(figsize=(8,8))
cmap = plt.cm.get_cmap('RdYlBu_r')

# Raster belief placeholder (full grid)
raster_belief = np.full_like(grid_points_all[:,0], np.nan, dtype=float)
raster_belief[inside_idx] = belief

## BEFORE loop: persistent plot handles
sc = ax.scatter(grid_points_all[:,0], grid_points_all[:,1],
                c=raster_belief, cmap=cmap, s=20, vmin=0.0, vmax=1.0)

drone_plots = []
for d_idx in range(num_drones):
    p, = ax.plot(drone_positions[d_idx][0], drone_positions[d_idx][1], 'o', markersize=8)
    drone_plots.append(p)

# victim scatter handle
victim_plot = ax.scatter(victims[:,0], victims[:,1], c='r', marker='x', s=60)


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

# -----------------------------
# SIMULATION LOOP WITH MOVING VICTIMS
# -----------------------------
v_drift = np.array([0.5, 0.2])  # drift velocity of victims [m/s] in XY



# For simplicity, victim velocity (constant drift)
victim_vel = np.array([0.5, 0.2, 0.0])  # m/s

# -----------------------------
# MAIN SIMULATION LOOP
# -----------------------------
# for t in range(300):
#     # --- Drift victims ---

#     victims[:, :2] += v_drift * dt_step
#     # Recompute victim_signal_full from current victim locations
#     victim_signal_full[:] = 0.0
#     for v in victims:
#         d2 = np.sum((grid_points_all - v[:2])**2, axis=1)
#         victim_signal_full[d2 <= FOV_radius**2] = 1.0
#     # debug
#     print(f"t={t:03d}s victims[0]: {victims[0]}")

#     # --- Update belief from observed cells ---
#     for d_idx in range(num_drones):
#         h = drone_positions[d_idx][2]
#         r_fov = h * np.tan(theta_FOV)

#         vis_mask_full, p_hit = visible_cells_at(drone_positions[d_idx], grid_points_all)
#         vis_mask = vis_mask_full[inside_idx]

#         f_res = min(1.0, h_ref / h)
#         victim_signal = victim_signal_full[inside_idx] * f_res

#         belief[vis_mask] += 0.3 * (victim_signal[vis_mask] - belief[vis_mask])
#         belief = np.clip(belief, 0, 1)


#     # --- Check for high-probability victim cells ---
#     detected_cells_idx = np.where(belief >= victim_thresh)[0]
#     victim_assigned = None
#     if len(detected_cells_idx) > 0:
#         # pick a representative victim location (centroid or first cell)
#         victim_cell_xy = grid_points[detected_cells_idx[0]]
#         # find closest true victim (optional) or treat victim_cell as location to reach
#         # Compute energies
#         energies = []
#         for d_idx in range(num_drones):
#             if drone_modes[d_idx] == 'explore':
#                 # ensure a 3D path
#                 path_end = np.array([victim_cell_xy[0], victim_cell_xy[1], drone_positions[d_idx][2]])
#                 path = [drone_positions[d_idx], path_end]
#                 energies.append(energy_of_path(path))
#             else:
#                 energies.append(np.inf)
#         assigned_drone_idx = int(np.argmin(energies))
#         if energies[assigned_drone_idx] < np.inf:
#             # assign that drone to track the *actual* nearest true victim (optional)
#             # For simplicity, store the nearest true victim index:
#             # find actual victim index whose xy is closest to victim_cell_xy
#             dists = np.linalg.norm(victims[:,:2] - victim_cell_xy, axis=1)
#             true_vidx = int(np.argmin(dists))
#             assigned_victim_idx[assigned_drone_idx] = true_vidx
#             assigned_victim_pos[assigned_drone_idx] = victims[true_vidx].copy()
#             assigned_victim_vel[assigned_drone_idx] = victim_vel.copy()
#             drone_modes[assigned_drone_idx] = 'track'
#             tracking_timers[assigned_drone_idx] = 0.0
#             victim_assigned = assigned_drone_idx


#     # --- Update UAV positions ---
#     for d_idx in range(num_drones):
#         print(f"Drone {d_idx} mode={drone_modes[d_idx]} v_idx={assigned_victim_idx[d_idx]}")
#         if drone_modes[d_idx] == 'track':
#             v_idx = assigned_victim_idx[d_idx]
#             if v_idx is None:
#                 drone_modes[d_idx] = 'explore'
#                 continue

#             victim_pos = assigned_victim_pos[d_idx]
#             victim_v = assigned_victim_vel[d_idx]
#             victim_pos_3d = np.array([victim_pos[0], victim_pos[1], drone_positions[d_idx][2]])

#             vx, vy, vz = velocity_to_follow_victim(drone_positions[d_idx], victim_pos_3d, victim_v)
#             drone_vels[d_idx] = np.array([vx, vy, vz])
#             drone_positions[d_idx] += drone_vels[d_idx] * dt_step

#             assigned_victim_pos[d_idx] = victims[v_idx].copy()

#             tracking_timers[d_idx] += dt_step
#             if tracking_timers[d_idx] >= tracking_duration:
#                 drone_modes[d_idx] = 'explore'
#                 print(f"[{t:03d}s] Drone {d_idx} confirmed victim at {victim_pos}")

#             print(f"[{t:03d}s] Drone {d_idx} TRACK vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")
        
#         else:
#             # --- 3D Receding-horizon IPP ---
#             vx, vy, vz, _, _, _ = plan_velocity_ipp_3D(
#                 drone_positions[d_idx], belief, grid_points, soft_poly,
#                 fov_angle=theta_FOV, v_max=v_max, n_directions=16,
#                 step_length=40.0, altitude_candidates=[30, 50, 70],
#                 I_scale=I_scale, E_scale=E_scale, alpha_d=alpha_d,
#                 buffer=buffer, pred_depth=pred_depth
#             )
#             drone_vels[d_idx] = np.array([vx, vy, vz])
#             drone_positions[d_idx] += drone_vels[d_idx] * dt_step

#             print(f"[{t:03d}s] Drone {d_idx} EXPLORE vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")


#     # --- Update visualization ---
#     # update belief raster
#     raster_belief[inside_idx] = belief
#     sc.set_array(raster_belief)

#     # update drone markers
#     for d_idx in range(num_drones):
#         drone_plots[d_idx].set_data(drone_positions[d_idx][0], drone_positions[d_idx][1])

#     # update victim markers
#     if len(victims) > 0:
#         victim_plot.set_offsets(victims[:, :2])
#     plt.pause(0.01)

#     # --- Print status ---
#     mean_H = np.mean(cell_entropy_map(belief))
#     print(
#         f"t={t:03d}s mean_H={mean_H:.3f} "
#         f"detected_cells={len(detected_cells_idx)} "
#         f"assigned={victim_assigned}"
#     )

# plt.ioff()
# plt.show()

# -----------------------------
# SIMULATION LOOP
# -----------------------------
victim_status = np.zeros(len(victims), dtype=int)  # 0=untracked, 1=being tracked, 2=already tracked
tracking_radius = 15.0  # distance to mark victim as tracked
track_threshold = 0.6   # belief threshold to switch from exploration to tracking

for t in range(300):
    # --- Drift victims ---
    victims[:, :2] += v_drift * dt_step

    # Recompute victim signal on full grid
    victim_signal_full[:] = 0.0
    for v in victims:
        d2 = np.sum((grid_points_all - v[:2])**2, axis=1)
        victim_signal_full[d2 <= FOV_radius**2] = 1.0

    # --- Update belief from observed cells ---
    for d_idx in range(num_drones):
        h = drone_positions[d_idx][2]
        vis_mask_full, p_hit = visible_cells_at(drone_positions[d_idx], grid_points_all)
        vis_mask = vis_mask_full[inside_idx]  # restrict to inside polygon

        f_res = min(1.0, h_ref / h)
        victim_signal = victim_signal_full[inside_idx] * f_res

        # Bayesian-ish update
        belief[vis_mask] += 0.3 * (victim_signal[vis_mask] - belief[vis_mask])
        belief = np.clip(belief, 0.0, 1.0)

    # --- Main UAV loop ---
    for d_idx in range(num_drones):

        # --- Decide mode ---
        if drone_modes[d_idx] == 'explore':
            # check if there is a high-belief cell corresponding to an untracked victim
            candidate_victims = []
            candidate_cells_idx = np.where(belief >= track_threshold)[0]

            for cell_idx in candidate_cells_idx:
                dists = np.linalg.norm(victims[:, :2] - grid_points[cell_idx], axis=1)
                for v_idx, dist_to_cell in enumerate(dists):
                    if victim_status[v_idx] == 0:  # untracked
                        candidate_victims.append((v_idx, dist_to_cell))

            # Assign closest candidate victim
            if candidate_victims:
                v_idx = min(candidate_victims, key=lambda x: x[1])[0]
                assigned_victim_idx[d_idx] = v_idx
                assigned_victim_pos[d_idx] = victims[v_idx].copy()
                assigned_victim_vel[d_idx] = victim_vel.copy()
                drone_modes[d_idx] = 'track'
            else:
                assigned_victim_idx[d_idx] = None

        # --- Tracking mode ---
        if drone_modes[d_idx] == 'track':
            v_idx = assigned_victim_idx[d_idx]
            if v_idx is None or victim_status[v_idx] != 0:
                # fallback to exploration if already tracked
                drone_modes[d_idx] = 'explore'
                assigned_victim_idx[d_idx] = None
            else:
                vx, vy, vz = velocity_to_follow_victim(
                    drone_positions[d_idx], assigned_victim_pos[d_idx], assigned_victim_vel[d_idx]
                )
                drone_vels[d_idx] = np.array([vx, vy, vz])
                drone_positions[d_idx] += drone_vels[d_idx] * dt_step

                # mark victim tracked if close enough
                if np.linalg.norm(drone_positions[d_idx][:2] - assigned_victim_pos[d_idx][:2]) < tracking_radius:
                    victim_status[v_idx] = 1  # being tracked
                    drone_modes[d_idx] = 'explore'
                    assigned_victim_idx[d_idx] = None
                continue  # skip exploration for this UAV

        # --- Exploration mode ---
        vx, vy, vz, _, _, _ = plan_velocity_ipp_3D(
            drone_positions[d_idx], belief, grid_points, soft_poly,
            fov_angle=theta_FOV, v_max=v_max, n_directions=16,
            step_length=40.0, altitude_candidates=[30,50,70],
            I_scale=I_scale, E_scale=E_scale, alpha_d=alpha_d,
            buffer=buffer, pred_depth=pred_depth
        )
        drone_vels[d_idx] = np.array([vx, vy, vz])
        drone_positions[d_idx] += drone_vels[d_idx] * dt_step

    # --- Update belief map visualization ---
    raster_belief[inside_idx] = belief
    sc.set_array(raster_belief)

    # update drone markers
    for d_idx in range(num_drones):
        drone_plots[d_idx].set_data(drone_positions[d_idx][0], drone_positions[d_idx][1])

    # update victim markers
    victim_plot.set_offsets(victims[:, :2])
    plt.pause(0.01)

    # --- Print status ---
    mean_H = np.mean(cell_entropy_map(belief))
    print(f"t={t:03d}s mean_H={mean_H:.3f} tracked={np.sum(victim_status)}")

plt.ioff()
plt.show()
