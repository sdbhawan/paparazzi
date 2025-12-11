import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
import pymap3d as pm
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from shapely.affinity import rotate, translate

# --- SETTINGS ---
USE_PPRZ = False
RANDOM_SEED = 112
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

x_min, y_min, x_max, y_max = soft_poly.bounds
grid_resolution = 10.0
grid_x = np.arange(x_min, x_max + grid_resolution, grid_resolution)
grid_y = np.arange(y_min, y_max + grid_resolution, grid_resolution)
cell_size = float(grid_x[1] - grid_x[0])


# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------

dt_step = 1.0
FOV_radius = 30.0 #also a function using this, but we still use this parameter???
v_drift = np.array([1.7, 1.0])
theta_FOV = np.deg2rad(45)
I_scale = 100.0
E_scale = 100.0
alpha_d = 0.4
pconf = 0.6
pth_detect = 0.77
rho_th = 0.2
gamma_wind = 5.0
v_wind = v_drift.copy()             # treat wind ≈ drift for Deff (7.46)
uav_nominal_altitudes = np.array([80.0, 40.0])  # UAV0 = high-alt, UAV1 = low-alt

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def cell_entropy_map(belief):
    p = np.clip(belief, 1e-9, 1 - 1e-9)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# --- Sensor model parameters (sec. 7.3.1) ---
h_ref = 20.0  # [m] altitude at which sensor reaches max resolution (confirmation height)
P0 = 0.9
P_false = 1e-5        # example

def f_res(h):
    """
    Resolution factor (Sec. 7.3.1).
    Use inverse-square scaling up to h_ref, then saturate at 1:

        f_res(h) = min(1, (h_ref / h)^2)

    So:
      - at h = h_ref:  f_res = 1
      - at h > h_ref:  f_res < 1  (worse resolution, lower P_hit)
      - at h < h_ref:  f_res = 1  (no >100% "super sensor")
    """
    h_eff = max(h, 1e-3)
    return min(1.0, (h_ref / h_eff))


def r_FOV(h):
    """Ground-projected FOV radius: h * tan(theta_FOV)."""
    return h * np.tan(theta_FOV)

def P_hit(h):
    """Detection probability as in thesis: P0 * f_res(h)."""
    return P0 * f_res(h)

def visible_cells_at(pos_xyz, grid_points, fov_angle=theta_FOV):
    """
    Compute which grid cells are visible from pos_xyz and their
    single-pass detection probability.

    fov_angle is the half-angle of the conical sensor model.
    """
    x, y, h = pos_xyz

    # altitude-dependent footprint
    r = h * np.tan(fov_angle)

    dx = grid_points[:, 0] - x
    dy = grid_points[:, 1] - y
    d2 = dx*dx + dy*dy

    mask = d2 <= r*r

    # detection probability depends only on altitude
    p_hit = np.zeros(len(grid_points), dtype=float)
    if np.any(mask):
        p_hit[mask] = P_hit(h)

    return mask, p_hit

def victim_signal_at_alt(grid_points, victim_pos, alt):
    """
    Compute which grid cells would contain the victim *if* the UAV were
    looking at altitude = alt, using the same FOV model as the sensor.
    """
    vx, vy = victim_pos
    r = r_FOV(alt)   # same FOV model as UAV

    dx = grid_points[:, 0] - vx
    dy = grid_points[:, 1] - vy
    d2 = dx*dx + dy*dy

    return (d2 <= r*r).astype(float)


def expected_info_gain(path, belief, grid_points, fov_angle=theta_FOV, pred_depth=3):
    """
    Compute expected information gain along a candidate path.

    We approximate the future by assuming that, along the planned path,
    we get *no detection* and update the belief accordingly using a
    Bayesian "no detection" update:

        p' = p (1 - q) / (1 - p q)

    where p is the prior belief and q = P_hit(h) is the single-pass
    detection probability at altitude h.

    The information gain is the reduction in entropy H(p) - H(p').
    """
    pred_belief = belief.copy()
    total_IG = 0.0
    vis_mask_total = np.zeros(len(grid_points), dtype=bool)

    for i in range(len(path) - 1):
        if i >= pred_depth:
            break

        p0, p1 = path[i], path[i + 1]
        seg_len = np.linalg.norm(p1[:2] - p0[:2])

        # sample points along the segment
        n_samples = max(2, int(np.ceil(seg_len / 20.0)))
        xs = np.linspace(p0[0], p1[0], n_samples)
        ys = np.linspace(p0[1], p1[1], n_samples)
        zs = np.linspace(p0[2], p1[2], n_samples)

        for k in range(n_samples):
            pos = np.array([xs[k], ys[k], zs[k]])
            mask, p_hit = visible_cells_at(pos, grid_points, fov_angle=fov_angle)

            if not np.any(mask):
                continue

            # prior belief and entropy for visible cells
            p_prior = pred_belief[mask]
            H_prior = cell_entropy_map(p_prior)

            # detection probability for visible cells
            q = p_hit[mask]  # already 0..1

            # Bayesian update for "no detection"
            # p' = p (1 - q) / (1 - p q)
            num = p_prior * (1.0 - q)
            denom = 1.0 - p_prior * q
            denom = np.where(denom < 1e-9, 1e-9, denom)   # avoid division by zero
            p_post = num / denom
            p_post = np.clip(p_post, 1e-9, 1.0 - 1e-9)

            H_post = cell_entropy_map(p_post)

            # entropy reduction (information gain) for this sample
            delta_H = H_prior - H_post
            total_IG += np.sum(delta_H)

            # update predicted belief along the path so we don't
            # double-count repeated looks at the same cells
            pred_belief[mask] = p_post
            vis_mask_total |= mask

    return total_IG, vis_mask_total


# ==============================================================================
# 1. HELPER FUNCTIONS (Must support multi-step paths)
# ==============================================================================

def expected_detection(path, belief, grid_points, fov_angle, pred_depth=3):
    DET = 0.0
    # Iterate through segments (p0->p1, p1->p2...)
    for i in range(len(path) - 1):
        if i >= pred_depth: break
        p0, p1 = path[i], path[i+1]
        
        # Sample points along the segment
        seg_len = np.linalg.norm(p1[:2] - p0[:2])
        n_samples = max(2, int(np.ceil(seg_len / 10.0))) # Sample every 10m
        
        xs = np.linspace(p0[0], p1[0], n_samples)
        ys = np.linspace(p0[1], p1[1], n_samples)
        zs = np.linspace(p0[2], p1[2], n_samples)

        for x, y, z in zip(xs, ys, zs):
            pos = np.array([x, y, z])
            vis_mask, p_hit = visible_cells_at(pos, grid_points, fov_angle=fov_angle)
            if np.any(vis_mask):
                DET += np.sum(belief[vis_mask] * p_hit[vis_mask])
    return DET

def energy_of_path(path):
    P_hover, P_cruise = 920.0, 300.0
    v_cruise, v_climb, v_desc = 19.0, 2.0, 3.0
    total_energy = 0.0

    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i+1]
        dist_xy = np.linalg.norm(p1[:2] - p0[:2])
        dz = p1[2] - p0[2]

        if dist_xy > 1e-3:
            E_horiz = P_cruise * (dist_xy / v_cruise)
        else:
            E_horiz = 0.0

        if dz > 0:   E_vert = P_hover * (dz / v_climb)
        elif dz < 0: E_vert = P_hover * (-dz / v_desc)
        else:        E_vert = 0.0
        
        total_energy += (E_horiz + E_vert)
    return total_energy

# ==============================================================================
# 2. THE PLANNER (Supports 'constraint_poly' and 'drone_vel')
# ==============================================================================

def plan_velocity_ipp_3D(drone_pos, drone_vel, belief, grid_points, soft_poly,
                         constraint_poly=None,  # <--- CONSTRAINT ARGUMENT
                         fov_angle=theta_FOV, v_max=20.0, n_directions=16,
                         step_length=40.0, altitude_candidates=[30, 50, 70],
                         I_scale=100.0, E_scale=100.0, alpha_d=0.5, buffer=0.0,
                         pred_depth=3, lam=0.0, MAP_bonus=False):

    # 1. Constraint Logic
    if constraint_poly is not None:
        # Use INTERSECTION of Geofence and Cone
        valid_area = soft_poly.intersection(constraint_poly).buffer(0)
    else:
        valid_area = Polygon(soft_poly)
        if buffer > 0.0: valid_area = valid_area.buffer(-buffer)

    if valid_area.is_empty:
        return 0.0, 0.0, 0.0, [drone_pos, drone_pos], None, 0.0

    # 2. Motion Primitives
    cx, cy, cz = drone_pos
    vx, vy, vz = drone_vel
    current_speed = np.linalg.norm([vx, vy])

    # If moving fast, limit turn rate. If slow/hovering, turn anywhere.
    if current_speed > 2.0:
        current_heading = np.arctan2(vy, vx)
        angles = current_heading + np.linspace(-np.pi/3, np.pi/3, n_directions)
    else:
        angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

    candidates = []

    # 3. Generate Trajectories
    for a in angles:
        dx, dy = np.cos(a), np.sin(a)
        for alt in altitude_candidates:
            # Start a new path
            trajectory = [np.array([cx, cy, cz])]
            valid_traj = True
            
            for k in range(pred_depth):
                prev = trajectory[-1]
                # Next waypoint
                nx = prev[0] + dx * step_length
                ny = prev[1] + dy * step_length
                nz = alt
                
                # Check if INSIDE cone
                if not valid_area.contains(Point(nx, ny)):
                    valid_traj = False
                    break
                trajectory.append(np.array([nx, ny, nz]))
            
            if valid_traj:
                candidates.append(trajectory)

    # Fallback: If no paths found (cone too small), try smaller steps
    if not candidates and constraint_poly is not None:
        small_step = 10.0
        for a in angles:
            dx, dy = np.cos(a), np.sin(a)
            end = np.array([cx + dx*small_step, cy + dy*small_step, cz])
            if valid_area.contains(Point(end[0], end[1])):
                candidates.append([np.array([cx, cy, cz]), end])

    # 4. Evaluation Loop
    best_J = -np.inf
    best_path = None
    best_mask = None

    for path in candidates:
        I_p, vis_mask = expected_info_gain(path, belief, grid_points, fov_angle, pred_depth)
        DET_p = expected_detection(path, belief, grid_points, fov_angle, pred_depth)
        E_p = energy_of_path(path)

        J = (1.0 - lam) * (I_p / I_scale) + lam * (DET_p / I_scale) - alpha_d * (E_p / E_scale)
        
        if J > best_J:
            best_J = J
            best_path = path
            best_mask = vis_mask

    if best_path is None:
        return 0.0, 0.0, 0.0, [drone_pos, drone_pos], None, 0.0

    # 5. Output Velocity
    p0, p1 = best_path[0], best_path[1]
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    
    # Scale velocity
    travel_time = max(dist / v_max, 0.1)
    vx = vec[0] / travel_time
    vy = vec[1] / travel_time
    vz = vec[2] / travel_time

    return float(vx), float(vy), float(vz), best_path, best_mask, best_J


# def energy_of_path(path):
#     """
#     Approximate energy cost of a multi-segment path.
#     Sums the energy of each segment.
#     """
#     P_hover = 920.0      # W
#     P_cruise = 300.0     # W
#     v_cruise = 19.0      # m/s
#     v_climb = 2.0
#     v_desc = 3.0

#     total_energy = 0.0

#     # Iterate over segments
#     for i in range(len(path) - 1):
#         p0, p1 = path[i], path[i+1]

#         # Horizontal distance and vertical difference for this segment
#         dist_xy = np.linalg.norm(p1[:2] - p0[:2])
#         dz = p1[2] - p0[2]

#         # --- Horizontal energy ---
#         if dist_xy > 1e-3:
#             dt_horiz = dist_xy / v_cruise
#             E_horiz = P_cruise * dt_horiz
#         else:
#             E_horiz = 0.0

#         # --- Vertical energy ---
#         if dz > 0:
#             # Climbing
#             dt_vert = dz / max(v_climb, 1e-3)
#             E_vert = P_hover * dt_vert
#         elif dz < 0:
#             # Descending
#             dt_vert = (-dz) / max(v_desc, 1e-3)
#             E_vert = P_hover * dt_vert
#         else:
#             E_vert = 0.0

#         total_energy += (E_horiz + E_vert)

#     return total_energy

# def energy_of_path(path):
#     """
#     Approximate energy cost of a path segment.

#     - Horizontal motion flown at cruise:
#         v_cruise ≈ 19 m/s, P_cruise ≈ 300 W
#     - Vertical motion done in hover:
#         P_hover ≈ 920 W, climb ≈ 2 m/s, descend ≈ 3 m/s

#     This makes energy scale with distance and altitude change,
#     instead of being (almost) constant for all paths.
#     """
#     P_hover = 920.0      # W
#     P_cruise = 300.0     # W
#     v_cruise = 19.0      # m/s (fixed-wing cruise)

#     v_climb = 2.0        # m/s (hover climb)
#     v_desc = 3.0         # m/s (hover descent)

#     p0, p1 = path[0], path[-1]

#     # horizontal distance and vertical difference
#     dist_xy = np.linalg.norm(p1[:2] - p0[:2])
#     dz = p1[2] - p0[2]

#     # --- horizontal energy (cruise) ---
#     if dist_xy > 1e-3:
#         dt_horiz = dist_xy / v_cruise
#         E_horiz = P_cruise * dt_horiz
#     else:
#         E_horiz = 0.0

#     # --- vertical energy (hover) ---
#     if dz > 0:
#         # climbing
#         dt_vert = dz / max(v_climb, 1e-3)
#         E_vert = P_hover * dt_vert
#     elif dz < 0:
#         # descending
#         dt_vert = (-dz) / max(v_desc, 1e-3)
#         E_vert = P_hover * dt_vert
#     else:
#         E_vert = 0.0

#     return E_horiz + E_vert


def propagate_belief_vectorized(belief, grid_points, v_drift, dt, sigma=5.0):
    prev_positions = grid_points - v_drift * dt
    tree = cKDTree(grid_points)
    _, idx = tree.query(prev_positions)
    new_belief = belief[idx]

    x_coords = np.unique(grid_points[:, 0])
    y_coords = np.unique(grid_points[:, 1])
    x_idx = np.searchsorted(x_coords, grid_points[:, 0])
    y_idx = np.searchsorted(y_coords, grid_points[:, 1])

    grid_2d = np.full((len(x_coords), len(y_coords)), np.nan)
    grid_2d[x_idx, y_idx] = new_belief
    mask = np.isnan(grid_2d)
    grid_2d[mask] = 0.5
    grid_2d = gaussian_filter(grid_2d, sigma=sigma)
    grid_2d = np.clip(grid_2d, 0.0, 1.0)

    new_belief = grid_2d[x_idx, y_idx]
    return new_belief

# def expected_detection(path, belief, grid_points, fov_angle, pred_depth=3):
#     """
#     Computes DET = sum( p * p_hit ) along the path, similar to entropy-based IG.
#     """
#     DET = 0.0

#     p0, p1 = path
#     seg = int(max(3, np.ceil(np.linalg.norm(p1[:2] - p0[:2]) / 20.0)))
#     xs = np.linspace(p0[0], p1[0], seg)
#     ys = np.linspace(p0[1], p1[1], seg)
#     zs = np.linspace(p0[2], p1[2], seg)

#     for x, y, z in zip(xs, ys, zs):
#         pos = np.array([x, y, z])
#         vis_mask, p_hit = visible_cells_at(pos, grid_points)
#         DET += np.sum(belief[vis_mask] * p_hit[vis_mask])

#     return DET

# def expected_detection(path, belief, grid_points, fov_angle, pred_depth=3):
#     """
#     Computes DET = sum( p * p_hit ) along the path, iterating through segments.
#     """
#     DET = 0.0
    
#     # Iterate over each segment of the trajectory (p0->p1, p1->p2, etc.)
#     for i in range(len(path) - 1):
#         if i >= pred_depth:
#             break
            
#         p0, p1 = path[i], path[i+1]
        
#         # Sample points along this specific segment
#         seg_len = np.linalg.norm(p1[:2] - p0[:2])
#         n_samples = max(2, int(np.ceil(seg_len / 20.0)))
        
#         xs = np.linspace(p0[0], p1[0], n_samples)
#         ys = np.linspace(p0[1], p1[1], n_samples)
#         zs = np.linspace(p0[2], p1[2], n_samples)

#         for x, y, z in zip(xs, ys, zs):
#             pos = np.array([x, y, z])
#             vis_mask, p_hit = visible_cells_at(pos, grid_points, fov_angle=fov_angle)
            
#             # Accumulate detection score
#             DET += np.sum(belief[vis_mask] * p_hit[vis_mask])

#     return DET

# def plan_velocity_ipp_3D(drone_pos, belief, grid_points, soft_poly, 
#                          fov_angle=theta_FOV, v_max=12.0, n_directions=16, 
#                          step_length=40.0, altitude_candidates=[30,50,70],
#                          I_scale=100.0, E_scale=100.0, alpha_d=0.5, buffer=0.0,
#                          pred_depth=3, lam=0.0, MAP_bonus=False):

#     # shrink polygon by buffer
#     if buffer>0.0:
#         poly = Polygon(soft_poly).buffer(-buffer)
#     else:
#         poly = Polygon(soft_poly)

#     cx, cy, cz = drone_pos
#     angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
#     candidates = []

#     # candidate generation
#     for a in angles:
#         dx, dy = np.cos(a), np.sin(a)
#         for alt in altitude_candidates:
#             end = np.array([cx + dx*step_length, cy + dy*step_length, alt])
#             if poly.contains(Point(end[0], end[1])):
#                 candidates.append([np.array([cx, cy, cz]), end])

#     best_J, best_path, best_mask = -np.inf, None, None

#     # -----------------------
#     # SCORING LOOP (patched)
#     # -----------------------
#     for path in candidates:

#         # information gain (existing)
#         I_p, vis_mask = expected_info_gain(
#             path, belief, grid_points, 
#             fov_angle=fov_angle, pred_depth=pred_depth
#         )

#         # exploitation utility (new)
#         DET_p = expected_detection(
#             path, belief, grid_points, fov_angle=fov_angle, pred_depth=pred_depth
#         )

#         # energy cost (existing)
#         E_p = energy_of_path(path)

#         # combined exploration + exploitation reward
#         J = (1.0 - lam) * (I_p / I_scale) \
#             + lam * (DET_p / I_scale) \
#             - alpha_d * (E_p / E_scale)

#         # optional MAP-cell bonus
#         if MAP_bonus:
#             # find MAP cell inside supplied grid_points region
#             max_idx = np.argmax(belief)
#             MAP_cell = grid_points[max_idx]
#             # distance from candidate endpoint
#             dist = np.linalg.norm(path[-1][:2] - MAP_cell)
#             bonus = 0.1 * np.exp(-0.02 * dist)
#             J += bonus

#         if J > best_J:
#             best_J = J
#             best_path = path
#             best_mask = vis_mask

#     # safety
#     if best_path is None:
#         return 0.0, 0.0, 0.0, None, None, 0.0

#     # compute velocity along best path (unchanged)
#     p0, p1 = best_path[0], best_path[-1]
#     vec = p1 - p0
#     dist = np.linalg.norm(vec)
#     if dist < 1e-3:
#         return 0.0, 0.0, 0.0, best_path, best_mask, best_J

#     travel_fraction = min(1.0, 1.0/max(dist/v_max,1e-6))
#     step_vec = vec * travel_fraction
#     vx, vy, vz = step_vec

#     return float(vx), float(vy), float(vz), best_path, best_mask, best_J

# def plan_velocity_ipp_3D(drone_pos, drone_vel, belief, grid_points, soft_poly,
#                          constraint_poly=None,  # <--- NEW: Accepts Drift Cone
#                          fov_angle=theta_FOV, v_max=20.0, n_directions=16,
#                          step_length=40.0, altitude_candidates=[30, 50, 70],
#                          I_scale=100.0, E_scale=100.0, alpha_d=0.5, buffer=0.0,
#                          pred_depth=3, lam=0.0, MAP_bonus=False):

#     # --- 1. Define Search Area (Constraints) ---
#     # Intersect Global Geofence (soft_poly) with Local Tracking Cone (constraint_poly)
#     if constraint_poly is not None:
#         # Buffer(0) fixes self-intersection topology errors
#         valid_area = soft_poly.intersection(constraint_poly).buffer(0)
#     else:
#         valid_area = Polygon(soft_poly)
#         if buffer > 0.0:
#             valid_area = valid_area.buffer(-buffer)

#     if valid_area.is_empty:
#         # Fallback: Stay in place if constraints are impossible
#         return 0.0, 0.0, 0.0, None, None, 0.0

#     cx, cy, cz = drone_pos
#     vx, vy, vz = drone_vel
#     current_speed = np.linalg.norm([vx, vy])

#     # --- 2. Motion Primitives (Thesis Sec 4.3.1) ---
#     # Cruise Mode: Constrained turning (Forward-facing cone)
#     # Hover Mode: Omnidirectional
#     if current_speed > 2.0:
#         current_heading = np.arctan2(vy, vx)
#         # Sample +/- 60 degrees around current velocity
#         angles = current_heading + np.linspace(-np.pi/3, np.pi/3, n_directions)
#     else:
#         # Sample full 360 degrees
#         angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

#     candidates = []

#     # --- 3. Trajectory Generation (Receding Horizon) ---
#     # Instead of just [start, end], we generate [p0, p1, p2, p3...]
    
#     for a in angles:
#         dx, dy = np.cos(a), np.sin(a)
        
#         for alt in altitude_candidates:
#             trajectory = [np.array([cx, cy, cz])] # Start point
#             valid_traj = True
            
#             # Project forward N steps
#             for k in range(1, pred_depth + 1):
#                 last_x, last_y, _ = trajectory[-1]
                
#                 # Next waypoint
#                 next_x = last_x + dx * step_length
#                 next_y = last_y + dy * step_length
#                 next_z = alt  # Assumption: Altitude change happens at step 1
                
#                 next_pt = np.array([next_x, next_y, next_z])
                
#                 # Check Constraints for EVERY point in the horizon
#                 if not valid_area.contains(Point(next_x, next_y)):
#                     valid_traj = False
#                     break
                
#                 trajectory.append(next_pt)

#             if valid_traj:
#                 candidates.append(trajectory)

#     # --- 4. Fallback for Tight Constraints ---
#     # If cone is too small for the step_length, candidates might be empty.
#     # Retry with a single small step (Hover behavior).
#     if not candidates and constraint_poly is not None:
#         small_step = 5.0
#         for a in angles:
#             dx, dy = np.cos(a), np.sin(a)
#             end = np.array([cx + dx*small_step, cy + dy*small_step, cz])
#             if valid_area.contains(Point(end[0], end[1])):
#                 candidates.append([np.array([cx, cy, cz]), end])

#     best_J = -np.inf
#     best_path = None
#     best_mask = None

#     # --- 5. Scoring Loop (Cost-Benefit Evaluation) ---
#     for path in candidates:
        
#         # A. Information Gain (Thesis Eq 3.13)
#         # expected_info_gain now processes the full list of points
#         I_p, vis_mask = expected_info_gain(
#             path, belief, grid_points, 
#             fov_angle=fov_angle, pred_depth=pred_depth
#         )

#         # B. Exploitation Utility (optional)
#         DET_p = expected_detection(
#             path, belief, grid_points, fov_angle=fov_angle, pred_depth=pred_depth
#         )

#         # C. Energy Cost (Thesis Eq 3.12)
#         # Calculates cost based on Cruise vs Hover power profile
#         E_p = energy_of_path(path) 

#         # D. Unified Utility Function
#         J = (1.0 - lam) * (I_p / I_scale) \
#             + lam * (DET_p / I_scale) \
#             - alpha_d * (E_p / E_scale)

#         if J > best_J:
#             best_J = J
#             best_path = path
#             best_mask = vis_mask

#     # --- 6. Execution ---
#     if best_path is None:
#         return 0.0, 0.0, 0.0, None, None, 0.0

#     # Extract first step command (Receding Horizon Principle)
#     p0, p1 = best_path[0], best_path[1]
#     vec = p1 - p0
#     dist = np.linalg.norm(vec)
    
#     if dist < 1e-3:
#         return 0.0, 0.0, 0.0, best_path, best_mask, best_J

#     # Calculate required velocity to reach p1
#     # Note: v_max is ground speed limit
#     travel_fraction = min(1.0, 1.0 / max(dist / v_max, 1e-6))
#     step_vec = vec * travel_fraction
#     vx, vy, vz = step_vec

#     return float(vx), float(vy), float(vz), best_path, best_mask, best_J



def tracking_region(t_now, detection_pos, detection_time, v_drift,
                    base_radius=60.0, uncertainty_gain=1.0,
                    cone_angle_deg=90):
    """
    Creates a drift cone region aligned with v_drift.
    Uncertainty radius grows proportionally to |v_drift|.
    """

    dt = max(t_now - detection_time, 0.0)

    # Drift speed magnitude
    drift_speed = np.linalg.norm(v_drift)

    # Drifted center
    drift_vec = v_drift * dt
    center_R = detection_pos + drift_vec

    # Expansion based on drift
    radius_R = base_radius + uncertainty_gain * drift_speed * dt

    # Cone geometry
    theta = np.arctan2(v_drift[1], v_drift[0])
    half_angle = np.deg2rad(cone_angle_deg / 2)

    angles = np.linspace(-half_angle, half_angle, 50)
    x_arc = radius_R * np.cos(angles)
    y_arc = radius_R * np.sin(angles)

    # Build polygon
    cone_points = np.vstack((
        [0,0],
        np.column_stack((x_arc, y_arc)),
        [0,0]
    ))

    cone_poly = Polygon(cone_points)
    cone_poly = rotate(cone_poly, np.degrees(theta), origin=(0,0))
    cone_poly = translate(cone_poly, detection_pos[0], detection_pos[1])

    return cone_poly, center_R, radius_R


# def cone_belief_update(
#     belief, grid_points, region_poly, detection_pos, v_drift,
#     t_now, detection_time,
#     p0=0.5, base_sigma_long=200.0, base_sigma_lat=120.0,
#     growth_long=25.0, growth_lat=15.0, decay_time=400.0,
#     blend=0.7
# ):
#     """
#     Hybrid drift-cone belief update.
#     - Keeps anisotropic diffusion shape (Eq. 7.14–7.17 in the thesis).
#     - Smooth visual behavior on a discrete grid.
#     - Preserves probability mass locally (inside the cone), not globally.
#     """
#     if region_poly is None or region_poly.is_empty:
#         return belief

#     dt = max(t_now - detection_time, 0.0)
#     new_belief = belief.copy()

#     # Drift-aligned basis
#     drift_hat = v_drift / np.linalg.norm(v_drift)
#     perp_hat = np.array([-drift_hat[1], drift_hat[0]])

#     # Drifted Gaussian center
#     center_shift = detection_pos + v_drift * dt

#     # Diffusion spread (√t scaling)
#     sigma_long = base_sigma_long + growth_long * np.sqrt(dt + 1.0)
#     sigma_lat  = base_sigma_lat  + growth_lat  * np.sqrt(dt + 1.0)

#     # Decay of peak probability with time
#     p_peak = p0 * np.exp(-dt / decay_time)

#     # Precompute mask for region
#     mask_cone = np.array([region_poly.contains(Point(p)) for p in grid_points])

#     for i, p in enumerate(grid_points[mask_cone]):
#         vec = p - center_shift
#         d_along = np.dot(vec, drift_hat)
#         d_cross = np.dot(vec, perp_hat)

#         # Anisotropic Gaussian kernel (no strict normalization)
#         g = np.exp(-0.5 * ((d_along / sigma_long) ** 2 + (d_cross / sigma_lat) ** 2))
#         p_val = p_peak * g

#         idx = np.where(mask_cone)[0][i]
#         new_belief[idx] = blend * belief[idx] + (1 - blend) * p_val

#     # Normalize locally within cone region to conserve probability mass
#     total_prev = np.sum(belief[mask_cone])
#     total_new  = np.sum(new_belief[mask_cone])
#     if total_new > 1e-9:
#         scale_factor = total_prev / total_new
#         new_belief[mask_cone] *= scale_factor

#     return np.clip(new_belief, 0.0, 1.0)

def clear_confirmed_region(belief, grid_points, victim_pos, wind_vec,
                           major_axis=120.0, minor_axis=60.0, decay=0.01):
    """
    Clear belief in a drift-aligned elliptical region after victim confirmation.

    major_axis : length (m) along wind direction
    minor_axis : length (m) perpendicular to wind
    decay      : residual probability after clearing (0.0–0.05 recommended)
    """

    # Normalize wind direction
    w = wind_vec[:2]
    if np.linalg.norm(w) < 1e-6:
        wind_hat = np.array([1.0, 0.0])   # fallback
    else:
        wind_hat = w / np.linalg.norm(w)

    # Perpendicular axis
    perp_hat = np.array([-wind_hat[1], wind_hat[0]])

    # Compute vector from ellipse center
    rel = grid_points - victim_pos

    # Project onto drift-aligned axes
    a = np.dot(rel, wind_hat)      # along-wind
    b = np.dot(rel, perp_hat)      # cross-wind

    # Ellipse mask (a/major)^2 + (b/minor)^2 <= 1
    mask = (a / major_axis) ** 2 + (b / minor_axis) ** 2 <= 1.0

    # Apply clearing with soft decay
    belief[mask] = decay

    return belief


def local_centroid(belief, grid_points, idx, radius=40):
    center = grid_points[idx]
    diffs = grid_points - center
    d2 = diffs[:,0]**2 + diffs[:,1]**2
    mask = d2 < radius**2

    if np.sum(mask) < 3:
        return center

    w = belief[mask]
    pts = grid_points[mask]
    pt = np.average(pts, axis=0, weights=w)
    return pt

def drift_aligned_centroid(belief, grid_points, idx, wind_vec,
                           major=150.0, minor=80.0):
    """
    Compute a weighted centroid around the MAP cell using a drift-aligned ellipse.
    major/minor are axis lengths (meters).
    """
    center = grid_points[idx]

    # Normalize drift direction
    w = wind_vec[:2]
    if np.linalg.norm(w) < 1e-6:
        wind_hat = np.array([1.0, 0.0])
    else:
        wind_hat = w / np.linalg.norm(w)

    # Perpendicular axis
    perp_hat = np.array([-wind_hat[1], wind_hat[0]])

    # Relative vectors to all grid points
    rel = grid_points - center
    a = np.dot(rel, wind_hat)     # along wind
    b = np.dot(rel, perp_hat)     # across wind

    # Elliptical selection region
    mask = (a/major)**2 + (b/minor)**2 <= 1.0

    # Fallback if not enough points
    if np.sum(mask) < 5:
        return center

    # Weighted average
    wts = belief[mask]
    pts = grid_points[mask]
    centroid = np.average(pts, axis=0, weights=wts)

    return centroid

def active_cone_mask(t_now, grid_points, track, v_drift):
    """
    Returns a boolean mask over grid_points that is True where
    ANY active tracker’s cone (buffered) is present.
    """
    mask = np.zeros(len(grid_points), dtype=bool)

    for tr in track:
        if not tr["active"] or tr["pos"] is None or tr["time"] is None:
            continue

        region_poly, _, _ = tracking_region(
            t_now=t_now,
            detection_pos=tr["pos"],
            detection_time=tr["time"],
            v_drift=v_drift
        )

        if region_poly and not region_poly.is_empty:
            # same 20 m buffer you were using
            region_poly_buffered = region_poly.buffer(20.0)
            in_cone = np.array([
                region_poly_buffered.contains(Point(p))
                for p in grid_points
            ])
            mask |= in_cone

    return mask



confirm_pconf = 0.25
rho_th_fov = 0.1
mean_thresh = 0.30

# --- Build grid and inside-polygon mask ---
XX, YY = np.meshgrid(grid_x, grid_y)
grid_points_all = np.column_stack([XX.ravel(), YY.ravel()])

inside_mask_full = np.array([soft_poly.contains(Point(p)) for p in grid_points_all])
grid_points = grid_points_all[inside_mask_full]

# belief must match grid_points, not inside_mask_full
belief = 0.5 * np.ones(len(grid_points))

# --- Mapping from ALL -> inside ---
all_to_inside = -np.ones(len(grid_points_all), dtype=int)
all_to_inside[np.where(inside_mask_full)[0]] = np.arange(np.sum(inside_mask_full))


# Precompute victim signal on full grid
victims = np.array([[50,-200],[50,-400]], dtype=float)

# -----------------------------
# SIMULATION PARAMETERS
# -----------------------------
v_max = 20.0        # max velocity [m/s]
a_max = 2.0         # max acceleration [m/s²]
# -----------------------------
# MULTI-UAV SETUP
# -----------------------------
num_drones = 2
drone_positions = [np.array([400.0, 0.0, uav_nominal_altitudes[0]]),
                   np.array([50.0, -300.0, uav_nominal_altitudes[1]])]
drone_vels = [np.zeros(3) for _ in range(num_drones)]
drone_modes = ['explore' for _ in range(num_drones)]
pred_depth = 3

# -----------------------------
# SIMULATION LOOP
# -----------------------------
max_dheading = np.deg2rad(10)  # max heading change per second
# For simplicity, victim velocity (constant drift)
victim_vel = np.array([0.5, 0.2, 0.0])  # m/s


# ============================
# Energy Visualization Setup
# ============================

fig_energy, ax_energy = plt.subplots(figsize=(8, 4))
fig_energy.suptitle("UAV Remaining Energy Over Time")

# Assume all UAVs start with 100% energy (normalized)
E_max = 100.0
E_rem = np.ones(num_drones) * E_max

# Create line handles
energy_lines = []
for d_idx in range(num_drones):
    color = plt.cm.tab10(d_idx)
    (line,) = ax_energy.plot([], [], color=color, label=f"UAV{d_idx}")
    energy_lines.append(line)

# Storage for energy history
energy_time = []
energy_history = [np.empty((0,)) for _ in range(num_drones)]

# Axis labels and legend
ax_energy.set_xlabel("Time [s]")
ax_energy.set_ylabel("Remaining Energy [%]")
ax_energy.set_ylim(0, 110)
ax_energy.grid(True)
ax_energy.legend(loc="upper right")


## ============================
# Visualization Setup (clean)
# ============================
cmap = plt.cm.get_cmap('RdYlBu_r')

# Initialize raster (beliefs for all grid points)
raster_belief = np.full_like(grid_points_all[:, 0], np.nan, dtype=float)
raster_belief[inside_mask_full] = belief


# --- Main occupancy map figure ---
fig, ax_map = plt.subplots(figsize=(8, 8))

sc = ax_map.scatter(
    grid_points_all[:, 0], grid_points_all[:, 1],
    c=raster_belief, cmap=cmap, s=20, vmin=0.0, vmax=1.0
)

# Drone markers
drone_plots = []
for d_idx in range(num_drones):
    (p,) = ax_map.plot(
        drone_positions[d_idx][0], drone_positions[d_idx][1],
        'o', markersize=8, label=f"UAV{d_idx}"
    )
    drone_plots.append(p)

# Victim markers
victim_plot = ax_map.scatter(
    victims[:, 0], victims[:, 1],
    c='r', marker='x', s=60, label='Victims'
)

# Persistent drift-cone graphics
(cone_line,) = ax_map.plot([], [], 'r--', lw=2.0, label="Drift cone R(t)")
(cone_center_dot,) = ax_map.plot([], [], 'bo', markersize=6, label="Cone center")
cone_line.set_zorder(5)
cone_center_dot.set_zorder(6)
ax_map.cone_fill = None  # placeholder for the red cone fill patch

(det_dot,) = ax_map.plot([], [], 'mo', markersize=8, label="Detection point")
det_dot.set_zorder(7)
# detection origin marker (pink)
(detection_marker,) = ax_map.plot([], [], 'mo', markersize=8, label='Detection point')


# Colorbar and labels
cbar = plt.colorbar(sc, ax=ax_map)
cbar.set_label('Occupancy probability')

ax_map.set_xlim(x_min - 20, x_max + 20)
ax_map.set_ylim(y_min - 20, y_max + 20)
ax_map.set_xlabel('X [m]')
ax_map.set_ylabel('Y [m]')
ax_map.set_title('Multi-UAV Occupancy Map')
ax_map.legend(loc='upper right')

plt.ion()
plt.show()

# ============================
# Velocity Visualization Setup
# ============================
fig_vel, axs_vel = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig_vel.suptitle("UAV Velocity Components (vx, vy, vz)")

# Prepare subplots for vx, vy, vz
for i, lbl in enumerate(["vx", "vy", "vz"]):
    axs_vel[i].set_ylabel(lbl + " [m/s]")
    axs_vel[i].grid(True)
axs_vel[-1].set_xlabel("Time [s]")

# Create line handles for each UAV
vel_lines = []
for d_idx in range(num_drones):
    color = plt.cm.tab10(d_idx)
    drone_lines = [
        axs_vel[0].plot([], [], color=color, label=f"UAV{d_idx} vx")[0],
        axs_vel[1].plot([], [], color=color, label=f"UAV{d_idx} vy")[0],
        axs_vel[2].plot([], [], color=color, label=f"UAV{d_idx} vz")[0],
    ]
    vel_lines.append(drone_lines)

for axv in axs_vel:
    axv.legend(loc="upper right")

# Initialize storage for velocity history
vel_time = []
vel_history = [np.empty((0, 3)) for _ in range(num_drones)]


# -----------------------------
# SIMULATION LOOP (multi-UAV)
# -----------------------------
max_dheading = np.deg2rad(10)  # max heading change per second
buffer = FOV_radius  # safe buffer for polygon

# Create a buffered polygon for safe navigation
safe_poly = soft_poly.buffer(-buffer) if buffer > 0 else soft_poly

update_strength = 0.3  # occupancy update weight

# Initialize UAV states and tracking flags
drone_modes = ["explore"] * num_drones   # 'explore' or 'track'
detection_active = [False]*num_drones
detection_confirmed = [False]*num_drones
detection_pos = [None]*num_drones
detection_time = [None]*num_drones
tracker_phase = [None]*num_drones
cooldown_until = np.zeros(num_drones)
COOLDOWN_TIME = 30  # seconds of detection lockout
cone_artists = []

# --- Tracking timing parameters ---
initial_lock_duration = 8.0    # seconds tracking cone center before IPP
cone_search_timeout   = 30.0   # max time doing IPP inside cone before abort
HOVER_CONFIRM_TIME = 10.0  # seconds at low altitude before abort if not confirmed

detection_event = [False]*num_drones


track = []
for _ in range(num_drones):
    track.append({
        "active": False,
        "confirmed": False,
        "pos": None,            # detection_pos for THIS UAV
        "time": None,           # detection_time for THIS UAV
        # phases: "to_detection", "initial_lock", "cone_tracking", "hover_confirm"
        "phase": None,
        "lock_start": None,     # time when initial_lock starts
        "cone_start": None      # time when cone_tracking (IPP in cone) starts
    })


# Pre-allocate cone graphics for each UAV
cone_lines = []
cone_centers = []
cone_fills = []

# =====================================
# SIMULATION GENERAL CONSTANTS
# =====================================

dt = 1.0           # simulation time step [s]
MIN_ALT = 20.0
DESCENT_STEP = 3.0
HOVER_CONFIRM_TIME = 10.0  # seconds at min altitude before confirmation or abort



for d_idx in range(num_drones):
    color = plt.cm.tab10(d_idx)

    (line,) = ax_map.plot([], [], '--', lw=2, color=color, alpha=0.9)
    (center,) = ax_map.plot([], [], 'o', color=color, markersize=6)

    from matplotlib.patches import Polygon as MplPolygon
    fill = MplPolygon([[0,0],[0,0],[0,0]], closed=True,
                      facecolor=color, alpha=0.15)
    ax_map.add_patch(fill)

    cone_lines.append(line)
    cone_centers.append(center)
    cone_fills.append(fill)



if __name__ == "__main__":
    

    for t in range(250):
        # ------------------------------------------
        # 0. Dynamic mapping: propagate belief (drift model)
        # ------------------------------------------
        belief = propagate_belief_vectorized(belief, grid_points, v_drift, dt_step, sigma=1.0)
        # ------------------------------------------
        # 1. Victim motion (drift)
        # ------------------------------------------
        victims[:, :2] += v_drift * dt_step
        # ------------------------------------------
        # 2. Victim signal (altitude-dependent footprint)
        # ------------------------------------------
        victim_signal_full = np.zeros(len(grid_points_all), dtype=float)
        # use the lowest UAV altitude (most sensitive footprint)
        min_alt = min(pos[2] for pos in drone_positions)

        for v in victims:
            vsig = victim_signal_at_alt(grid_points_all, v, min_alt)
            victim_signal_full = np.maximum(victim_signal_full, vsig)

        victim_signal = victim_signal_full[inside_mask_full]

        # ------------------------------------------
        # 3. FOV overlap counting with altitude-dependent p_hit
        # ------------------------------------------
        # update_weight = np.zeros(len(grid_points), dtype=float)
        update_weight = np.zeros(len(belief), dtype=float)


        for d_idx in range(num_drones):
            vis_mask_all, p_hit_all = visible_cells_at(drone_positions[d_idx], grid_points_all)

            vis_mask_inside = vis_mask_all[inside_mask_full]       # mask over grid_points
            p_hit_inside = p_hit_all[inside_mask_full]             # probabilities over grid_points
            # update_weight[vis_mask_inside] += p_hit_inside
            update_weight[vis_mask_inside] += p_hit_inside[vis_mask_inside]

        # cap at 1.0 so multiple UAVs don't exceed "certain" detection
        update_weight = np.clip(update_weight, 0.0, 1.0)

        # ------------------------------------------
        # 4. Bayesian update of visible cells (single global update)
        # ------------------------------------------
        # alpha = update_strength * update_weight          # per-cell learning rate
        # belief += alpha * (victim_signal - belief)
        # belief = np.clip(belief, 0.0, 1.0)

        # ------------------------------------------
        # 4. Bayesian mapping update (per UAV — FIXED MODEL)
        # ------------------------------------------

        # Precompute which grid cells contain the victim(s)
        victim_cell_mask = np.zeros(len(grid_points), dtype=bool)
        for v in victims:
            d2 = np.sum((grid_points - v[:2])**2, axis=1)
            # victim_cell_mask |= (d2 <= cell_size**2)   # victim occupies its nearest cell
            victim_cell_mask |= (d2 == np.min(d2))


        for d_idx in range(num_drones):

            drone_pos = drone_positions[d_idx]

            # Visibility on the belief grid (not full grid)
            vis_mask, p_hit = visible_cells_at(drone_pos, grid_points)

            # Extract priors for visible cells
            p_prior = belief[vis_mask]
            if p_prior.size == 0:
                continue

            # --- Which visible cells truly contain a victim? ---
            visible_victim_mask = vis_mask & victim_cell_mask
            visible_victim_mask_local = victim_cell_mask[vis_mask]   # for q-aligned indexing

            # --- DETECTION PROBABILITY & EVENT (Bug #3 Fix) ---
            if np.any(visible_victim_mask):
                # True victim inside FOV → boost probability to avoid false negatives
                base_p_det = np.max(p_hit[visible_victim_mask])
                p_detect = min(1.0, base_p_det + 0.4)   # strong signal when victim is actually visible
            else:
                # No victim in FOV → only false positives possible
                p_detect = P_false

            # Binary detection event
            detection_flag = (np.random.rand() < p_detect)
            detection_event[d_idx] = detection_flag


            # Extract q = p_hit for visible cells, aligned to vis_mask
            q = p_hit[vis_mask]

            # --- BAYES UPDATE ---
            if detection_flag:
                # HIT update
                num = p_prior * q
                den = p_prior * q + (1 - p_prior) * P_false
                den = np.maximum(den, 1e-12)
                p_post = num / den

            else:
                # NO-HIT update (softened to avoid victim collapse)
                num = p_prior * (1 - q)
                den = 1 - p_prior * q
                den = np.maximum(den, 1e-12)
                p_post = num / den

                # Prevent blue collapse when victim is actually there
                p_post = np.maximum(p_post, 0.7 * p_prior)

            # Clip posterior and write back
            p_post = np.clip(p_post, 1e-6, 1 - 1e-6)
            belief[vis_mask] = p_post

        # ------------------------------------------
        # 5. Centralized detection & tracker assignment (multi-UAV)
        # ------------------------------------------

        # 5A) Build a belief map that ignores ALL active cones
        belief_for_detection = belief.copy()

        # ---------------------------------------------------------------
        # (5.1) Build detection belief that EXCLUDES all active cones
        # ---------------------------------------------------------------
        in_active_cone = active_cone_mask(
            t_now=t,
            grid_points=grid_points,
            track=track,
            v_drift=v_drift
        )

        belief_for_detection = belief.copy()
        belief_for_detection[in_active_cone] = 0.0


        # ---------------------------------------------------------------
        # (5.2) Select exploration UAVs eligible for new tasks
        # ---------------------------------------------------------------
        eligible_explorers = [
            d_idx
            for d_idx, mode in enumerate(drone_modes)
            if mode == "explore" and t >= cooldown_until[d_idx]
        ]

        # ---------------------------------------------------------------
        # (5.3) FOV-based physical detection trigger
        # ---------------------------------------------------------------
        NEW_DETECTION = False
        detection_uav = None
        detection_pos = None

        # Check all eligible explorers
        for d_idx in eligible_explorers:

            # Must have had a physical sensor detection this timestep
            if not detection_event[d_idx]:
                continue

            # Victim must be physically inside the FOV of this UAV
            vis_mask, _ = visible_cells_at(drone_positions[d_idx], grid_points)
            if not np.any(victim_cell_mask[vis_mask]):
                continue

            # ---- TRUE DETECTION ----
            NEW_DETECTION = True
            detection_uav = d_idx

            # Choose position = MAP cell within FOV intersection
            vis_indices = np.where(vis_mask)[0]
            local_idx = vis_indices[np.argmax(belief[vis_mask])]
            detection_pos = grid_points[local_idx].copy()

            break   # Only first real detection matters


        # ---------------------------------------------------------------
        # (5.4) If no explorers OR no true real detection → skip
        # ---------------------------------------------------------------
        if (not NEW_DETECTION) or detection_uav is None:
            pass
            # IMPORTANT: do NOT fall back to belief-based detection
        else:

            # For convenience
            det_point = Point(detection_pos[0], detection_pos[1])

            # -----------------------------------------------------------
            # (5.5) Duplicate detection suppression
            # -----------------------------------------------------------
            duplicate_thresh = 100.0
            is_duplicate = False

            for tr_exist in track:
                if (not tr_exist["active"]
                    or tr_exist["pos"] is None
                    or tr_exist["time"] is None):
                    continue

                region_poly_exist, _, radius_R = tracking_region(
                    t_now=t,
                    detection_pos=tr_exist["pos"],
                    detection_time=tr_exist["time"],
                    v_drift=v_drift
                )

                if region_poly_exist is None or region_poly_exist.is_empty:
                    continue

                if region_poly_exist.buffer(10.0).contains(det_point):
                    is_duplicate = True
                    break

                d_center = np.linalg.norm(detection_pos - tr_exist["pos"])
                if d_center < max(duplicate_thresh, radius_R * 0.8):
                    is_duplicate = True
                    break

            if is_duplicate:
                print(f"[t={t}] Skipping detection at {detection_pos} — overlaps existing region.")
                pass

            else:
                # -------------------------------------------------------
                # (5.7) Assign nearest explorer to track
                # -------------------------------------------------------
                print(f"[t={t}] TRUE DETECTION at {detection_pos}")

                best_d, best_cost = None, np.inf

                for d_idx in eligible_explorers:
                    r = detection_pos - drone_positions[d_idx][:2]
                    dist = np.linalg.norm(r)
                    r_hat = r / (dist + 1e-6)
                    Deff = dist + gamma_wind * abs(np.dot(v_wind[:2], r_hat))

                    if Deff < best_cost:
                        best_cost = Deff
                        best_d = d_idx

                if best_d is not None:
                    tr_assign = track[best_d]
                    tr_assign["active"] = True
                    tr_assign["confirmed"] = False
                    tr_assign["pos"] = detection_pos.copy()
                    tr_assign["pos_at_detection"] = detection_pos.copy()
                    tr_assign["time"] = t
                    tr_assign["phase"] = "to_detection"

                    drone_modes[best_d] = "track"
                    cooldown_until[best_d] = t + COOLDOWN_TIME

                    print(f"[t={t}] Assigning UAV{best_d} to track region.")



        # ------------------------------------------
        # 6. Plan next velocities
        # ------------------------------------------
        for d_idx in range(num_drones):
            tr = track[d_idx]

            # --- A. EXPLORATION MODE ---
            if drone_modes[d_idx] == "explore":
                vx_des, vy_des, vz_des, *_ = plan_velocity_ipp_3D(
                    drone_positions[d_idx], 
                    drone_vels[d_idx],      # <--- Pass Velocity
                    belief, grid_points, soft_poly,
                    constraint_poly=None,   # <--- No Constraint
                    step_length=40.0,       # Large steps
                    fov_angle=theta_FOV, v_max=v_max, n_directions=16,
                    altitude_candidates=[30, 50, 70], pred_depth=pred_depth
                )
                # Maintain nominal altitude
                vz_des = np.clip(uav_nominal_altitudes[d_idx] - drone_positions[d_idx][2], -1.0, 1.0)

                # 2) Maintain nominal altitude
                # target_alt = uav_nominal_altitudes[d_idx]
                # current_alt = drone_positions[d_idx][2]
                # vz_hold = np.clip(target_alt - current_alt, -1.0, 1.0)
                # vz_des = vz_hold

            elif drone_modes[d_idx] == "track" and tr["active"] and not tr["confirmed"]:

                # === TRACKER STATE MACHINE (per UAV) ===
                if tr["phase"] == "to_detection":
                    # Fix: Track the drifting target, not the static detection point
                    t_elapsed = t - tr["time"]
                    target_now = tr["pos"] + v_drift * t_elapsed
                    
                    vec = target_now - drone_positions[d_idx][:2]
                    dist = np.linalg.norm(vec)
                    print(f"[t={t}] UAV{d_idx} distance to moving target={dist:.2f} m")

                    # Braking logic: Slow down as we get closer
                    if dist > 40.0: 
                        dir_unit = vec / (dist + 1e-6)
                        # Linear braking: 20m/s at distance, slowing to 2m/s at 40m
                        speed = np.clip((dist - 40.0) * 0.5, 5.0, v_max)
                        vx_des, vy_des = dir_unit * speed
                        vz_des = 0.0
                    else:
                        # ARRIVED
                        tr["phase"] = "initial_lock"
                        tr["lock_start"] = t
                        print(f"[t={t}] UAV{d_idx} arrived → starting initial lock on cone.")
                        # Match drift velocity immediately to stop relative motion
                        vx_des, vy_des = v_drift[0], v_drift[1]
                        vz_des = 0.0


                elif tr["phase"] == "initial_lock":
                    det_pos = tr["pos"]
                    det_time = tr["time"]

                    # Follow cone center for a short time (stabilization)
                    region_poly, center_R, radius_R = tracking_region(
                        t_now=t,
                        detection_pos=det_pos,
                        detection_time=det_time,
                        v_drift=v_drift
                    )

                    # Move toward the drifting cone center
                    vec_center = center_R - drone_positions[d_idx][:2]
                    dist_center = np.linalg.norm(vec_center)

                    if dist_center > 5.0:
                        dir_unit = vec_center / (dist_center + 1e-6)
                        vx_des = dir_unit[0] * 5.0
                        vy_des = dir_unit[1] * 5.0
                    else:
                        vx_des = vy_des = 0.0

                    vz_des = 0.0  # hold altitude during initial lock

                    # After initial_lock_duration, switch to IPP inside cone
                    if (t - tr["lock_start"]) >= initial_lock_duration:
                        tr["phase"] = "cone_tracking"
                        tr["cone_start"] = t
                        print(f"[t={t}] UAV{d_idx} finished initial lock → starting cone IPP.")




                elif tr["phase"] == "cone_tracking":

                    det_pos = tr["pos"]
                    det_time = tr["time"]

                    # Cone geometry at this time
                    region_poly, center_R, radius_R = tracking_region(
                        t_now=t, detection_pos=det_pos, detection_time=det_time, v_drift=v_drift
                    )

                    # Apply IPP *inside* cone
                    if region_poly is not None and not region_poly.is_empty:
                        in_R_mask = np.array([region_poly.contains(Point(p)) for p in grid_points])
                        if np.any(in_R_mask):

                            # Extract probabilities inside cone
                            p_vals = belief[in_R_mask]
                            frac_high = np.mean(p_vals > pconf)
                            mean_p = np.mean(p_vals)
                            max_p = np.max(p_vals)

                            print(f"[t={t}] UAV{d_idx} cone stats: ρ={frac_high:.3f}, mean_p={mean_p:.3f}, max_p={max_p:.3f}")

                            # -------------------------------------------------------------
                            # 1. SUCCESS → go to hover-confirm
                            # -------------------------------------------------------------
                            # if (frac_high >= rho_th or mean_p >= 0.30 or max_p >= 0.55):
                            #     tr["phase"] = "hover_confirm"
                            #     tr["fail_timer"] = 0.0
                            #     print(
                            #         f"[t={t}] UAV{d_idx} entering hover-confirmation "
                            #         f"(ρ={frac_high:.2f}, mean_p={mean_p:.2f}, max_p={max_p:.2f})."
                            #     )
                            #     continue

                            # --- SUCCESS: Transition to Confirmation ---
                            if (frac_high >= rho_th or mean_p >= 0.30 or max_p >= 0.55):
                                tr["phase"] = "hover_confirm"
                                tr["fail_timer"] = 0.0
                                
                                # FIX: Update the 'pos' and 'time' to NOW.
                                # This resets the drift calculation so 't_elapsed' starts at 0,
                                # and the drone stays right here (where the detection is)
                                # instead of flying back to the old det_pos.
                                tr["pos"] = drone_positions[d_idx][:2].copy()
                                tr["time"] = t
                                
                                print(f"[t={t}] UAV{d_idx} entering hover-confirmation (ρ={frac_high:.2f}, mean={mean_p:.2f}). Anchoring track here.")
                                continue

                            if n_cells == 0:
                                print(f"[t={t}] UAV{d_idx} lost visual (n_cells=0). Aborting.")
                                tr["active"] = False; tr["confirmed"] = False; tr["phase"] = None
                                drone_modes[d_idx] = "explore"
                                vz_des = 2.0
                                continue

                            # -------------------------------------------------------------
                            # ALTITUDE DESCENT LOGIC (Step C addition)
                            # -------------------------------------------------------------
                            # If we see *some* evidence (not enough for success), descend gradually
                            if max_p > 0.20 and (drone_positions[d_idx][2] > MIN_ALT + 0.5):
                                # Descend slowly as probability increases
                                current_alt = drone_positions[d_idx][2]
                                target_alt = max(MIN_ALT, current_alt - DESCENT_STEP)

                                vz_des = np.clip(target_alt - current_alt, -1.5, 0.0)  # descend at 1.5 m/s
                                print(f"[t={t}] UAV{d_idx} descending for better certainty: alt={current_alt:.1f} → {target_alt:.1f}")


                            # -------------------------------------------------------------
                            # 2. ALTITUDE-BASED FAILURE (your original logic)
                            # -------------------------------------------------------------
                            z = drone_positions[d_idx][2]
                            if z <= MIN_ALT + 0.5:     # UAV already at minimum altitude
                                tr["fail_timer"] += dt

                                if tr["fail_timer"] > 8.0:
                                    print(f"[t={t}] UAV{d_idx} BAD DETECTION — no victim found. Returning to IPP.")

                                    # Clear false hotspot
                                    if tr["pos_at_detection"] is not None:
                                        dp = tr["pos_at_detection"]
                                        dists = np.linalg.norm(grid_points_all - dp, axis=1)
                                        belief[dists < 40.0] = 0.0
                                        print(f"[t={t}] Cleared false hotspot near {dp}")

                                    # Reset tracking state
                                    tr["active"] = False
                                    tr["confirmed"] = False
                                    tr["phase"] = None
                                    tr["pos"] = None
                                    tr["time"] = None
                                    tr["pos_at_detection"] = None
                                    tr["fail_timer"] = 0.0

                                    # Back to exploration
                                    drone_modes[d_idx] = "explore"

                                    vx_des = vy_des = 0.0
                                    vz_des = +1.5     # climb out

                                    # Clear cone graphics
                                    if hasattr(ax_map, "cone_fill") and ax_map.cone_fill is not None:
                                        ax_map.cone_fill.remove()
                                        ax_map.cone_fill = None

                                    cone_line.set_data([], [])
                                    cone_center_dot.set_data([], [])

                                    continue

                            # -------------------------------------------------------------
                            # 3. TIMEOUT FAILURE (Step B addition)
                            # -------------------------------------------------------------
                            if (t - tr["cone_start"]) >= cone_search_timeout and max_p < 0.50:
                                print(
                                    f"[t={t}] UAV{d_idx} cone IPP timeout: no strong evidence "
                                    f"(max_p={max_p:.2f}) → abort tracking."
                                )

                                # Same cleanup as above
                                tr["active"] = False
                                tr["confirmed"] = False
                                tr["phase"] = None
                                tr["pos"] = None
                                tr["time"] = None
                                tr["pos_at_detection"] = None
                                tr["fail_timer"] = 0.0

                                drone_modes[d_idx] = "explore"

                                current_alt = drone_positions[d_idx][2]
                                nominal_alt = uav_nominal_altitudes[d_idx]
                                vz_des = np.clip(nominal_alt - current_alt, -1.0, 1.0)

                                continue

                        # If cone shape exists but no grid cells inside → fallback
                        else:
                            # Tiny cone or drift displacement → treat as timeout
                            if (t - tr["cone_start"]) >= 10.0:
                                print(f"[t={t}] UAV{d_idx} empty cone → aborting tracking")
                                tr["active"] = False
                                tr["phase"] = None
                                drone_modes[d_idx] = "explore"
                                continue

                    # If region_poly invalid → abort
                    else:
                        print(f"[t={t}] UAV{d_idx} lost cone geometry → abort tracking")
                        tr["active"] = False
                        tr["phase"] = None
                        drone_modes[d_idx] = "explore"
                        continue

                    # =============================================================
                    # 2. THE MISSING MOTION LOGIC (Insert this at the end)
                    # =============================================================
                    # If we haven't confirmed, aborted, or timed out yet, WE MUST MOVE.
                    
                    # A. Calculate the center of the drifting cone
                    t_elapsed = t - det_time
                    drift_center = det_pos + v_drift * t_elapsed
                    
                    # B. Execute IPP (with small steps)
                    # We pass 'constraint_poly=None' because this old planner doesn't support it,
                    # but we manually penalize deviation below.
                    vx_ipp, vy_ipp, vz_ipp, best_path, _, _ = plan_velocity_ipp_3D(
                        drone_positions[d_idx], drone_vels[d_idx], belief, grid_points, soft_poly,
                        fov_angle=theta_FOV, v_max=v_max, 
                        step_length=20.0,  # Small steps for fine tracking
                        pred_depth=2
                    )

                    # C. Centering Guard (The "Leash")
                    # If IPP tries to send us outside the cone (Global Exploration), pull back.
                    ipp_target_2d = best_path[-1][:2]
                    dist_from_center = np.linalg.norm(ipp_target_2d - drift_center)
                    
                    if dist_from_center > (1.2 *radius_R):
                        # IPP is trying to run away. Force "Drift Following".
                        vec_to_center = drift_center - drone_positions[d_idx][:2]
                        dist_error = np.linalg.norm(vec_to_center)
                        
                        # P-Controller towards center
                        dir_u = vec_to_center / (dist_error + 1e-6)
                        kp = 0.5
                        
                        # V_cmd = Correction + Feedforward (Match Drift)
                        vx_des = dir_u[0] * kp * dist_error + v_drift[0]
                        vy_des = dir_u[1] * kp * dist_error + v_drift[1]
                        
                        # Keep the vertical velocity calculated by your Descent Logic
                        # (If your descent logic didn't set vz_des, default to 0)
                        if 'vz_des' not in locals(): vz_des = 0.0
                        
                    else:
                        # IPP is behaving well (staying near target). Use it.
                        # Add Drift Feedforward to help IPP fight wind
                        vx_des = vx_ipp + v_drift[0] * 0.5 
                        vy_des = vy_ipp + v_drift[1] * 0.5
                        
                        # IMPORTANT: Overwrite vz from IPP with your smart descent logic
                        # (Your descent logic in the block above set 'vz_des' based on certainty. Keep that.)
                        # Only use IPP's vz if you didn't calculate one.


                                

                elif tr["phase"] == "hover_confirm":
                    det_pos = tr["pos"]
                    det_time = tr["time"]
                        # --- STEP C: Initialize hover timer if first entry ---
                    if "hover_start" not in tr or tr["hover_start"] is None:
                        tr["hover_start"] = t


                    # --- recompute current cone region based on t ---
                    region_poly, center_R, radius_R = tracking_region(
                            t_now=t,
                            detection_pos=track[d_idx]["pos"],
                            detection_time=track[d_idx]["time"],
                            v_drift=v_drift
                    )


                    # 1) Follow drifting cone center
                    t_elapsed = t - det_time
                    drift_center = det_pos + v_drift * t_elapsed

                    vec = drift_center - drone_positions[d_idx][:2]
                    dist = np.linalg.norm(vec)

                    if dist > 5.0:
                        dir_unit = vec / (dist + 1e-6)
                        vx_des = dir_unit[0] * 3.0
                        vy_des = dir_unit[1] * 3.0
                    else:
                        vx_des = vy_des = 0.0

                    # 2) Descend gently
                    current_alt = drone_positions[d_idx][2]
                    descent_alt = 20.0
                    vz_des = -1.5 if current_alt > descent_alt else 0.0

                    # 3) Compute FOV belief stats
                    vis_mask_inside, _ = visible_cells_at(
                        drone_positions[d_idx],
                        grid_points
                    )

                    # ... [Inside hover_confirm, after calculating stats] ...
                    
                    # 4) ABORT CONDITION: Lost Signal
                    # If we are descending but the probability has collapsed, ABORT immediately.
                    if current_alt < 70.0 and mean_p < 0.20 and max_p < 0.40:
                        print(f"[t={t}] UAV{d_idx} signal lost during descent (mean_p={mean_p:.2f}) → ABORT.")
                        
                        # Reset tracking state
                        tr["active"] = False
                        tr["confirmed"] = False
                        tr["pos"] = None
                        tr["time"] = None
                        tr["phase"] = None
                        tr["hover_start"] = None
                        
                        drone_modes[d_idx] = "explore"
                        
                        # Climb out immediately
                        nominal_alt = uav_nominal_altitudes[d_idx]
                        vz_des = np.clip(nominal_alt - current_alt, 0.5, 2.0)
                        
                        # IMPORTANT: Don't forget to zero horizontal velocity or let IPP handle it next frame
                        vx_des, vy_des = 0.0, 0.0
                        
                        continue

                    region_mask_inside = np.array([
                        region_poly.contains(Point(p)) for p in grid_points
                    ]) if (region_poly is not None and not region_poly.is_empty) else np.zeros(len(grid_points), dtype=bool)

                    hover_mask = vis_mask_inside & region_mask_inside
                    n_cells = int(np.sum(hover_mask))

                    frac_high = 0.0
                    mean_p = 0.0
                    p_vals = np.array([])

                    if n_cells > 0:
                        p_vals = belief[hover_mask]
                        frac_high = np.mean(p_vals > confirm_pconf)
                        mean_p = np.mean(p_vals)

                    print(
                        f"[DEBUG hover t={t}] alt={current_alt:.1f}  "
                        f"n_cells={n_cells}  frac_high={frac_high:.3f}  mean_p={mean_p:.3f}  "
                        f"(rho_th={rho_th}, confirm_pconf={confirm_pconf}, mean_thresh={mean_thresh})"
                    )
                    if 0 < n_cells < 10:
                        print(f"[WARN hover t={t}] hover_mask too small (n_cells={n_cells}) — check FOV/grid settings.")

                    print(
                        f"[DEBUG hover t={t}] conditions: "
                        f"(frac_high >= rho_th)={frac_high >= rho_th}  "
                        f"(mean_p >= mean_thresh)={mean_p >= mean_thresh}"
                    )

                    required_confirmation_alt = 22.0
                    alt_ready = (current_alt <= required_confirmation_alt)

                    if (
                        alt_ready and
                        (
                            frac_high >= rho_th or
                            mean_p >= mean_thresh or
                            (len(p_vals) > 0 and np.max(p_vals) >= 0.55)
                        )
                    ):
                        print(f"[t={t}] UAV{d_idx} confirmed victim → resuming exploration.")

                        # (A) CLEAR VICTIM REGION FROM BELIEF
                        belief = clear_confirmed_region(
                            belief,
                            grid_points,
                            det_pos,
                            v_drift,
                            major_axis=120.0,
                            minor_axis=60.0,
                            decay=0.01
                        )

                        # extra hard clear around the detection position
                        clear_r = 50.0
                        cp = det_pos
                        diff = grid_points - cp
                        mask = (diff[:, 0]**2 + diff[:, 1]**2) <= clear_r**2
                        belief[mask] *= 0.05

                        print(f"[t={t}] Fully cleared victim region and renormalized belief.")

                        # (B) Reset this UAV's tracking state
                        tr["active"] = False
                        tr["confirmed"] = True
                        tr["pos"] = None
                        tr["time"] = None
                        tr["phase"] = None
                        tr["hover_start"] = None


                        drone_modes[d_idx] = "explore"

                        nominal_alt = uav_nominal_altitudes[d_idx]
                        vz_des = np.clip(nominal_alt - current_alt, -2.0, 2.0)

                        # ---------------------------------------------------------
                    # STEP C: hover-confirm timeout (10 seconds at min altitude)
                    # Only do this when we have NOT confirmed yet.
                    # ---------------------------------------------------------
                    if (
                        alt_ready
                        and tr.get("hover_start") is not None
                        and (t - tr["hover_start"]) > 10.0
                    ):
                        print(f"[t={t}] UAV{d_idx} hover-confirm timeout → aborting and returning to IPP.")

                        # Reset tracking state
                        tr["active"] = False
                        tr["confirmed"] = False
                        tr["pos"] = None
                        tr["time"] = None
                        tr["phase"] = None
                        tr["hover_start"] = None

                        drone_modes[d_idx] = "explore"

                        nominal_alt = uav_nominal_altitudes[d_idx]
                        vz_des = np.clip(nominal_alt - current_alt, -2.0, 2.0)

                        continue



                else:
                    # Some unexpected phase; just do nothing
                    vx_des = vy_des = vz_des = 0.0

            # --- Fallback: any other mode → regular IPP ---
            else:
                vx_des, vy_des, vz_des, *_ = plan_velocity_ipp_3D(
                    drone_positions[d_idx], belief, grid_points, soft_poly,
                    fov_angle=theta_FOV, v_max=v_max, n_directions=16, step_length=40.0,
                    I_scale=I_scale, E_scale=E_scale, alpha_d=alpha_d,
                    buffer=buffer, pred_depth=pred_depth
                )

            # Smooth heading change (unchanged)
            speed = np.linalg.norm([vx_des, vy_des])
            if speed < 1e-3:
                vx_smooth, vy_smooth = 0.0, 0.0
            else:
                current_heading = np.arctan2(drone_vels[d_idx][1], drone_vels[d_idx][0])
                desired_heading = np.arctan2(vy_des, vx_des)
                delta_heading = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                delta_heading = np.clip(delta_heading, -max_dheading * dt_step, max_dheading * dt_step)
                new_heading = current_heading + delta_heading
                vx_smooth = speed * np.cos(new_heading)
                vy_smooth = speed * np.sin(new_heading)

            drone_vels[d_idx][:2] = [vx_smooth, vy_smooth]
            drone_vels[d_idx][2] = vz_des

        # ------------------------------------------
        # 7. Boundary correction
        # ------------------------------------------
        for d_idx in range(num_drones):
            next_pos = drone_positions[d_idx] + drone_vels[d_idx] * dt_step
            point_next = Point(next_pos[0], next_pos[1])
            if not safe_poly.contains(point_next):
                nearest = np.array(safe_poly.exterior.interpolate(
                    safe_poly.exterior.project(point_next)
                ).coords[0])
                direction = nearest - drone_positions[d_idx][:2]
                norm = np.linalg.norm(direction)
                if norm > 1e-3:
                    vx_corr, vy_corr = direction / norm * min(norm / dt_step, v_max)
                else:
                    vx_corr, vy_corr = 0.0, 0.0
                drone_vels[d_idx][:2] = [vx_corr, vy_corr]

        # ------------------------------------------
        # 8. Advance UAV positions
        # ------------------------------------------
        for d_idx in range(num_drones):
            drone_positions[d_idx] += drone_vels[d_idx] * dt_step

        # ============================
        # Energy consumption update
        # ============================
        # (Implements the preliminary energy model from Section 7.3.3)
        P_hover = 920.0         # W hover
        P_cruise = 300.0        # W fixed-wing cruise
        P_transition = 1500.0   # W transition
        transition_time = 5.0   # s
        battery_capacity = 1000.0 * 60  # 1000 W·min = 16.7 Wh, scaled
        E_scale = E_max / battery_capacity

        for d_idx in range(num_drones):
            v = np.linalg.norm(drone_vels[d_idx][:2])

            # --- Determine regime based on speed ---
            if v < 2.0:  # hovering / low-speed
                P = P_hover
            elif 2.0 <= v < 10.0:  # transition regime
                P = P_transition
            else:  # fixed-wing / cruise
                P = P_cruise + 0.05 * v**2  # mild drag penalty

            # --- Tracking mode energy penalty (hover-confirm phase) ---

            if drone_modes[d_idx] == "track" and track[d_idx]["phase"] == "hover_confirm":
                P *= 1.2 # 20 % more vertical thrust


            # --- Update remaining energy ---
            E_rem[d_idx] -= P * dt_step * E_scale
            E_rem[d_idx] = max(0, E_rem[d_idx])

            # --- Log to history ---
            energy_history[d_idx] = np.append(energy_history[d_idx], E_rem[d_idx])

        energy_time.append(t)


        # ============================
        # Visualization update (each timestep)
        # ============================

        # --- Update belief raster (occupancy map) ---
        raster_belief[:] = np.nan
        raster_belief[inside_mask_full] = belief
        sc.set_array(raster_belief)

        # --- Update UAV and victim positions on the map ---
        for d_idx in range(num_drones):
            drone_plots[d_idx].set_data(drone_positions[d_idx][0], drone_positions[d_idx][1])
        victim_plot.set_offsets(victims[:, :2])
        # --- Update detection point marker ---
        if detection_active and detection_pos is not None:
            detection_marker.set_data(detection_pos[0], detection_pos[1])
        else:
            detection_marker.set_data([], [])

        # --- Update drift cone visualization (only for active trackers) ---

        # Find tracker UAVs with valid detection info
        # --- Plot cone regions for all active trackers ---
        for d_idx in range(num_drones):
            tr = track[d_idx]

            if tr["active"] and not tr["confirmed"]:
                region_poly, center_R, _ = tracking_region(
                    t_now=t,
                    detection_pos=tr["pos"],
                    detection_time=tr["time"],
                    v_drift=v_drift
                )

                if not region_poly.is_empty:
                    xR, yR = region_poly.exterior.xy

                    # Update existing pre-allocated artists
                    cone_lines[d_idx].set_data(xR, yR)
                    cone_centers[d_idx].set_data(center_R[0], center_R[1])
                    cone_fills[d_idx].set_xy(np.column_stack([xR, yR]))
                else:
                    # No region — hide
                    cone_lines[d_idx].set_data([], [])
                    cone_centers[d_idx].set_data([], [])
                    cone_fills[d_idx].set_xy([[0,0],[0,0],[0,0]])

            else:
                # UAV is not tracking → hide cone
                cone_lines[d_idx].set_data([], [])
                cone_centers[d_idx].set_data([], [])
                cone_fills[d_idx].set_xy([[0,0],[0,0],[0,0]])



        # --- Draw map updates ---
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # ============================
        # Velocity plot update
        # ============================
        vel_time.append(t)
        for d_idx in range(num_drones):
            vx, vy, vz = drone_vels[d_idx]
            vel_history[d_idx] = np.vstack((vel_history[d_idx], [vx, vy, vz]))

            # Update lines (vx, vy, vz)
            vel_lines[d_idx][0].set_data(np.arange(len(vel_history[d_idx])), vel_history[d_idx][:, 0])
            vel_lines[d_idx][1].set_data(np.arange(len(vel_history[d_idx])), vel_history[d_idx][:, 1])
            vel_lines[d_idx][2].set_data(np.arange(len(vel_history[d_idx])), vel_history[d_idx][:, 2])

        # Adjust x-limits dynamically
        for axv in axs_vel:
            axv.relim()
            axv.autoscale_view()

        # --- Draw velocity figure updates ---
        fig_vel.canvas.draw_idle()
        fig_vel.canvas.flush_events()

        # ============================
        # Energy plot update
        # ============================

        for d_idx in range(num_drones):
            energy_lines[d_idx].set_data(np.arange(len(energy_history[d_idx])), energy_history[d_idx])

        ax_energy.relim()
        ax_energy.autoscale_view()

        fig_energy.canvas.draw_idle()
        fig_energy.canvas.flush_events()


        # --- Small delay for real-time smoothness ---
        plt.pause(0.01)


    plt.ioff()
    plt.show()