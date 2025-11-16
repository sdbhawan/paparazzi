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
# from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon


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
# grid_x = np.arange(x_min, x_max, grid_resolution)
# grid_y = np.arange(y_min, y_max, grid_resolution)

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


# def expected_info_gain(path, belief, grid_points, fov_angle=theta_FOV, pred_depth=3):
#     pred_belief = belief.copy()
#     total_IG = 0.0
#     vis_mask_total = np.zeros(len(grid_points), dtype=bool)
#     ent = cell_entropy_map(pred_belief)

#     for i in range(len(path) - 1):
#         if i >= pred_depth:
#             break
#         p0, p1 = path[i], path[i + 1]
#         seg_len = np.linalg.norm(p1[:2] - p0[:2])
#         n_samples = max(2, int(np.ceil(seg_len / 20.0)))
#         xs = np.linspace(p0[0], p1[0], n_samples)
#         ys = np.linspace(p0[1], p1[1], n_samples)
#         zs = np.linspace(p0[2], p1[2], n_samples)

#         for x, y, z in zip(xs, ys, zs):
#             mask, p_hit = visible_cells_at(np.array([x, y, z]), grid_points)
#             delta_H = ent[mask] * p_hit[mask]
#             total_IG += np.sum(delta_H)
#             pred_belief[mask] = pred_belief[mask] * (1 - p_hit[mask]) + 0.5 * p_hit[mask]
#             vis_mask_total |= mask

#         ent = cell_entropy_map(pred_belief)

#     return total_IG, vis_mask_total

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


# def energy_of_path(path):
#     p0, p1 = path[0], path[-1]
#     dist = np.linalg.norm(p1[:2] - p0[:2])
#     dz = abs(p1[2] - p0[2])
#     speed = dist / 1.0
#     if speed < 12:
#         E = 920.0
#     elif speed < 17:
#         E = 1500.0
#     else:
#         E = 300.0
#     E += 5.0 * dz
#     return E

def energy_of_path(path):
    """
    Approximate energy cost of a path segment.

    - Horizontal motion flown at cruise:
        v_cruise ≈ 19 m/s, P_cruise ≈ 300 W
    - Vertical motion done in hover:
        P_hover ≈ 920 W, climb ≈ 2 m/s, descend ≈ 3 m/s

    This makes energy scale with distance and altitude change,
    instead of being (almost) constant for all paths.
    """
    P_hover = 920.0      # W
    P_cruise = 300.0     # W
    v_cruise = 19.0      # m/s (fixed-wing cruise)

    v_climb = 2.0        # m/s (hover climb)
    v_desc = 3.0         # m/s (hover descent)

    p0, p1 = path[0], path[-1]

    # horizontal distance and vertical difference
    dist_xy = np.linalg.norm(p1[:2] - p0[:2])
    dz = p1[2] - p0[2]

    # --- horizontal energy (cruise) ---
    if dist_xy > 1e-3:
        dt_horiz = dist_xy / v_cruise
        E_horiz = P_cruise * dt_horiz
    else:
        E_horiz = 0.0

    # --- vertical energy (hover) ---
    if dz > 0:
        # climbing
        dt_vert = dz / max(v_climb, 1e-3)
        E_vert = P_hover * dt_vert
    elif dz < 0:
        # descending
        dt_vert = (-dz) / max(v_desc, 1e-3)
        E_vert = P_hover * dt_vert
    else:
        E_vert = 0.0

    return E_horiz + E_vert


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

def expected_detection(path, belief, grid_points, fov_angle, pred_depth=3):
    """
    Computes DET = sum( p * p_hit ) along the path, similar to entropy-based IG.
    """
    DET = 0.0

    p0, p1 = path
    seg = int(max(3, np.ceil(np.linalg.norm(p1[:2] - p0[:2]) / 20.0)))
    xs = np.linspace(p0[0], p1[0], seg)
    ys = np.linspace(p0[1], p1[1], seg)
    zs = np.linspace(p0[2], p1[2], seg)

    for x, y, z in zip(xs, ys, zs):
        pos = np.array([x, y, z])
        vis_mask, p_hit = visible_cells_at(pos, grid_points)
        DET += np.sum(belief[vis_mask] * p_hit[vis_mask])

    return DET

def plan_velocity_ipp_3D(drone_pos, belief, grid_points, soft_poly, 
                         fov_angle=theta_FOV, v_max=12.0, n_directions=16, 
                         step_length=40.0, altitude_candidates=[30,50,70],
                         I_scale=100.0, E_scale=100.0, alpha_d=0.5, buffer=0.0,
                         pred_depth=3, lam=0.0, MAP_bonus=False):

    # shrink polygon by buffer
    if buffer>0.0:
        poly = Polygon(soft_poly).buffer(-buffer)
    else:
        poly = Polygon(soft_poly)

    cx, cy, cz = drone_pos
    angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    candidates = []

    # candidate generation
    for a in angles:
        dx, dy = np.cos(a), np.sin(a)
        for alt in altitude_candidates:
            end = np.array([cx + dx*step_length, cy + dy*step_length, alt])
            if poly.contains(Point(end[0], end[1])):
                candidates.append([np.array([cx, cy, cz]), end])

    best_J, best_path, best_mask = -np.inf, None, None

    # -----------------------
    # SCORING LOOP (patched)
    # -----------------------
    for path in candidates:

        # information gain (existing)
        I_p, vis_mask = expected_info_gain(
            path, belief, grid_points, 
            fov_angle=fov_angle, pred_depth=pred_depth
        )

        # exploitation utility (new)
        DET_p = expected_detection(
            path, belief, grid_points, fov_angle=fov_angle, pred_depth=pred_depth
        )

        # energy cost (existing)
        E_p = energy_of_path(path)

        # combined exploration + exploitation reward
        J = (1.0 - lam) * (I_p / I_scale) \
            + lam * (DET_p / I_scale) \
            - alpha_d * (E_p / E_scale)

        # optional MAP-cell bonus
        if MAP_bonus:
            # find MAP cell inside supplied grid_points region
            max_idx = np.argmax(belief)
            MAP_cell = grid_points[max_idx]
            # distance from candidate endpoint
            dist = np.linalg.norm(path[-1][:2] - MAP_cell)
            bonus = 0.1 * np.exp(-0.02 * dist)
            J += bonus

        if J > best_J:
            best_J = J
            best_path = path
            best_mask = vis_mask

    # safety
    if best_path is None:
        return 0.0, 0.0, 0.0, None, None, 0.0

    # compute velocity along best path (unchanged)
    p0, p1 = best_path[0], best_path[-1]
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    if dist < 1e-3:
        return 0.0, 0.0, 0.0, best_path, best_mask, best_J

    travel_fraction = min(1.0, 1.0/max(dist/v_max,1e-6))
    step_vec = vec * travel_fraction
    vx, vy, vz = step_vec

    return float(vx), float(vy), float(vz), best_path, best_mask, best_J

def tracking_region(t_now, detection_pos, detection_time, v_drift,
                    base_radius=60.0, drift_scale=0.3, cone_angle_deg=90):
    """
    Creates a drift cone region aligned with v_drift.

    base_radius  : initial cone radius at detection time [m]
    drift_scale  : how fast the cone radius grows [m/s]
    """

    dt = max(t_now - detection_time, 0.0)

    # Drifted center
    drift_vec = v_drift * dt
    center_R = detection_pos + drift_vec

    # Slower expansion, larger initial area
    radius_R = base_radius + drift_scale * (dt ** 1.1)


    theta = np.arctan2(v_drift[1], v_drift[0])
    half_angle = np.deg2rad(cone_angle_deg / 2)

    angles = np.linspace(-half_angle, half_angle, 50)
    x_arc = radius_R * np.cos(angles)
    y_arc = radius_R * np.sin(angles)
    cone_points = np.vstack(([0, 0], np.column_stack([x_arc, y_arc]), [0, 0]))

    cone_poly = Polygon(cone_points)
    cone_poly = rotate(cone_poly, np.degrees(theta), origin=(0, 0))
    cone_poly = translate(cone_poly, detection_pos[0], detection_pos[1])

    return cone_poly, center_R, radius_R


def update_cone(
        detect_pos,
        v_drift,
        detect_time,
        t_now,
        map_xy,
        initial_radius=50.0,
        growth_rate=0.3,
        sigma_scale=0.5
    ):
    """
    Pure procedural cone update.
    Returns: cone_center, cone_prob, frac_high, mean_p, max_p
    """

    # --- time since detection ---
    dt_elapsed = t_now - detect_time

    # --- predicted victim drift position ---
    cone_center = detect_pos + v_drift * dt_elapsed

    # --- cone radius expanding over time ---
    radius = initial_radius + growth_rate * dt_elapsed

    # --- distance of each grid cell to cone center ---
    dist = np.linalg.norm(map_xy - cone_center, axis=-1)

    # --- Gaussian weights ---
    sigma = radius * sigma_scale
    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))

    # Normalize to max = 1
    weights /= (weights.max() + 1e-9)

    # Limit cone to its radius
    cone_prob = np.zeros_like(weights)
    mask = dist <= radius
    cone_prob[mask] = weights[mask]

    # Metrics for confirm-check
    frac_high = np.mean(cone_prob >= 0.5)
    mean_p    = np.mean(cone_prob)
    max_p     = np.max(cone_prob)

    return cone_center, cone_prob, frac_high, mean_p, max_p

def hover_confirm_step(
        drone_pos,
        cone_center,
        frac_high,
        mean_p,
        rho_th=0.30,
        mean_thresh=0.30
    ):
    """
    Procedural hover-confirm.
    Returns vx_des, vy_des, confirmed_flag
    """

    offset = cone_center - drone_pos[:2]
    distance = np.linalg.norm(offset)

    # Move toward predicted victim location
    if distance > 5.0:
        direction = offset / (distance + 1e-9)
        vx = direction[0] * 5.0
        vy = direction[1] * 5.0
    else:
        vx = 0.0
        vy = 0.0

    # Confirmation conditions
    cond1 = (frac_high >= rho_th)
    cond2 = (mean_p   >= mean_thresh)

    confirmed = cond1 and cond2

    return vx, vy, confirmed


def cone_belief_update(
    belief, grid_points, region_poly, detection_pos, v_drift,
    t_now, detection_time,
    p0=0.5, base_sigma_long=200.0, base_sigma_lat=120.0,
    growth_long=25.0, growth_lat=15.0, decay_time=400.0,
    blend=0.7
):
    """
    Hybrid drift-cone belief update.
    - Keeps anisotropic diffusion shape (Eq. 7.14–7.17 in the thesis).
    - Smooth visual behavior on a discrete grid.
    - Preserves probability mass locally (inside the cone), not globally.
    """
    if region_poly is None or region_poly.is_empty:
        return belief

    dt = max(t_now - detection_time, 0.0)
    new_belief = belief.copy()

    # Drift-aligned basis
    drift_hat = v_drift / np.linalg.norm(v_drift)
    perp_hat = np.array([-drift_hat[1], drift_hat[0]])

    # Drifted Gaussian center
    center_shift = detection_pos + v_drift * dt

    # Diffusion spread (√t scaling)
    sigma_long = base_sigma_long + growth_long * np.sqrt(dt + 1.0)
    sigma_lat  = base_sigma_lat  + growth_lat  * np.sqrt(dt + 1.0)

    # Decay of peak probability with time
    p_peak = p0 * np.exp(-dt / decay_time)

    # Precompute mask for region
    mask_cone = np.array([region_poly.contains(Point(p)) for p in grid_points])

    for i, p in enumerate(grid_points[mask_cone]):
        vec = p - center_shift
        d_along = np.dot(vec, drift_hat)
        d_cross = np.dot(vec, perp_hat)

        # Anisotropic Gaussian kernel (no strict normalization)
        g = np.exp(-0.5 * ((d_along / sigma_long) ** 2 + (d_cross / sigma_lat) ** 2))
        p_val = p_peak * g

        idx = np.where(mask_cone)[0][i]
        new_belief[idx] = blend * belief[idx] + (1 - blend) * p_val

    # Normalize locally within cone region to conserve probability mass
    total_prev = np.sum(belief[mask_cone])
    total_new  = np.sum(new_belief[mask_cone])
    if total_new > 1e-9:
        scale_factor = total_prev / total_new
        new_belief[mask_cone] *= scale_factor

    return np.clip(new_belief, 0.0, 1.0)

def check_victim_confirmed(belief, grid_points, victim_pos, confirm_radius=25.0, confirm_threshold=0.4):
    d2 = np.sum((grid_points - victim_pos[:2])**2, axis=1)
    mask = d2 <= confirm_radius**2
    if np.mean(belief[mask]) < confirm_threshold:
        return True
    return False

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



confirm_pconf = 0.25
rho_th_fov = 0.1
mean_thresh = 0.30


# --- Build grid and inside-polygon mask ---
# --- Build grid and inside-polygon mask ---
# grid_points_all = np.array([[x, y] for x in grid_x for y in grid_y])
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
# victim_signal_full = np.zeros(len(grid_points_all))
victims = np.array([[50,-200],[150,-350]], dtype=float)
# for v in victims:
#     d2 = np.sum((grid_points_all - v[:2])**2, axis=1)
#     victim_signal_full[d2 <= FOV_radius**2] = 1.0

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
detection_active = False
detection_confirmed = False
detection_pos = None          # np.array([x, y])
detection_time = None         # int (time step)
tracker_idx = None            # which UAV is tracking
tracker_phase = None          # 'to_detection', 'cone_tracking', or 'hover_confirm'
cooldown_until = np.zeros(num_drones)
COOLDOWN_TIME = 30  # seconds of detection lockout


for t in range(300):
    # ------------------------------------------
    # 0. Dynamic mapping: propagate belief (drift model)
    # ------------------------------------------
    belief = propagate_belief_vectorized(belief, grid_points, v_drift, dt_step, sigma=1.0)

    # ------------------------------------------
    # 1. Victim motion (drift)
    # ------------------------------------------
    victims[:, :2] += v_drift * dt_step

    # ------------------------------------------
    # 2. Victim signal
    # ------------------------------------------
    # victim_signal_full = np.zeros(len(grid_points_all))
    # for v in victims:
    #     d2 = np.sum((grid_points_all - v[:2]) ** 2, axis=1)
    #     victim_signal_full[d2 <= FOV_radius ** 2] = 1.0
    # victim_signal = victim_signal_full[inside_mask_full]

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
    # print("grid_points_all:", grid_points_all.shape)
    # print("inside_mask_full:", inside_mask_full.shape, "sum:", np.sum(inside_mask_full))
    # print("grid_points:", grid_points.shape)



    # ------------------------------------------
    # 3. FOV overlap counting with altitude-dependent p_hit
    # ------------------------------------------
    # update_weight = np.zeros(len(grid_points), dtype=float)
    update_weight = np.zeros(len(belief), dtype=float)


    for d_idx in range(num_drones):
        vis_mask_all, p_hit_all = visible_cells_at(drone_positions[d_idx], grid_points_all)

        vis_mask_inside = vis_mask_all[inside_mask_full]       # mask over grid_points
        p_hit_inside = p_hit_all[inside_mask_full]             # probabilities over grid_points
        # print("len(update_weight) =", len(update_weight))
        # print("vis_mask_inside shape =", vis_mask_inside.shape)
        # print("p_hit_inside shape =", p_hit_inside.shape)


        # update_weight[vis_mask_inside] += p_hit_inside
        update_weight[vis_mask_inside] += p_hit_inside[vis_mask_inside]

        # print("vis_mask_all:", vis_mask_all.shape)
        # print("vis_mask_inside:", vis_mask_inside.shape)
        # print("p_hit_inside:", p_hit_inside.shape)
        # print("update_weight:", update_weight.shape)



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
    # ------------------------------------------
    # 5. Centralized detection & tracker assignment
    # ------------------------------------------
    if not detection_active:

        # ----------------------------------------------------------
        # DETECTION ELIGIBILITY CHECK:
        # Only allow detection if at least one drone is:
        #   - in "explore" mode
        #   - NOT in cooldown
        # ----------------------------------------------------------
        eligible_explorers = [
            d_idx
            for d_idx, mode in enumerate(drone_modes)
            if mode == "explore" and t >= cooldown_until[d_idx]
        ]

        if len(eligible_explorers) == 0:
            # No UAV available to handle a new detection
            continue

        # ------------------------------------------
        # (A) Evaluate belief map for detection trigger
        # ------------------------------------------
        max_p = np.max(belief)
        if max_p > pth_detect:

            # MAP index (downwind lobe)
            det_idx_inside = int(np.argmax(belief))
            detection_pos_raw = grid_points[det_idx_inside]

            # Corrected position using drift-aligned centroid
            detection_pos = drift_aligned_centroid(
                belief,
                grid_points,
                det_idx_inside,
                v_drift,
                major=150.0,
                minor=80.0
            )

            print(f"raw detection_pos = {detection_pos_raw}")
            print(f"centroid-corrected detection_pos = {detection_pos}")

            print("\n=== DETECTION TRIGGERED ===")
            print("belief max =", max_p)
            print("detection_pos =", detection_pos)
            print("argmax grid point =", grid_points[np.argmax(belief)])
            print("victim positions:\n", victims)
            print("==========================\n")

            detection_time = t
            detection_active = True
            detection_confirmed = False

            # ------------------------------------------
            # (B) Select best tracker UAV
            # ------------------------------------------
            best_d, best_cost = None, np.inf

            for d_idx in eligible_explorers:

                r_vec = detection_pos - drone_positions[d_idx][:2]
                dist = np.linalg.norm(r_vec)
                r_hat = np.zeros(2) if dist < 1e-6 else r_vec / dist
                Deff = dist + gamma_wind * abs(np.dot(v_wind[:2], r_hat))

                if Deff < best_cost:
                    best_cost, best_d = Deff, d_idx

            if best_d is not None:
                tracker_idx = best_d
                drone_modes[tracker_idx] = "track"
                tracker_phase = "to_detection"
                print(f"[t={t}] Detection triggered at {detection_pos}, assigning UAV{tracker_idx} as tracker.")


    # ------------------------------------------
    # 6. Plan next velocities
    # ------------------------------------------
    for d_idx in range(num_drones):

        # --- Exploration / normal IPP mode ---
        if drone_modes[d_idx] == "explore":
            # 1) Plan horizontal motion normally
            vx_des, vy_des, vz_des, *_ = plan_velocity_ipp_3D(
                drone_positions[d_idx], belief, grid_points, soft_poly,
                fov_angle=theta_FOV, v_max=v_max, n_directions=16,
                step_length=40.0, I_scale=I_scale, E_scale=E_scale,
                alpha_d=alpha_d, buffer=buffer, pred_depth=pred_depth
            )

            # 2) Override vertical velocity to maintain nominal altitude
            target_alt = uav_nominal_altitudes[d_idx]
            current_alt = drone_positions[d_idx][2]
            vz_hold = np.clip(target_alt - current_alt, -1.0, 1.0)

            vz_des = vz_hold


        # --- Active tracker UAV (during detection) ---
        elif (drone_modes[d_idx] == "track"
            and detection_active
            and d_idx == tracker_idx
            and not detection_confirmed):

            # === TRACKER STATE MACHINE ===
            if tracker_phase == "to_detection":
                vec = detection_pos - drone_positions[d_idx][:2]
                dist = np.linalg.norm(vec)
                print(f"[t={t}] UAV{d_idx} distance to detection={dist:.2f} m")
                if dist > 80.0:  # <- more lenient threshold
                    dir_unit = vec / dist
                    vx_des, vy_des = dir_unit * min(v_max, dist)
                    vz_des = 0.0
                else:
                    tracker_phase = "cone_tracking"
                    print(f"[t={t}] UAV{d_idx} arrived → starting cone tracking.")
                    vx_des = vy_des = vz_des = 0.0


            elif tracker_phase == "cone_tracking":
                # --- Compute and update cone belief ---
                region_poly, center_R, radius_R = tracking_region(
                    t_now=t,
                    detection_pos=detection_pos,
                    detection_time=detection_time,
                    v_drift=v_drift
                )

                belief = cone_belief_update(
                    belief, grid_points, region_poly,
                    detection_pos=detection_pos,
                    v_drift=v_drift,
                    t_now=t,
                    detection_time=detection_time
                )

                # --- Restrict planner to inside cone (not global softgeo) ---
                in_R_mask = np.array([region_poly.contains(Point(p)) for p in grid_points])
                if np.any(in_R_mask):
                    grid_R = grid_points[in_R_mask]
                    belief_R = belief[in_R_mask]

                    # Perform IPP only within this local region
                    vx_des, vy_des, vz_des, best_path, best_mask, Jval = plan_velocity_ipp_3D(
                                drone_positions[d_idx],
                                belief_R, grid_R, region_poly,
                                fov_angle=theta_FOV,
                                v_max=10.0,
                                n_directions=16,
                                step_length=40.0,
                                I_scale=I_scale,
                                E_scale=E_scale,
                                alpha_d=alpha_d,
                                buffer=0.0,
                                pred_depth=pred_depth,
                                lam=0.7,            # exploitation ON
                                MAP_bonus=True      # gentle pull to MAP cell
                        )


                    # Keep UAV inside cone boundary (safety constraint)
                    uav_point = Point(drone_positions[d_idx][0], drone_positions[d_idx][1])
                    if not region_poly.contains(uav_point):
                        vec_back = center_R - drone_positions[d_idx][:2]
                        dist_back = np.linalg.norm(vec_back)
                        if dist_back > 1e-3:
                            dir_back = vec_back / dist_back
                            speed_back = min(8.0, dist_back)
                            vx_des, vy_des = dir_back * speed_back
                            vz_des = 0.0
                else:
                    vx_des = vy_des = vz_des = 0.0


                # --- Check if high-probability region achieved (Eq. 7.40) ---
                # Evaluate directly over the drift cone, not just the UAV FOV
                if region_poly is not None and not region_poly.is_empty:
                    in_R_mask = np.array([region_poly.contains(Point(p)) for p in grid_points])
                    if np.any(in_R_mask):
                        p_vals = belief[in_R_mask]
                        frac_high = np.mean(p_vals > pconf)
                        mean_p = np.mean(p_vals)
                        max_p = np.max(p_vals)

                        # Diagnostic output to monitor evolution
                        print(f"[t={t}] UAV{d_idx} cone stats: ρ={frac_high:.3f}, mean_p={mean_p:.3f}, max_p={max_p:.3f}")

                        # Hover-confirm condition (Eq. 7.40)
                        if (
                            frac_high >= rho_th            # original condition
                            or mean_p >= 0.30              # NEW: real cone concentration
                            or max_p >= 0.55               # NEW: UAV is over victim
                        ):
                            tracker_phase = "hover_confirm"
                            print(f"[t={t}] UAV{d_idx} entering hover-confirmation (ρ={frac_high:.2f}, mean_p={mean_p:.2f}, max_p={max_p:.2f}).")


            elif tracker_phase == "hover_confirm":

                # --- recompute current cone region based on t ---
                region_poly, center_R, radius_R = tracking_region(
                    t_now=t,
                    detection_pos=detection_pos,
                    detection_time=detection_time,
                    v_drift=v_drift
                )

                # ===========================
                # 1) Follow the drifting cone center horizontally
                # ===========================
                t_elapsed = t - detection_time
                drift_center = detection_pos + v_drift * t_elapsed

                vec = drift_center - drone_positions[d_idx][:2]
                dist = np.linalg.norm(vec)

                if dist > 5.0:   # follow the drift until within 5m
                    dir_unit = vec / (dist + 1e-6)
                    vx_des = dir_unit[0] * 3.0      # small slow tracking speed
                    vy_des = dir_unit[1] * 3.0
                else:
                    vx_des = vy_des = 0.0           # once close enough, hover

                # ===========================
                # 2) Descend gently for better sensing
                # ===========================
                current_alt = drone_positions[d_idx][2]
                descent_alt = 20.0
                vz_des = -1.5 if current_alt > descent_alt else 0.0

                                # ===========================
                # 3) Compute FOV belief stats (fixed mask)
                # ===========================
                # FOV on the *inside* grid only
                vis_mask_inside, _ = visible_cells_at(
                    drone_positions[d_idx],
                    grid_points
                )

                # Cone region on the inside grid
                region_mask_inside = np.array([
                    region_poly.contains(Point(p)) for p in grid_points
                ])

                # Cells that are both in the cone and in the current FOV
                hover_mask = vis_mask_inside & region_mask_inside
                n_cells = int(np.sum(hover_mask))

                frac_high = 0.0
                mean_p = 0.0
                p_vals = np.array([])

                if n_cells > 0:
                    p_vals = belief[hover_mask]
                    frac_high = np.mean(p_vals > confirm_pconf)
                    mean_p = np.mean(p_vals)

                # Extra diagnostics on footprint quality
                print(
                    f"[DEBUG hover t={t}] alt={current_alt:.1f}  "
                    f"n_cells={n_cells}  frac_high={frac_high:.3f}  mean_p={mean_p:.3f}  "
                    f"(rho_th={rho_th}, confirm_pconf={confirm_pconf}, mean_thresh={mean_thresh})"
                )
                if n_cells > 0 and n_cells < 10:
                    print(f"[WARN hover t={t}] hover_mask too small (n_cells={n_cells}) — check FOV/grid settings.")

                print(
                    f"[DEBUG hover t={t}] conditions: "
                    f"(frac_high >= rho_th)={frac_high >= rho_th}  "
                    f"(mean_p >= mean_thresh)={mean_p >= mean_thresh}"
                )

                # ===========================
                # 4) Confirmation condition
                # ===========================
                required_confirmation_alt = 22.0            # must descend to ~20 m
                alt_ready = (current_alt <= required_confirmation_alt)

                if (
                    alt_ready and                                                  # <-- NEW
                    (
                        frac_high >= rho_th or
                        mean_p >= mean_thresh or
                        (len(p_vals) > 0 and np.max(p_vals) >= 0.55)
                    )
                ):



                    print(f"[t={t}] UAV{d_idx} confirmed victim → resuming exploration.")

                    # -------------------------------------
                    # (A) CLEAR VICTIM REGION FROM BELIEF
                    # -------------------------------------
                    belief = clear_confirmed_region(
                        belief,
                        grid_points,
                        detection_pos,
                        v_drift,
                        major_axis=120.0,
                        minor_axis=60.0,
                        decay=0.01
                    )

                    # Hard clear to prevent retriggering
                    clear_r = 25.0
                    cp = detection_pos
                    diff = grid_points - cp
                    mask = (diff[:,0]**2 + diff[:,1]**2) <= clear_r**2
                    belief[mask] *= 0.05
                    # belief /= np.sum(belief)

                    print(f"[t={t}] Fully cleared victim region and renormalized belief.")


                    # -------------------------------------
                    # (B) Reset tracking state
                    # -------------------------------------
                    detection_confirmed = True
                    detection_active = False
                    tracker_phase = None
                    drone_modes[d_idx] = "explore"

                    # gentle climb back to nominal exploration altitude
                    nominal_alt = uav_nominal_altitudes[d_idx]  # <-- use the real one
                    vz_des = np.clip(nominal_alt - current_alt, -2.0, 2.0)
                    drone_vels[d_idx][2] = vz_des




        else:
            vx_des, vy_des, vz_des, *_ = plan_velocity_ipp_3D(
                drone_positions[d_idx], belief, grid_points, soft_poly,
                fov_angle=theta_FOV, v_max=v_max, n_directions=16, step_length=40.0,
                I_scale=I_scale, E_scale=E_scale, alpha_d=alpha_d,
                buffer=buffer, pred_depth=pred_depth
            )

        # Smooth heading change
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
        if drone_modes[d_idx] == "track" and tracker_phase == "hover_confirm":
            P *= 1.2  # 20% more for vertical thrust

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


    # --- Update drift cone visualization (if tracking) ---
    # if detection_active and tracker_phase in ("cone_tracking", "hover_confirm"):
    # --- Update drift cone visualization (if tracking) ---
    if detection_active:

        # Always draw the original detection point (fixed)
        det_dot.set_data(detection_pos[0], detection_pos[1])

        region_poly, center_R, radius_R = tracking_region(
            t_now=t,
            detection_pos=detection_pos,       # <-- FIXED POINT
            detection_time=detection_time,
            v_drift=v_drift
        )

        if region_poly and not region_poly.is_empty:
            xR, yR = region_poly.exterior.xy

            # Update drifting cone outline
            cone_line.set_data(xR, yR)

            # Update drifting cone center
            cone_center_dot.set_data(center_R[0], center_R[1])

            # Fade-out cone fill
            dt_since_det = max(t - detection_time, 0)
            alpha = max(0.2, 1.0 - dt_since_det / 200.0)
            fill_color = (1.0, 0.0, 0.0, alpha)

            if hasattr(ax_map, "cone_fill") and ax_map.cone_fill is not None:
                ax_map.cone_fill.remove()

            # New filled region
            ax_map.cone_fill = ax_map.fill(xR, yR, color=fill_color, zorder=1)[0]

        else:
            # No valid region
            cone_line.set_data([], [])
            cone_center_dot.set_data([], [])
            det_dot.set_data([], [])   # hide detection point
            if hasattr(ax_map, "cone_fill") and ax_map.cone_fill is not None:
                ax_map.cone_fill.remove()
                ax_map.cone_fill = None

    else:
        # No detection active
        cone_line.set_data([], [])
        cone_center_dot.set_data([], [])
        det_dot.set_data([], [])   # hide detection marker
        if hasattr(ax_map, "cone_fill") and ax_map.cone_fill is not None:
            ax_map.cone_fill.remove()
            ax_map.cone_fill = None


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

