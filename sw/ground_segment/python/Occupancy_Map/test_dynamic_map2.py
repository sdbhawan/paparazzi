import numpy as np
import matplotlib.pyplot as plt
import random
import time
from shapely.geometry import Polygon, Point
import numpy as np
import math
import os
import sys
import xml.etree.ElementTree as ET
import pymap3d as pm


USE_PPRZ = False  # set True if you want to send to Paparazzi (not configured here)
RANDOM_SEED = 112
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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

soft_poly = Polygon(softgeo_xy)

# Victims
amount_of_victims = 8
victims = []
while len(victims) < amount_of_victims:
    x = np.random.uniform(soft_poly.bounds[0], soft_poly.bounds[2])
    y = np.random.uniform(soft_poly.bounds[1], soft_poly.bounds[3])
    if soft_poly.contains(Point(x, y)):
        victims.append([x, y])
victims = np.array(victims)
detected_victims = np.zeros(len(victims), dtype=bool)

# Grid
grid_res_fine = 5.0
margin = 20.0
min_x, min_y, max_x, max_y = soft_poly.bounds[0]-margin, soft_poly.bounds[1]-margin, soft_poly.bounds[2]+margin, soft_poly.bounds[3]+margin
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res_fine),
                     np.arange(min_y, max_y, grid_res_fine))

# ---------------------------
# UAV initial state
# ---------------------------
uav_pos = np.array([20.0, 50.0], dtype=float)
uav_heading = 0.0  # radians
uav_speed = 0.0    # start with no motion
uav_mode = "quad"  # start hovering
transition_timer = 0
dt = 0.5
target_speed = 19.0
alpha_speed = 0.2
alpha_accel = 0.5
max_turn_rate_deg = 5.0
max_turn_rate = np.deg2rad(max_turn_rate_deg) * dt
max_speed_change = 1.0 * dt
detection_radius = 80.0
# look_ahead_cells = 75  # increased for smoother IPP gradient

current_heading = 0.0

smoothed_path = [uav_pos.copy()]
vx_log, vy_log, speed_log, heading_log = [], [], [], []

# Base probability map
base_prob = 0.1
prob_map_fine = np.ones_like(xv) * base_prob
prob_map_fine /= prob_map_fine.sum()

# Victim-proximity heatmap
visited_heatmap = np.zeros_like(prob_map_fine)

# ---------------------------
# Helper functions
# ---------------------------
def world_to_cell(x, y):
    j = int((x - min_x) // grid_res_fine)
    i = int((y - min_y) // grid_res_fine)
    i = np.clip(i, 0, xv.shape[0]-1)
    j = np.clip(j, 0, xv.shape[1]-1)
    return i, j

def angle_wrap_rad(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def clip_to_polygon(uav_pos, polygon):
    if not polygon.contains(Point(uav_pos[0], uav_pos[1])):
        nearest = np.array(polygon.exterior.interpolate(
            polygon.exterior.project(Point(uav_pos[0], uav_pos[1]))
        ).coords[0])
        return nearest
    return uav_pos

def flight_mode(speed):
    if speed < 12:
        return "quad"
    elif speed < 17:
        return "transition"
    else:
        return "fixedwing"
    

def get_ipp_gradient(uav_pos, prob_map, look_ahead=5, last_heading=None, uniform_thresh=1e-4):
    """
    Returns a velocity vector toward high-probability cells.
    If local patch is uniform, return last heading (straight flight).
    """
    i, j = world_to_cell(uav_pos[0], uav_pos[1])
    i_min, i_max = max(0, i-look_ahead), min(prob_map.shape[0], i+look_ahead+1)
    j_min, j_max = max(0, j-look_ahead), min(prob_map.shape[1], j+look_ahead+1)
    patch = prob_map[i_min:i_max, j_min:j_max]

    if patch.size == 0:
        return np.array([0.0, 0.0])

    # Check uniformity
    if np.var(patch) < uniform_thresh and last_heading is not None:
        # Keep flying straight
        return last_heading.copy()

    # Compute gradient toward higher-probability cells
    js = np.arange(j_min, j_max)
    is_ = np.arange(i_min, i_max)
    jj, ii = np.meshgrid(js, is_)
    cx = (jj - j) * grid_res_fine
    cy = (ii - i) * grid_res_fine

    vx = np.sum(patch * cx)
    vy = np.sum(patch * cy)

    if np.linalg.norm([vx, vy]) < 1e-8:
        # Fallback if gradient is zero
        return last_heading.copy() if last_heading is not None else np.array([1.0, 0.0])
    
    return np.array([vx, vy])



def boundary_avoidance(uav_pos, polygon, turn_cells, grid_res, slowdown_factor=0.5):
    """Returns (avoidance_vector, speed_factor) to keep UAV inside polygon."""
    point = Point(uav_pos[0], uav_pos[1])
    dist_to_edge = polygon.exterior.distance(point)
    turn_distance = turn_cells * grid_res

    if dist_to_edge < turn_distance:
        nearest = np.array(polygon.exterior.interpolate(
            polygon.exterior.project(point)
        ).coords[0])
        away_vec = uav_pos - nearest
        if np.linalg.norm(away_vec) > 1e-6:
            away_vec /= np.linalg.norm(away_vec)
            strength = (1 - dist_to_edge / turn_distance)**2
            speed_scale = max(slowdown_factor, dist_to_edge / turn_distance)
            return away_vec * strength, speed_scale
    return np.array([0.0, 0.0]), 1.0




def sensor_update_continuous(prob_map, uav_pos, victims, detected_victims, xv, yv,
                             sense_radius=60.0, sigma=20.0, gain=3.0, base_prob=0.1):
    """
    Update probability map based on UAV proximity to victims.
    - prob_map: 2D probability grid
    - uav_pos: UAV [x, y]
    - victims: array of victim positions
    - detected_victims: boolean array of detected victims
    - xv, yv: meshgrid for world coords
    """
    for vi, (vx, vy) in enumerate(victims):
        if detected_victims[vi]:
            continue  # already found

        d = np.linalg.norm(uav_pos - np.array([vx, vy]))
        if d < sense_radius:
            bump = np.exp(-((xv - vx)**2 + (yv - vy)**2) / (2*sigma**2))
            strength = gain * (1 - d/sense_radius)
            prob_map += strength * bump

            # ✅ Mark victim as detected
            detected_victims[vi] = True

    # Reset to uniform if all victims are found
    if detected_victims.all():
        prob_map[:] = base_prob
    else:
        # normalize
        prob_map /= prob_map.sum() + 1e-9

    return prob_map, detected_victims

def clear_victim_region(prob_map, pos, xv, yv, radius=60.0, sigma=25.0, base_prob=0.1):
    """ Flatten probability around a detected victim with a Gaussian-shaped mask """
    bump = np.exp(-((xv - pos[0])**2 + (yv - pos[1])**2) / (2*sigma**2))
    mask = bump > np.exp(-radius**2 / (2*sigma**2))
    prob_map[mask] = base_prob
    return prob_map

def flatten_victim_region(prob_map, pos, xv, yv, radius=120.0, sigma=40.0, base_prob=1e-6):
    """Flatten probability around a detected victim so UAV won't turn back"""
    bump = np.exp(-((xv - pos[0])**2 + (yv - pos[1])**2) / (2*sigma**2))
    mask = bump > np.exp(-radius**2 / (2*sigma**2))
    prob_map[mask] = base_prob
    return prob_map

def repel_victim_region(prob_map, pos, xv, yv, detection_radius=50.0, sigma=20.0, repulsion=-0.5):
    """
    Add a negative Gaussian bump around a detected victim (repulsion),
    truncated at detection radius.
    """
    bump = np.exp(-((xv - pos[0])**2 + (yv - pos[1])**2) / (2*sigma**2))
    mask = (xv - pos[0])**2 + (yv - pos[1])**2 <= detection_radius**2
    prob_map[mask] += repulsion * bump[mask]
    return prob_map


def inject_victim_signal(prob_map, pos, xv, yv, detection_radius=50.0, sigma=20.0, bump_strength=1.0):
    """
    Add a Gaussian-shaped probability bump around a victim,
    but truncate it at the sensor detection radius.
    
    - detection_radius: max sensor range
    - sigma: smoothness of bump
    - bump_strength: how much probability to add
    """
    # Gaussian bump centered at victim
    bump = np.exp(-((xv - pos[0])**2 + (yv - pos[1])**2) / (2*sigma**2))

    # Hard cut-off at detection radius
    mask = (xv - pos[0])**2 + (yv - pos[1])**2 <= detection_radius**2

    # Apply truncated bump
    prob_map[mask] += bump_strength * bump[mask]

    return prob_map

def detect_victim(uav_pos, victim_pos, detection_radius=50.0, sigma_prob=15.0, false_negative_rate=0.1):
    """
    Probabilistic victim detection.
    
    - detection_radius: max sensor range
    - sigma_prob: smooth decay of detection probability with distance
    - false_negative_rate: base chance to miss a victim even in range
    """
    d = np.linalg.norm(uav_pos - victim_pos)
    if d > detection_radius:
        return False  # out of range

    # Probability decays with distance (Gaussian)
    prob_detect = np.exp(-d**2 / (2 * sigma_prob**2))

    # Apply base false negative chance
    prob_detect *= (1 - false_negative_rate)

    # Random check
    return np.random.rand() < prob_detect

def detection_prob_map(uav_pos, victims, xv, yv, detection_radius=50.0, sigma_prob=15.0):
    """
    Generate a map showing detection probability of any victim from current UAV position.
    """
    prob_map = np.zeros_like(xv, dtype=float)
    for pos in victims:
        d2 = (xv - pos[0])**2 + (yv - pos[1])**2
        bump = np.exp(-d2 / (2 * sigma_prob**2))
        bump[d2 > detection_radius**2] = 0  # outside sensor range
        prob_map += bump
    # Normalize for visualization
    if prob_map.max() > 0:
        prob_map /= prob_map.max()
    return prob_map

def dynamic_repel_victim(prob_map, pos, xv, yv, radius=100.0, sigma=50.0,
                         repulsion=-0.5, decay_factor=0.99, t_last_seen=None, t_now=None):
    """
    Add repulsion around a detected victim that decays over distance and optionally time.
    - radius: max distance to consider
    - sigma: Gaussian smoothness
    - repulsion: negative bump strength
    - decay_factor: multiplies repulsion each timestep if t_last_seen is given
    - t_last_seen / t_now: for time decay
    """
    bump = np.exp(-((xv - pos[0])**2 + (yv - pos[1])**2) / (2*sigma**2))
    mask = bump > np.exp(-radius**2 / (2*sigma**2))
    
    rep_value = repulsion * bump[mask]
    
    # Apply time decay if applicable
    if t_last_seen is not None and t_now is not None:
        dt = t_now - t_last_seen
        rep_value *= decay_factor**dt
    
    prob_map[mask] += rep_value
    return prob_map


# --- Visualization setup ---
plt.ion()
fig = plt.figure(figsize=(14,10))
ax_map = fig.add_subplot(3,2,1)
ax_v = fig.add_subplot(3,2,2)
ax_s = fig.add_subplot(3,2,3)
ax_heading = fig.add_subplot(3,2,4)
ax_mode = fig.add_subplot(3,2,5)

# Probability map
im_prob = ax_map.imshow(prob_map_fine, origin='lower',
                        extent=(min_x, max_x, min_y, max_y),
                        cmap='viridis', alpha=1.0)

# Visited-cell heatmap overlay
visited_heatmap = np.zeros_like(prob_map_fine)
im_visited = ax_map.imshow(visited_heatmap, origin='lower',
                           extent=(min_x, max_x, min_y, max_y),
                           cmap='Reds', alpha=0.5)

# Polygon boundary
ax_map.plot(*soft_poly.exterior.xy, 'k-', lw=1)

# UAV path & dot
smoothed_path = [uav_pos.copy()]
path_line, = ax_map.plot([], [], 'r-', lw=1)
uav_dot, = ax_map.plot(uav_pos[0], uav_pos[1], 'ro')

# Victim scatter
colors_v = ['red' for _ in range(len(victims))]
vict_scat = ax_map.scatter(victims[:,0], victims[:,1], c=colors_v, s=30)

# Add colorbar for probability
cbar = plt.colorbar(im_prob, ax=ax_map, fraction=0.046, pad=0.04)
cbar.set_label("Probability")

# --- Velocity subplot ---
line_vx, = ax_v.plot([], [], label='vx')
line_vy, = ax_v.plot([], [], label='vy')
ax_v.set_title("vx / vy (m/s)")
ax_v.legend()

# --- Speed subplot ---
line_speed, = ax_s.plot([], [], label='speed')
ax_s.set_title("speed (m/s)")
ax_s.legend()

# --- Heading subplot ---
line_heading, = ax_heading.plot([], [], label="heading (deg)")
ax_heading.set_title("heading (deg)")
ax_heading.legend()

# --- Flight mode subplot ---
mode_line, = ax_mode.plot([], [], label="mode")
ax_mode.set_title("Flight mode")
ax_mode.set_yticks([0,1,2])
ax_mode.set_yticklabels(["quad","transition","fixedwing"])
ax_mode.set_xlabel("Step")
ax_mode.set_ylabel("Mode")
ax_mode.legend()

plt.tight_layout()
plt.pause(0.1)

# ---------------------------
# --- Simulation loop ---

visited_map = np.zeros_like(prob_map_fine)  # trail repulsion map
visited_decay = 0.995
trail_strength = -0.3

# Initialization
current_speed = 0.0
# ---------------------------
# Start of simulation loop
# ---------------------------

found_victims = []  # initialize outside loop
turn_cells = 40
slowdown_factor = 0.6

STEPS = 650
dt = 0.5

uniform_thresh = 1e-4        # threshold to detect uniform field
max_turn_deg = 5.0
max_turn_rad = np.deg2rad(max_turn_deg)
patch_radius_reset = 2       # neighborhood around detected victim to reset
boundary_threshold = 1.0     # meters to consider "touching boundary"
time_log = []

# Initialize UAV state
# current_speed = target_speed * 0.1
current_heading = 0.0
boundary_escape_steps = 0
last_heading_vec = np.array([1.0, 0.0])
smoothed_path = [uav_pos.copy()]
vx_log, vy_log, speed_log, heading_log, mode_log = [], [], [], [], []

# Initialize heatmap
visited_heatmap = np.zeros_like(prob_map_fine)

# ---------------------------
# Initialize boundary escape
# ---------------------------
mask = np.zeros_like(prob_map_fine, dtype=bool)
for ii in range(prob_map_fine.shape[0]):
    for jj in range(prob_map_fine.shape[1]):
        x, y = xv[ii, jj], yv[ii, jj]
        mask[ii, jj] = soft_poly.contains(Point(x, y))

# ---------------------------
# Initialization outside loop
# ---------------------------
mode = "quad"
prev_mode = mode
last_heading_vec = np.array([math.cos(current_heading), math.sin(current_heading)])
boundary_escape_steps = 0
BOUNDARY_ESCAPE_DURATION = 10
visited_heatmap = np.zeros_like(prob_map_fine)
sigma_vis = 20.0
look_ahead_cells = 5       # look further for gradient
sense_radius = 100.0
sigma_sense = 100.0
gain_sense = 4.0
uniform_thresh = 1e-4
mode_log = []

# Energy accounting
energy_used = 0.0
P_hover = 920.0    # W
P_fw_cruise = 300.0 # W
P_trans = 1500.0   # W
quad_max_speed = 12.0

transition_count = 0
BASE_TRANSITION_PENALTY = 2000.0   # J

plot_interval = 1   # update plots every 10 steps

vx_log, vy_log, speed_log, heading_log, mode_log = [], [], [], [], []
detected_victims = [False] * len(victims)
found_victims = []
victim_last_seen = [0.0] * len(victims)


# Use:
dt = 0.5  # timestep in seconds
T_total = 300.0
STEPS = np.arange(0, T_total + dt, dt)  # now iterable


# ---------------------------
# Simulation loop
# # ---------------------------

# --- Main simulation loop ---

for t_idx, t in enumerate(STEPS):
    
    # --- 1) Exploration bonus ---
    prob_map_fine += 1e-3

    # --- 2) Repel detected victims ---
    for k, pos in enumerate(victims):
        if detected_victims[k]:
            prob_map_fine = dynamic_repel_victim(prob_map_fine, pos, xv, yv,
                                                 radius=120.0, sigma=40.0,
                                                 repulsion=-1.0,
                                                 t_last_seen=t, t_now=t)
    
    # --- 3) Trail repulsion ---
    ix, iy = int((uav_pos[0]-min_x)/grid_res_fine), int((uav_pos[1]-min_y)/grid_res_fine)
    visited_map[iy, ix] = 1.0
    visited_map *= visited_decay
    prob_map_fine += trail_strength * visited_map

    # --- 4) Determine desired vector ---
    if boundary_escape_steps > 0:
        desired_vec = np.array([math.cos(escape_heading), math.sin(escape_heading)])
        boundary_escape_steps -= 1
        mode = "quad"
        max_mode_speed = quad_max_speed
    else:
        num_detected = sum(detected_victims)
        look_ahead = look_ahead_cells
        if num_detected >= 3:
            look_ahead = min(look_ahead_cells + num_detected, 12)

        grad = get_ipp_gradient(uav_pos, prob_map_fine, look_ahead=look_ahead)
        if np.linalg.norm(grad) < uniform_thresh:
            desired_vec = last_heading_vec
            if mode != "fixedwing":
                grad *= 0.5
        else:
            desired_vec = grad / (np.linalg.norm(grad)+1e-9)
            last_heading_vec = desired_vec

        avoid_vec, speed_factor = boundary_avoidance(uav_pos, soft_poly,
                                                     turn_cells=turn_cells,
                                                     grid_res=grid_res_fine,
                                                     slowdown_factor=slowdown_factor)
        desired_vec += avoid_vec
        desired_vec /= np.linalg.norm(desired_vec)+1e-9

    # --- 5) Boundary touch check ---
    point = Point(uav_pos[0], uav_pos[1])
    dist_to_edge = soft_poly.exterior.distance(point)
    if dist_to_edge <= 0.5 and boundary_escape_steps == 0:
        boundary_escape_steps = BOUNDARY_ESCAPE_DURATION
        nearest_point = np.array(soft_poly.exterior.interpolate(
                                soft_poly.exterior.project(point)).coords[0])
        escape_vec = uav_pos - nearest_point
        if np.linalg.norm(escape_vec) < 1e-3:
            escape_vec = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        escape_heading = math.atan2(escape_vec[1], escape_vec[0])
        current_heading = angle_wrap_rad(escape_heading)
        mode = "quad"
        current_speed = min(current_speed, quad_max_speed)
        continue

    # --- 6) Heading smoothing ---
    desired_heading = math.atan2(desired_vec[1], desired_vec[0])
    delta_heading = angle_wrap_rad(desired_heading - current_heading)
    delta_heading_clipped = np.clip(delta_heading, -np.deg2rad(5), np.deg2rad(5))
    current_heading = angle_wrap_rad(current_heading + delta_heading_clipped)

    # --- 7) Update flight mode ---
    turn_rate = abs(delta_heading_clipped)/dt
    if boundary_escape_steps > 0:
        mode = "quad"
        max_mode_speed = quad_max_speed
    elif turn_rate > 0.7*max_turn_rate:
        mode = "transition"
        max_mode_speed = target_speed
    else:
        mode = "fixedwing"
        max_mode_speed = target_speed

    if mode == "quad":
        current_speed = quad_max_speed * max(0.5, 1 - turn_rate/max_turn_rate)

    # --- 8) Smooth speed update ---
    current_speed += 0.5*(max_mode_speed - current_speed)
    current_speed = np.clip(current_speed, 0, max_mode_speed)

    # --- 9) Move UAV ---
    step_vec = current_speed * np.array([math.cos(current_heading),
                                        math.sin(current_heading)]) * dt
    uav_pos += step_vec
    uav_pos = clip_to_polygon(uav_pos, soft_poly)

    # --- 10) Smooth vx/vy logging ---
    vx = step_vec[0]/dt
    vy = step_vec[1]/dt
    vx_log.append(0.2*vx + 0.8*(vx_log[-1] if vx_log else vx))
    vy_log.append(0.2*vy + 0.8*(vy_log[-1] if vy_log else vy))

    # --- 11) Log position, speed, heading, mode ---
    smoothed_path.append(uav_pos.copy())
    speed_log.append(current_speed)
    heading_log.append(np.degrees(current_heading)%360)
    mode_log.append({"quad":0,"transition":1,"fixedwing":2}[mode])

    # --- 12) Detect victims ---
    for k, pos in enumerate(victims):
        if not detected_victims[k]:
            d = np.linalg.norm(uav_pos - pos)
            if d <= detection_radius:
                detected_victims[k] = True
                found_victims.append(pos)
                print(f"[t={t:.1f}s] Detected victim {k} at {pos}, distance {d:.1f} m")

    # --- 13) Energy accounting ---
    if mode == "quad":
        P = P_hover
    elif mode == "fixedwing":
        P = P_fw_cruise
    elif mode == "transition":
        # Penalize only if coming from fixedwing
        if prev_mode == "fixedwing":
            P = P_trans + 200  # extra energy penalty
        else:
            P = P_trans
    else:
        P = 0.0
    energy_used += P * dt
    prev_mode = mode

    # --- 14) Update plots every plot_interval ---
    if t_idx % plot_interval == 0:
        # Probability map and visited overlay
        im_prob.set_data(prob_map_fine)
        im_visited.set_data(visited_map)

        # UAV path
        path_line.set_data([p[0] for p in smoothed_path],
                        [p[1] for p in smoothed_path])
        uav_dot.set_data(uav_pos[0], uav_pos[1])

        # vx/vy
        line_vx.set_data(np.arange(len(vx_log))*dt, vx_log)
        line_vy.set_data(np.arange(len(vy_log))*dt, vy_log)
        ax_v.relim()
        ax_v.autoscale_view()

        # Speed
        line_speed.set_data(np.arange(len(speed_log))*dt, speed_log)
        ax_s.relim()
        ax_s.autoscale_view()

        # Heading
        line_heading.set_data(np.arange(len(heading_log))*dt, heading_log)
        ax_heading.relim()
        ax_heading.autoscale_view()

        # Flight mode
        mode_line.set_data(np.arange(len(mode_log)), mode_log)
        ax_mode.relim()
        ax_mode.autoscale_view()

        plt.pause(0.001)


    # --- 13) Stop if all victims found ---
    if all(detected_victims):
        print(f"All victims detected by step {STEPS[-1]}, sim time {t:.1f}s.")
        break


# ---------------------------
# --- Keep final plot open ---
plt.ioff()
plt.show()

# --- Mission summary ---
print("\n--- Mission summary ---")
if found_victims:
    print(f"Total victims detected: {len(found_victims)} / {len(victims)}")
    for i, v in enumerate(found_victims):
        print(f"  Victim {i}: at {v}")
else:
    print("No victims detected.")
print(f"Total energy used: {energy_used:.1f} J, Transitions: {transition_count}")

