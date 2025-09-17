import numpy as np
from time import sleep
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import random
from matplotlib import colors
import scipy.ndimage as nd
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import time
from shapely.geometry import Polygon, Point
import numpy as np
import math
import os
import sys
import xml.etree.ElementTree as ET
import pymap3d as pm

# class UAVPlanner:
#     def __init__(self, world_size=(100, 100), sensor_radius=5):
#         self.world_size = world_size
#         self.grid_map = np.zeros(world_size)  # belief map
#         self.sensor_radius = sensor_radius
#         self.visited = np.zeros(world_size)   # track explored cells

#     def simulate_measurement(self, pos):
#         """Simulate sensor measurement at pos."""
#         new_map = self.grid_map.copy()
#         x, y = int(pos[0]), int(pos[1])
#         x_min = max(x - self.sensor_radius, 0)
#         x_max = min(x + self.sensor_radius + 1, self.world_size[0])
#         y_min = max(y - self.sensor_radius, 0)
#         y_max = min(y + self.sensor_radius + 1, self.world_size[1])
#         # Increase probability for unseen cells
#         delta = 0.2 * (1 - self.grid_map[x_min:x_max, y_min:y_max])
#         new_map[x_min:x_max, y_min:y_max] += delta
#         new_map = np.clip(new_map, 0, 1.0)
#         return new_map

#     def information_gain(self, old_map, new_map):
#         """Compute gain only for previously unseen/low-probability cells."""
#         gain = np.sum(new_map - old_map)
#         return gain

#     def plan_next_horizon(self, uav_pos, candidate_points):
#         best_objective = -np.inf
#         best_point = None

#         for candidate in candidate_points:
#             candidate = np.array(candidate)
#             simulated_map = self.simulate_measurement(candidate)
#             gain = self.information_gain(self.grid_map, simulated_map)
#             distance = np.linalg.norm(candidate - uav_pos)
#             distance = max(distance, 1e-3)  # avoid div by zero
#             objective = gain / distance

#             # Random tie-breaker
#             if np.isclose(objective, best_objective):
#                 if np.random.rand() > 0.5:
#                     best_point = candidate
#                     best_objective = objective
#             elif objective > best_objective:
#                 best_objective = objective
#                 best_point = candidate

#         return best_point, best_objective

#     def update_map(self, pos):
#         self.grid_map = self.simulate_measurement(pos)
#         x, y = int(pos[0]), int(pos[1])
#         self.visited[x, y] = 1

# # === From previous GridMap class (simplified) ===

# class GridMap:
#     def __init__(self, width, height, res_x, res_y, pos_x=0, pos_y=0, frame_id="map"):
#         self.frame_id = frame_id
#         self.resolution = np.array([res_x, res_y])
#         self.length = np.array([int(round(height / res_x)), int(round(width / res_y))])
#         self.position = np.array([pos_x, pos_y])

#         # Belief map and covariance
#         self.data = np.ones((self.length[0], self.length[1])) * 0.5
#         self.covariance = np.eye(self.length[0] * self.length[1])

#     def fill_unknown(self):
#         self.data.fill(0.5)

#     def compute_covariance_trace(self):
#         return np.trace(self.covariance)

#     def construct_measurement_model(self, indices):
#         H = csr_matrix((np.ones(len(indices)), (range(len(indices)), indices)),
#                        shape=(len(indices), self.data.size))
#         return H

#     def KF_update(self, z, var, H):
#         # Flatten data into vector
#         x = self.data.flatten()

#         # Innovation
#         v = z - H.dot(x)

#         # Kalman gain using Cholesky
#         PHt = self.covariance @ H.T
#         S = H @ PHt + np.eye(H.shape[0]) * var
#         c, lower = cho_factor(S, lower=True)
#         W = cho_solve((c, lower), PHt.T).T

#         # Update state and covariance
#         x_new = x + W @ v
#         self.covariance -= PHt @ W.T

#         self.data = x_new.reshape(self.data.shape)

#     def predict_update(self, indices, var):
#         H = self.construct_measurement_model(indices)
#         S = H @ self.covariance @ H.T + np.eye(H.shape[0]) * var
#         c, lower = cho_factor(S, lower=True)
#         W = cho_solve((c, lower), (self.covariance @ H.T).T).T
#         self.covariance -= W @ W.T

# # Fake ground-truth world (binary: 0 free, 1 victim/target)

# # --- LatticePlanner from before ---

# # --- LatticePlanner from before with logging ---
# class LatticePlanner:
#     def __init__(self, grid_map, uav_altitude=20, lattice_resolution=10, planning_horizon=1):
#         self.grid_map = grid_map.copy()
#         self.uav_altitude = uav_altitude
#         self.lattice_resolution = lattice_resolution
#         self.planning_horizon = planning_horizon
#         self.grid_height, self.grid_width = grid_map.shape
#         xs = np.arange(0, self.grid_width, lattice_resolution)
#         ys = np.arange(0, self.grid_height, lattice_resolution)
#         self.lattice_points = [(x, y) for x in xs for y in ys]

#     def simulate_measurement(self, pos, sensor_range=5):
#         x, y = int(pos[0]), int(pos[1])
#         new_map = self.grid_map.copy()
#         x_min = max(x - sensor_range, 0)
#         x_max = min(x + sensor_range, self.grid_width - 1)
#         y_min = max(y - sensor_range, 0)
#         y_max = min(y + sensor_range, self.grid_height - 1)
#         new_map[y_min:y_max + 1, x_min:x_max + 1] += 0.05
#         return np.clip(new_map, 0, 1.0)

#     def information_gain(self, old_map, new_map):
#         return np.sum(new_map) - np.sum(old_map)

#     def plan_next_horizon(self, current_pos):
#         control_poses = [(*current_pos, self.uav_altitude)]
#         simulated_map = self.grid_map.copy()
#         prev_pos = current_pos
#         for _ in range(self.planning_horizon):
#             best_objective = -np.inf
#             best_point = None
#             for candidate in self.lattice_points:
#                 candidate_pos = candidate
#                 new_map = self.simulate_measurement(candidate_pos)
#                 gain = self.information_gain(simulated_map, new_map)
#                 cost = np.linalg.norm(np.array(candidate_pos) - np.array(prev_pos)) + 1e-6
#                 objective = gain / cost
#                 if objective > best_objective:
#                     best_objective = objective
#                     best_point = candidate_pos
#                     best_map = new_map
#             if best_point is None or best_objective <= 0:
#                 print("No informative move found. Stopping planning.")
#                 break
#             control_poses.append((*best_point, self.uav_altitude))
#             simulated_map = best_map
#             prev_pos = best_point
#             # --- Logging ---
#             print(f"Current UAV pos: {control_poses[-2][:2]} -> Next waypoint: {best_point}, Objective: {best_objective:.4f}")
#         return control_poses, simulated_map
    

USE_PPRZ = False  # set True if you want to send to Paparazzi (not configured here)
RANDOM_SEED = 111
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Map / victims


# Build a toy soft polygon for demonstration (replace with your softgeo_xy / soft_poly)
# For production use, replace corners below with softgeo_xy constructed from your flight-plan.
# softgeo_xy = np.array([
#     [0, 0], [200, 0], [220, 80], [180, 180], [60, 170], [0, 100]
# ])



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

# UAV initial state
uav_pos = np.array([50.0, 50.0], dtype=float)
dt = 0.5
target_speed = 19.0
min_speed = 6.0
alpha_speed = 0.2
max_turn_rate_deg = 10.0
max_turn_rate = np.deg2rad(max_turn_rate_deg) * dt
max_speed_change = 1.0 * dt
detection_radius = 20.0
look_ahead_cells = 1

current_speed = target_speed * 0.5
current_heading = 0.0

smoothed_path = [uav_pos.copy()]
vx_log, vy_log, speed_log, heading_log = [], [], [], []

# Base probability
base_prob = 0.1
prob_map_fine = np.ones_like(xv) * base_prob

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

def get_velocity_gradient(uav_pos, prob_map, look_ahead=1):
    i, j = world_to_cell(uav_pos[0], uav_pos[1])
    i_min, i_max = max(0, i-look_ahead), min(prob_map.shape[0], i+look_ahead+1)
    j_min, j_max = max(0, j-look_ahead), min(prob_map.shape[1], j+look_ahead+1)
    patch = prob_map[i_min:i_max, j_min:j_max]
    if patch.size == 0:
        return np.array([0.0, 0.0])
    js = np.arange(j_min, j_max)
    is_ = np.arange(i_min, i_max)
    jj, ii = np.meshgrid(js, is_)
    cx = (jj - j) * grid_res_fine
    cy = (ii - i) * grid_res_fine
    w = patch
    vx = np.sum(w * cx)
    vy = np.sum(w * cy)
    if abs(vx) < 1e-8 and abs(vy) < 1e-8:
        return np.array([random.uniform(-1,1), random.uniform(-1,1)])
    return np.array([vx, vy])

def sensor_update_continuous(prob_map, uav_pos, victims, xv, yv, influence_radius=50, decay=0.95):
    """
    Probabilistic sensor update that increases probability locally around UAV
    to simulate exploitation of nearby high-probability regions.
    """
    # Decay existing probabilities slightly (to avoid runaway sum)
    prob_map *= decay
    
    # Compute influence patch
    dx = xv - uav_pos[0]
    dy = yv - uav_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    mask = dist <= influence_radius

    # Increase probability in local patch (exploitation)
    prob_map[mask] += np.exp(-(dist[mask]/influence_radius)**2)
    
    # Re-normalize
    prob_map /= prob_map.sum()
    return prob_map


def clip_to_polygon(uav_pos, polygon):
    if not polygon.contains(Point(uav_pos[0], uav_pos[1])):
        # Move back to nearest point inside polygon
        nearest = np.array(polygon.exterior.interpolate(polygon.exterior.project(Point(uav_pos[0], uav_pos[1]))).coords[0])
        return nearest
    return uav_pos

# --- Boundary avoidance parameters ---
turn_cells = 7  # number of cells from edge where we start turning
slowdown_factor = 0.5  # slow down to 50% of current speed at boundary

# --- Updated boundary avoidance function ---
def boundary_avoidance(uav_pos, polygon, prob_map, turn_cells, grid_res, slowdown_factor=0.5):
    """
    Returns (avoidance_vector, speed_factor) to keep UAV inside polygon.
    UAV starts turning and slowing down when within 'turn_cells' of boundary.
    """
    point = Point(uav_pos[0], uav_pos[1])
    dist_to_edge = polygon.exterior.distance(point)
    turn_distance = turn_cells * grid_res

    if dist_to_edge < turn_distance:
        nearest = np.array(polygon.exterior.interpolate(polygon.exterior.project(point)).coords[0])
        away_vec = uav_pos - nearest
        if np.linalg.norm(away_vec) > 1e-6:
            away_vec = away_vec / np.linalg.norm(away_vec)
            strength = 1 - dist_to_edge / turn_distance
            speed_scale = max(slowdown_factor, dist_to_edge / turn_distance)
            return away_vec * strength, speed_scale
    return np.array([0.0, 0.0]), 1.0


# ---------------------------
# Plot setup
# ---------------------------
plt.ion()
fig = plt.figure(figsize=(14,8))
ax_map = fig.add_subplot(2,2,1)
ax_v = fig.add_subplot(2,2,2)
ax_s = fig.add_subplot(2,2,3)
ax_heading = fig.add_subplot(2,2,4)

im = ax_map.imshow(prob_map_fine.T, origin='lower', extent=(min_x,max_x,min_y,max_y), cmap='viridis')
ax_map.plot(*soft_poly.exterior.xy,'k-',lw=1)
vict_scat = ax_map.scatter(victims[:,0], victims[:,1], c='r', s=30)
path_line, = ax_map.plot([],[], 'r-', lw=1)
uav_dot, = ax_map.plot(uav_pos[0], uav_pos[1],'ro')

line_vx, = ax_v.plot([],[], label='vx')
line_vy, = ax_v.plot([],[], label='vy')
ax_v.set_title("vx / vy (m/s)")
ax_v.legend()
line_speed, = ax_s.plot([],[], label='speed')
ax_s.set_title("speed (m/s)")
ax_s.legend()
line_heading, = ax_heading.plot([],[], label='heading (deg)')
ax_heading.set_title("heading (deg)")
ax_heading.legend()
plt.tight_layout()
plt.pause(0.1)

# ---------------------------
# Main simulation loop
# ---------------------------

# --- Main simulation loop ---
# --- Simulation loop ---
STEPS = 600
alpha_accel = 0.05          # smooth speed ramp-up factor
visited_decay_factor = 0.95 # decay visited cells to prevent circling
turn_cells = 5
slowdown_factor = 0.5
time_log = []

for step in range(STEPS):
    t = step * dt
    time_log.append(t)

    # --- 1) Update probability map dynamically near UAV ---
    prob_map_fine = sensor_update_continuous(prob_map_fine, uav_pos, victims, xv, yv)

    # --- Decay visited cell to reduce circling ---
    i, j = world_to_cell(uav_pos[0], uav_pos[1])
    prob_map_fine[i,j] *= visited_decay_factor
    prob_map_fine /= prob_map_fine.sum()  # normalize

    # --- 2) Compute velocity gradient ---
    grad = get_velocity_gradient(uav_pos, prob_map_fine, look_ahead=look_ahead_cells)
    desired_vec = np.array(grad)
    if np.linalg.norm(desired_vec) < 1e-6:
        desired_vec = np.array([random.uniform(-1,1), random.uniform(-1,1)])

    # --- 2b) Boundary avoidance ---
    avoid_vec, speed_factor = boundary_avoidance(uav_pos, soft_poly, prob_map_fine,
                                                 turn_cells=turn_cells,
                                                 grid_res=grid_res_fine,
                                                 slowdown_factor=slowdown_factor)
    desired_vec += avoid_vec
    if np.linalg.norm(desired_vec) < 1e-6:
        desired_vec = np.array([random.uniform(-1,1), random.uniform(-1,1)])

    # --- Desired heading & speed ---
    desired_heading = math.atan2(desired_vec[1], desired_vec[0])
    desired_speed = np.clip(np.linalg.norm(desired_vec) * 0.2, min_speed, target_speed)

    # --- 3) Smooth heading ---
    delta_ang = angle_wrap_rad(desired_heading - current_heading)
    delta_ang_clipped = np.clip(delta_ang, -max_turn_rate, max_turn_rate)
    current_heading = angle_wrap_rad(current_heading + delta_ang_clipped)

    # --- 4) Smooth speed with gradual acceleration & boundary slowdown ---
    current_speed = current_speed + alpha_accel * (desired_speed - current_speed)
    current_speed *= speed_factor  # reduce only near boundaries

    # --- 5) Move UAV ---
    step_vec = current_speed * np.array([math.cos(current_heading), math.sin(current_heading)]) * dt
    uav_pos += step_vec
    uav_pos = clip_to_polygon(uav_pos, soft_poly)

    vx = step_vec[0] / dt
    vy = step_vec[1] / dt

    smoothed_path.append(uav_pos.copy())
    vx_log.append(vx)
    vy_log.append(vy)
    speed_log.append(current_speed)
    heading_log.append(np.degrees(current_heading) % 360)

    # --- 6) Detect victims ---
    for k, pos in enumerate(victims):
        if not detected_victims[k]:
            d = np.linalg.norm(uav_pos - pos)
            if d <= detection_radius:
                detected_victims[k] = True
                print(f"[t={t:.1f}s] Detected victim {k} at {pos}, distance {d:.1f} m.")

    # --- 7) Visualization update ---
    im.set_data(prob_map_fine.T)
    colors_v = ['gray' if detected_victims[k] else 'red' for k in range(len(victims))]
    vict_scat.remove()
    vict_scat = ax_map.scatter(victims[:,0], victims[:,1], c=colors_v, s=30)

    path_arr = np.array(smoothed_path)
    path_line.set_data(path_arr[:,0], path_arr[:,1])
    uav_dot.set_data(uav_pos[0], uav_pos[1])

    xs = np.arange(len(vx_log))
    line_vx.set_data(xs, vx_log)
    line_vy.set_data(xs, vy_log)
    ax_v.relim(); ax_v.autoscale_view()

    line_speed.set_data(xs, speed_log)
    ax_s.relim(); ax_s.autoscale_view()

    line_heading.set_data(xs, heading_log)
    ax_heading.relim(); ax_heading.autoscale_view()

    plt.pause(0.001)

    # --- 8) Stop if all victims detected ---
    if detected_victims.all():
        print(f"All victims detected by step {step}, sim time {t:.1f}s.")
        break

plt.ioff()
plt.show()


# ====== Parameters ======

# # ====== Parameters ======
# WORLD_SIZE = (100, 100)
# STEPS = 200
# SENSOR_RADIUS = 10
# MAX_SPEED = 1.0
# MIN_SPEED = 0.1
# MAX_SPEED_CHANGE = 0.2
# MAX_HEADING_CHANGE = 15.0  # degrees per step
# CANDIDATES = np.array([[1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]])

# # ====== Ground truth ======
# GROUND_TRUTH = np.zeros(WORLD_SIZE)
# GROUND_TRUTH[30:35,60:65] = 1.0
# GROUND_TRUTH[70:72,20:25] = 1.0
# GROUND_TRUTH = gaussian_filter(GROUND_TRUTH, sigma=15)

# # ====== Belief map ======
# belief_map = np.zeros(WORLD_SIZE)

# # ====== UAV start ======
# uav_pos = np.array([50.0, 50.0])
# current_speed = 0.0
# current_heading = 0.0

# # ====== Update function ======
# def sensor_update(pos):
#     x, y = int(pos[0]), int(pos[1])
#     x_min, x_max = max(0,x-SENSOR_RADIUS), min(WORLD_SIZE[0], x+SENSOR_RADIUS+1)
#     y_min, y_max = max(0,y-SENSOR_RADIUS), min(WORLD_SIZE[1], y+SENSOR_RADIUS+1)
#     belief_map[x_min:x_max, y_min:y_max] += GROUND_TRUTH[x_min:x_max, y_min:y_max]
#     np.clip(belief_map,0,1.0,out=belief_map)

# def compute_gain(candidate_pos):
#     x, y = int(candidate_pos[0]), int(candidate_pos[1])
#     x_min, x_max = max(0,x-SENSOR_RADIUS), min(WORLD_SIZE[0], x+SENSOR_RADIUS+1)
#     y_min, y_max = max(0,y-SENSOR_RADIUS), min(WORLD_SIZE[1], y+SENSOR_RADIUS+1)
#     gain = np.sum(GROUND_TRUTH[x_min:x_max, y_min:y_max]*(1-belief_map[x_min:x_max, y_min:y_max]))
#     return gain

# # ====== Storage ======
# path = [uav_pos.copy()]
# vx_list, vy_list, speed_list, heading_list = [], [], [], []

# # ====== Simulation loop ======
# # ====== Simulation loop ======
# plt.ion()
# fig, axes = plt.subplots(2,1,figsize=(8,10))
# im_ax = axes[0]
# line_ax = axes[1]

# im = im_ax.imshow(belief_map.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
# uav_dot, = im_ax.plot(uav_pos[0], uav_pos[1], 'ro')
# trace, = im_ax.plot([], [], 'r-', lw=1)

# for step in range(STEPS):
#     # Candidates
#     candidates = uav_pos + CANDIDATES
#     candidates = [c for c in candidates if 0<=c[0]<WORLD_SIZE[0] and 0<=c[1]<WORLD_SIZE[1]]
    
#     # Compute gain
#     gains = [compute_gain(c) for c in candidates]
#     if len(gains)==0 or max(gains)<=0:
#         raw_step = np.random.randn(2)
#     else:
#         raw_step = candidates[np.argmax(gains)] - uav_pos

#     # Desired heading & speed
#     desired_heading = np.degrees(np.arctan2(raw_step[1], raw_step[0]))
#     desired_speed = np.linalg.norm(raw_step)
#     desired_speed = np.clip(desired_speed, MIN_SPEED, MAX_SPEED)

#     # Clip heading change
#     delta_heading = (desired_heading - current_heading + 180) % 360 - 180
#     delta_heading = np.clip(delta_heading, -MAX_HEADING_CHANGE, MAX_HEADING_CHANGE)
#     current_heading = (current_heading + delta_heading) % 360

#     # Clip speed change
#     delta_speed = desired_speed - current_speed
#     delta_speed = np.clip(delta_speed, -MAX_SPEED_CHANGE, MAX_SPEED_CHANGE)
#     current_speed += delta_speed

#     # Convert back to velocity vector
#     step_vec = current_speed * np.array([np.cos(np.radians(current_heading)),
#                                         np.sin(np.radians(current_heading))])

#     # Move UAV
#     uav_pos += step_vec
#     sensor_update(uav_pos)
#     path.append(uav_pos.copy())

#     # Logging: print vx, vy, speed, heading
#     vx, vy = step_vec
#     vx_list.append(vx)
#     vy_list.append(vy)
#     speed_list.append(current_speed)
#     heading_list.append(current_heading)
#     print(f"Step {step}: UAV ({uav_pos[0]:.1f},{uav_pos[1]:.1f}) | vx {vx:.2f} | vy {vy:.2f} | Heading {current_heading:.1f}° | Speed {current_speed:.2f}")

#     # Update plot
#     im.set_data(belief_map.T)
#     uav_dot.set_data(uav_pos[0], uav_pos[1])
#     trace.set_data(np.array(path)[:,0], np.array(path)[:,1])
#     plt.pause(0.01)

# # Plot vx, vy, speed, heading
# line_ax.plot(vx_list, label='vx')
# line_ax.plot(vy_list, label='vy')
# line_ax.plot(speed_list, label='speed')
# line_ax.plot(heading_list, label='heading')
# line_ax.legend()
# plt.ioff()
# plt.show()

########################################################################
########################################################################

# WORLD_SIZE = (100, 100)
# SENSOR_RADIUS = 10  # footprint in cells
# CANDIDATES = np.array([
#     [1, 0], [-1, 0], [0, 1], [0, -1],
#     [1, 1], [-1, -1], [1, -1], [-1, 1]
# ])
# STEPS = 300

# # ====== Ground truth ======
# GROUND_TRUTH = np.zeros(WORLD_SIZE)
# GROUND_TRUTH[30:35, 60:65] = 1.0
# GROUND_TRUTH[70:72, 20:25] = 1.0

# # Spread the targets to create a gradient (Gaussian blur)
# GROUND_TRUTH = nd.gaussian_filter(GROUND_TRUTH, sigma=10)

# # ====== Belief map ======
# belief_map = np.zeros(WORLD_SIZE)

# # ====== UAV start ======
# uav_pos = np.array([10, 10], dtype=float)

# # ====== Functions ======
# def sensor_update(pos):
#     """Update belief map around UAV position using sensor model."""
#     x, y = int(pos[0]), int(pos[1])
#     x_min, x_max = max(0, x-SENSOR_RADIUS), min(WORLD_SIZE[0], x+SENSOR_RADIUS+1)
#     y_min, y_max = max(0, y-SENSOR_RADIUS), min(WORLD_SIZE[1], y+SENSOR_RADIUS+1)
#     # Add measured probability from ground truth
#     belief_map[x_min:x_max, y_min:y_max] += GROUND_TRUTH[x_min:x_max, y_min:y_max]
#     # Clip belief map to 1.0
#     np.clip(belief_map, 0, 1.0, out=belief_map)

# def compute_gain(candidate_pos, current_pos):
#     """Compute expected information gain per distance."""
#     x, y = int(candidate_pos[0]), int(candidate_pos[1])
#     x_min, x_max = max(0, x-SENSOR_RADIUS), min(WORLD_SIZE[0], x+SENSOR_RADIUS+1)
#     y_min, y_max = max(0, y-SENSOR_RADIUS), min(WORLD_SIZE[1], y+SENSOR_RADIUS+1)
    
#     # Gain is higher for high-probability, unexplored cells
#     gain = np.sum(GROUND_TRUTH[x_min:x_max, y_min:y_max] * (1 - belief_map[x_min:x_max, y_min:y_max]))
    
#     dist = np.linalg.norm(candidate_pos - current_pos) + 1e-6  # avoid div by zero
#     return gain / dist

# # ====== Simulation loop ======
# import matplotlib.pyplot as plt

# plt.ion()
# fig, ax = plt.subplots()
# im = ax.imshow(belief_map.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
# uav_dot, = ax.plot(uav_pos[0], uav_pos[1], 'ro')

# for step in range(STEPS):
#     # Generate candidate waypoints
#     candidates = uav_pos + CANDIDATES
#     candidates = [c for c in candidates if 0 <= c[0] < WORLD_SIZE[0] and 0 <= c[1] < WORLD_SIZE[1]]
    
#     # Compute gains
#     gains = [compute_gain(c, uav_pos) for c in candidates]
    
#     if len(gains) == 0 or max(gains) <= 0:
#         next_wp = uav_pos
#     else:
#         next_wp = candidates[np.argmax(gains)]
    
#     # Move UAV
#     uav_pos = next_wp
#     sensor_update(uav_pos)
    
#     # Print info
#     print(f"Step {step}: UAV ({uav_pos[0]:.1f},{uav_pos[1]:.1f}) -> Next ({next_wp[0]:.1f},{next_wp[1]:.1f}), Max Gain {max(gains):.4f}")
    
#     # Update plot
#     im.set_data(belief_map.T)
#     uav_dot.set_data(uav_pos[0], uav_pos[1])
#     plt.pause(0.1)

# plt.ioff()
# plt.show()