import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
import pymap3d as pm
import time
import threading
from math import atan2, hypot, degrees

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

# --- Parse waypoints from XML ---
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
margin = 50
min_x, max_x = np.min(all_x)-margin, np.max(all_x)+margin
min_y, max_y = np.min(all_y)-margin, np.max(all_y)+margin

grid_res = 20   # coarse grid cells
xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res),
                     np.arange(min_y, max_y, grid_res))

# start with uniform probability for each cell
prob_map = np.ones_like(xv, dtype=float) * 0.1  

# add victim hotspots
for pos in victims:
    dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
    prob_map += np.exp(-(dist/50)**2)   # wider spread for coarse grid

# normalize to proper probability distribution
prob_map /= np.sum(prob_map)

# --- Build static map ---
plt.ion()
fig, ax = plt.subplots(figsize=(16,12))
ax.set_title("UAV Mission Area (ENU)")
ax.set_aspect('equal', adjustable='datalim')

ax.pcolormesh(xv, yv, prob_map, cmap='coolwarm', alpha=0.6, shading='auto')

for name, (x, y, _) in waypoints.items():
    ax.scatter(x, y, c='blue', marker='o')
    ax.text(x+5, y+5, name, fontsize=8, color='white')

ax.scatter(victims[:,0], victims[:,1], c='red', marker='x', s=80, label='Victims')

ax.plot(np.append(ehvb_xy[:,0], ehvb_xy[0,0]),
        np.append(ehvb_xy[:,1], ehvb_xy[0,1]),
        'orange', linewidth=2, label='EHVB / Flyzone')

ax.plot(np.append(softgeo_xy[:,0], softgeo_xy[0,0]),
        np.append(softgeo_xy[:,1], softgeo_xy[0,1]),
        'purple', linewidth=2, label='SoftGeofence')

# UAV dot (dynamic)
uav_dot, = ax.plot([], [], 'ro', markersize=12, label='UAV')

ax.legend()
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)


lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0

# -----------------------------
# Shared UAV state
# -----------------------------
uav_state = {"x": None, "y": None, "z": None, "vx": None, "vy": None, "heading": None}
cmd_state = {"vx": 0.0, "vy": 0.0, "heading": 0.0}

# -----------------------------
# Parameters
# -----------------------------
ac_id = 37
update_dt = 0.1         # seconds per loop
velocity_max = 10.0     # max commanded speed m/s

# --- Callback for messages ---
def on_gps_int(ac_id, msg):
    if msg.name == "GPS_INT":
        lat = float(msg["lat"]) * 1e-7
        lon = float(msg["lon"]) * 1e-7
        alt = float(msg["alt"]) / 100.0  # cm → m
        x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        uav_state["x"], uav_state["y"], uav_state["z"] = x, y, z

    # Custom made message for waypoints:

    # elif msg.name == "WP_LIST_ENU":
    #     n = int(msg['nb_wp'])   # convert to int
    #     print(f"\n[WP_LIST_ENU] received {n} waypoints from AC {ac_id}")
    #     for i in range(n):
    #         east = int(msg['east'][i])
    #         north = int(msg['north'][i])
    #         up = int(msg['up'][i])
    #         print(f" WP {i}: east={east}, north={north}, up={up}")



# scales from messages.xml
POS_SCALE = 0.0039063    # int -> meters for ins_x/ins_y/ins_z
VEL_SCALE = 0.0000019    # int -> m/s for ins_xd/ins_yd/ins_zd

def on_ins(ac_id, msg):
    """
    Parse INS message and update uav_state in ENU (east,north,up) convention.
    Note: INS message fields appear to be in NED-like order: ins_x = north, ins_y = east.
    """
    if msg.name != "INS":
        return

    # Raw ints from message
    ins_x   = float(msg["ins_x"])   # likely north (m when scaled)
    ins_y   = float(msg["ins_y"])   # likely east
    ins_z   = float(msg["ins_z"])
    ins_xd  = float(msg["ins_xd"])  # velocity along ins_x (north) (scaled)
    ins_yd  = float(msg["ins_yd"])  # velocity along ins_y (east)
    ins_zd  = float(msg["ins_zd"])

    # Convert to meters / m/s using scales from messages.xml
    north_m = ins_x * POS_SCALE
    east_m  = ins_y * POS_SCALE
    up_m    = -ins_z * POS_SCALE     # INS z is usually down, so invert for 'up' if needed

    north_v = ins_xd * VEL_SCALE
    east_v  = ins_yd * VEL_SCALE
    up_v    = -ins_zd * VEL_SCALE    # same inversion assumption

    # Update uav_state in ENU convention (x=east, y=north, z=up)
    uav_state["x"] = east_m
    uav_state["y"] = north_m
    uav_state["z"] = up_m
    uav_state["vx"] = east_v
    uav_state["vy"] = north_v
    uav_state["vz"] = up_v

    # Compute ground heading (ground course) as radians: atan2(East_vel, North_vel)
    if abs(east_v) > 1e-6 or abs(north_v) > 1e-6:
        uav_state["heading"] = atan2(east_v, north_v)
    else:
        uav_state["heading"] = None

    # Debug prints to validate both axis interpretations
    heading_deg = degrees(uav_state["heading"]) if uav_state["heading"] is not None else float('nan')
    print(f"INS -> AC{ac_id}: east={east_m:.2f}, north={north_m:.2f}, "
          f"east_v={east_v:.2f}, north_v={north_v:.2f}, heading={heading_deg:.1f}°")

# -----------------------------
# Helper functions
# -----------------------------
def compute_prob_gradient(prob_map, xv, yv, uav_x, uav_y):
    ix = np.abs(xv[0,:] - uav_x).argmin()
    iy = np.abs(yv[:,0] - uav_y).argmin()
    gy, gx = np.gradient(prob_map)
    return gx[iy, ix], gy[iy, ix]

def gradient_to_course_speed(grad_x, grad_y, cruise_speed=3.0, max_speed=20.0, min_speed=0.5):
    norm = hypot(grad_x, grad_y)
    if norm < 1e-6:
        return 0.0, 0.0
    dx = grad_x / norm
    dy = grad_y / norm
    desired_speed = max(min_speed, min(cruise_speed, max_speed))
    course = atan2(dy, dx)   # ENU convention
    return course, desired_speed

# def send_guided_full_ned(ac_id, vx_enu, vy_enu, vz_enu=0.0, heading_rad=0.0):
#     """
#     Send velocity commands in ENU coordinates, converted to NED for Paparazzi.
#     """
#     msg = PprzMessage("datalink", "GUIDED_FULL_NED")
#     msg['ac_id'] = int(ac_id)

#     # Convert current UAV ENU position to NED
#     x_ned = uav_state.get('y', 0.0)       # ENU North -> NED X
#     y_ned = uav_state.get('x', 0.0)       # ENU East  -> NED Y
#     z_ned = -uav_state.get('z', 0.0)      # ENU Up   -> NED Z (down)

#     # Convert velocities
#     vx_ned = vy_enu   # ENU North -> NED X
#     vy_ned = vx_enu   # ENU East  -> NED Y
#     vz_ned = -vz_enu  # ENU Up    -> NED Z

#     msg['x'] = float(x_ned)
#     msg['y'] = float(y_ned)
#     msg['z'] = float(z_ned)
#     msg['vx'] = float(vx_ned)
#     msg['vy'] = float(vy_ned)
#     msg['vz'] = float(vz_ned)
#     msg['ax'] = 0.0
#     msg['ay'] = 0.0
#     msg['az'] = 0.0
#     msg['heading'] = float(heading_rad)

#     cmd_state.update({"vx": vx_enu, "vy": vy_enu, "heading": heading_rad})

#     interface.send(msg, ac_id=int(ac_id))
#     print(f"GUIDED_FULL_NED -> AC{ac_id}: vx={vx_enu:.2f}, vy={vy_enu:.2f}, heading={degrees(heading_rad):.1f}°")

def send_setpoint_guided_full_ned(ac_id, vx_enu, vy_enu, vz_enu=0.0):
    """
    Send velocity commands in ENU coordinates, converted to NED for Paparazzi.
    """
    msg = PprzMessage("datalink", "GUIDED_SETPOINT_NED")
    msg['ac_id'] = int(ac_id)
    msg['flags'] = np.packbits([0, 0, 0, 0, 0, 1, 0, 0], bitorder='little')[0].astype(np.uint8)

    # Convert velocities
    vx_ned = vy_enu   # ENU North -> NED X
    vy_ned = vx_enu   # ENU East  -> NED Y
    vz_ned = -vz_enu  # ENU Up    -> NED Z

    msg['x'] = float(vx_ned)
    msg['y'] = float(vy_ned)
    msg['z'] = float(vz_ned)
    msg['yaw'] = 0
  
    cmd_state.update({"vx": vx_enu, "vy": vy_enu})

    interface.send(msg, ac_id=int(ac_id))
    print(f"GUIDED_SETPOINT_NED -> AC{ac_id}: Sending vx_enu={vx_enu:.2f} m/s, vy_enu={vy_enu:.2f} m/s")


# -----------------------------
# Ivy interface
# -----------------------------
interface = IvyMessagesInterface("rotwingframe", ivy_bus="127.255.255.255:2010")
interface.subscribe(on_gps_int)
interface.subscribe(on_ins)
interface.start()


plt.ion()

while True:
    if uav_state["x"] is not None:
        # 1) Gradient-based control
        grad_x, grad_y = compute_prob_gradient(prob_map, xv, yv, uav_state["x"], uav_state["y"])
        desired_course, desired_speed = gradient_to_course_speed(grad_x, grad_y, max_speed=velocity_max)
        vx = desired_speed * np.cos(desired_course)
        vy = desired_speed * np.sin(desired_course)
        vx_enu = vx   # East velocity
        vy_enu = vy  # North velocity (negative = south)
        send_setpoint_guided_full_ned(ac_id, vx_enu, vy_enu)
        # send_guided_full_ned(ac_id, vx, vy, heading=desired_course)

        # # 2) Debug print
        # ins_heading_deg = degrees(uav_state["heading"]) if uav_state["heading"] is not None else float('nan')
        # cmd_heading_deg = degrees(cmd_state["heading"]) if cmd_state["heading"] is not None else float('nan')
        print(f"UAV pos: x={uav_state['x']:.2f}, y={uav_state['y']:.2f} | "
              f"Cmd vx={cmd_state['vx']:.2f}, vy={cmd_state['vy']:.2f} | "
              f"INS vx={uav_state['vx']:.2f}, vy={uav_state['vy']:.2f}")

        # 3) Plot UAV dot
        uav_dot.set_data(uav_state["x"], uav_state["y"])
        plt.pause(0.001)

    time.sleep(update_dt)


while True:
    if uav_state["x"] is not None:
        uav_dot.set_data(uav_state["x"], uav_state["y"])
        fig.canvas.draw()
        fig.canvas.flush_events()
    time.sleep(0.1)


# # -----------------------------
# # Parameters
# # -----------------------------
# ac_id = 37              # Aircraft ID
# update_dt = 0.1         # seconds per guidance update
# cruise_speed = 3.0      # m/s along gradient
# fixed_bearing_deg = 0.0 # degrees, 0 = north
# fixed_bearing_rad = np.radians(fixed_bearing_deg)

# # -----------------------------
# # Helper: send GUIDED_FULL_NED
# # -----------------------------
# def send_guided_full_ned(ac_id, grad_x, grad_y, uav_state, fixed_bearing_rad):
#     if uav_state['x'] is None or uav_state['y'] is None:
#         return

#     norm = np.hypot(grad_x, grad_y)
#     if norm < 1e-6:
#         vx, vy = 0.0, 0.0
#     else:
#         vx = (grad_x / norm) * cruise_speed  # East
#         vy = (grad_y / norm) * cruise_speed  # North

#     vz = 0.0  # hold altitude

#     msg = PprzMessage("datalink", "GUIDED_FULL_NED")
#     msg['ac_id'] = int(ac_id)
#     msg['x'] = float(uav_state['x'])
#     msg['y'] = float(uav_state['y'])
#     msg['z'] = float(uav_state.get('z', 0.0))
#     msg['vx'] = float(vx)
#     msg['vy'] = float(vy)
#     msg['vz'] = float(vz)
#     msg['ax'] = 0.0
#     msg['ay'] = 0.0
#     msg['az'] = 0.0
#     msg['heading'] = float(fixed_bearing_rad)

#     interface.send(msg, ac_id=int(ac_id))
#     print(f"GUIDED_FULL_NED -> AC{ac_id}: vx={vx:.2f}, vy={vy:.2f}, heading={np.degrees(fixed_bearing_rad):.1f}°")

# def compute_prob_gradient(prob_map, xv, yv, uav_x, uav_y):
#     """
#     Returns the local gradient vector at UAV position
#     """
#     # Find closest grid cell
#     ix = np.abs(xv[0,:] - uav_x).argmin()
#     iy = np.abs(yv[:,0] - uav_y).argmin()

#     # Compute gradient at that point
#     gy, gx = np.gradient(prob_map)
#     grad_x = gx[iy, ix]
#     grad_y = gy[iy, ix]
#     return grad_x, grad_y

# # -----------------------------
# # Main loop
# # -----------------------------
# while True:
#     if uav_state["x"] is not None:
#         # Update UAV dot on map
#         uav_dot.set_data(uav_state["x"], uav_state["y"])

#         # Compute probability gradient
#         grad_x, grad_y = compute_prob_gradient(prob_map, xv, yv, uav_state["x"], uav_state["y"])
#         print(f"UAV pos: x={uav_state['x']:.1f}, y={uav_state['y']:.1f}, grad_x={grad_x:.5f}, grad_y={grad_y:.5f}")

#         # Send velocity setpoint in NED with fixed ground heading
#         send_guided_full_ned(ac_id, grad_x, grad_y, uav_state, fixed_bearing_rad)

#         # Redraw map
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#     time.sleep(update_dt)

# def send_hybrid_guidance(ac_id, desired_course, desired_speed, uav_state):
#     """
#     Send velocity reference to a hybrid UAV using HYBRID_GUIDANCE.

#     Parameters:
#         ac_id          : Aircraft ID (int)
#         desired_course : desired ground course in radians (0 = north, clockwise)
#         desired_speed  : desired ground speed in m/s
#         uav_state      : dict with keys 'x', 'y' (ENU meters), optional 'z'
#     """
#     if uav_state['x'] is None or uav_state['y'] is None:
#         return

#     # Convert course + speed to ENU components
#     vx = desired_speed * np.sin(desired_course)  # East
#     vy = desired_speed * np.cos(desired_course)  # North

#     # Scaling for HYBRID_GUIDANCE (from messages.xml)
#     # Scaling for HYBRID_GUIDANCE (from messages.xml)
#     pos_scale   = 1 / 0.0039063     # m → int32 (pos_x, pos_y)
#     vel_scale   = 1 / 0.0000019     # m/s → int32 (speed_x, speed_y)
#     speed_scale = 1 / 0.0039063     # m/s → int32 (norm_ref_speed)


#     # Build message
#     msg = PprzMessage("telemetry", "HYBRID_GUIDANCE")

#     # Position: use current UAV location
#     msg['pos_x'] = int(uav_state['x'] * pos_scale)
#     msg['pos_y'] = int(uav_state['y'] * pos_scale)

#     # ✅ Desired velocity directly in ENU (used by GCS + guidance)
#     msg['speed_x'] = int(vx * vel_scale)
#     msg['speed_y'] = int(vy * vel_scale)

#     # Norm of reference speed
#     msg['norm_ref_speed'] = int(desired_speed * speed_scale)

#     # Other fields left at 0
#     msg['wind_x'] = 0
#     msg['wind_y'] = 0
#     msg['pos_err_x'] = 0
#     msg['pos_err_y'] = 0
#     msg['speed_sp_x'] = 0  # unused
#     msg['speed_sp_y'] = 0  # unused
#     msg['heading_diff'] = 0
#     msg['phi'] = 0
#     msg['theta'] = 0
#     msg['psi'] = 0

#     # Send via Ivy
#     interface.send(msg, ac_id=int(ac_id))

#     print(f"Sent HYBRID_GUIDANCE to AC {ac_id}: vx={vx:.2f} m/s, vy={vy:.2f} m/s, speed={desired_speed:.2f} m/s")

# # -----------------------------
# # Parameters
# # -----------------------------
# ac_id = 37              # Aircraft ID
# update_dt = 0.1         # seconds per guidance update
# velocity_max = 10.0     # m/s, maximum allowed UAV speed
# grad_scale = 500.0        # scaling factor for gradient → speed


# # -----------------------------
# # Helper: compute probability map gradient
# # -----------------------------
# def compute_prob_gradient(prob_map, xv, yv, uav_x, uav_y):
#     """
#     Returns the local gradient vector at UAV position
#     """
#     # Find closest grid cell
#     ix = np.abs(xv[0,:] - uav_x).argmin()
#     iy = np.abs(yv[:,0] - uav_y).argmin()

#     # Compute gradient at that point
#     gy, gx = np.gradient(prob_map)
#     grad_x = gx[iy, ix]
#     grad_y = gy[iy, ix]
#     return grad_x, grad_y

# # -----------------------------
# # Helper: convert gradient → course & speed
# # -----------------------------
# # -----------------------------
# # Helper: convert gradient → course & speed
# # -----------------------------
# def gradient_to_course_speed(grad_x, grad_y,
#                              cruise_speed=3.0,
#                              max_speed=10.0,
#                              min_speed=0.5):
#     """
#     Converts gradient vector to a course (rad) and speed (m/s).
#     - grad_x, grad_y: local gradient (East, North)
#     - cruise_speed: nominal forward speed when following a gradient
#     - max_speed: cap for safety
#     - min_speed: if gradient exists but small, enforce some motion
#     """
#     norm = np.hypot(grad_x, grad_y)
#     if norm < 1e-6:
#         return 0.0, 0.0  # no gradient, no movement

#     # Normalize gradient to unit vector
#     dx = grad_x / norm
#     dy = grad_y / norm

#     # Pick desired speed independent of raw gradient size
#     desired_speed = cruise_speed

#     # Apply safety limits
#     desired_speed = max(min_speed, min(desired_speed, max_speed))

#     # Course angle: atan2(East, North)
#     course = np.arctan2(dx, dy)

#     return course, desired_speed


# # -----------------------------
# # Main loop
# # -----------------------------
# while True:
#     if uav_state["x"] is not None:
#         # 1) Update UAV dot
#         uav_dot.set_data(uav_state["x"], uav_state["y"])

#         # 2) Compute probability gradient at current UAV position
#         grad_x, grad_y = compute_prob_gradient(prob_map, xv, yv, uav_state["x"], uav_state["y"])
#         print(f"UAV pos: x={uav_state['x']:.1f}, y={uav_state['y']:.1f}, grad_x={grad_x:.5f}, grad_y={grad_y:.5f}")


#         # 3) Convert gradient to course & speed
#         desired_course, desired_speed = gradient_to_course_speed(grad_x, grad_y, max_speed=velocity_max)

#         # 4) Send hybrid guidance
#         send_hybrid_guidance(ac_id, desired_course, desired_speed, uav_state)

#         # 5) Redraw map
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#     time.sleep(update_dt)


# # --- Live update loop ---
# while True:
#     if uav_state["x"] is not None:
#         uav_dot.set_data(uav_state["x"], uav_state["y"])
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#     time.sleep(0.1)
