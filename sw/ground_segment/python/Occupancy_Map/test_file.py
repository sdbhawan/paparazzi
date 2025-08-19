# waypoints
# drone id? ---> not yet implemented
import os
import sys
PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

PPRZ_SRC = os.getenv("PAPARAZZI_SRC", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

# sys.path.append(PPRZ_HOME + "/var/lib/python")
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface

interface = IvyMessagesInterface("rotwingframe", ivy_bus= "127.255.255.255:2010")


def message_recv(ac_id, msg):
    if msg.name == "WP_LIST_ENU":
        n = int(msg['nb_wp'])   # convert to int
        print(f"\n[WP_LIST_ENU] received {n} waypoints from AC {ac_id}")
        for i in range(n):
            east = int(msg['east'][i])
            north = int(msg['north'][i])
            up = int(msg['up'][i])
            print(f" WP {i}: east={east}, north={north}, up={up}")

# Subscribe to all messages
interface.subscribe(message_recv)

# Keep Ivy running
interface.loop()


while True: 
    pass

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation settings ---
grid_size = (100, 100)
num_uavs = 3
num_victims = 5
step_size = 1.5  # UAV movement per timestep
detect_thresh = 0.8  # Probability threshold to "detect" a victim
np.random.seed(42)

# --- Victim positions (static) ---
victim_positions = np.random.randint(10, 90, size=(num_victims, 2))
victim_found = np.zeros(num_victims, dtype=bool)

# --- UAV initial positions ---
uav_positions = np.random.randint(0, 100, size=(num_uavs, 2)).astype(float)
uav_paths = [ [pos.copy()] for pos in uav_positions ]
uav_targets = [-1]*num_uavs  # assigned victim indices
ghost_waypoints = uav_positions.copy()

# --- Static probability map ---
x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
prob_map = np.zeros(grid_size)
for pos in victim_positions:
    # dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
    dist = np.sqrt((x - pos[1])**2 + (y - pos[0])**2)

    prob_map += np.exp(-(dist/10)**2)  # Gaussian around each victim
prob_map /= np.max(prob_map)  # normalize to [0,1]

# --- Simulation loop ---
timesteps = 50
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

for t in range(timesteps):
    ax.clear()
    ax.imshow(prob_map.T, origin='lower', cmap='coolwarm', alpha=0.5)
    
    # Assign UAVs to closest un-found victim
    for i in range(num_uavs):
        if uav_targets[i] == -1 or victim_found[uav_targets[i]]:
            # Choose nearest victim not yet found
            assigned = set()
            for i in range(num_uavs):
                if uav_targets[i] == -1 or victim_found[uav_targets[i]]:
                    un_found_idx = [idx for idx in range(num_victims) if not victim_found[idx] and idx not in assigned]
                    if len(un_found_idx) > 0:
                        distances = np.linalg.norm(uav_positions[i] - victim_positions[un_found_idx], axis=1)
                        chosen = un_found_idx[np.argmin(distances)]
                        uav_targets[i] = chosen
                        assigned.add(chosen)
                    else:
                        uav_targets[i] = -1


    # Update UAV positions towards ghost waypoints
    for i in range(num_uavs):
        target_idx = uav_targets[i]
        if target_idx == -1:
            continue
        # Ghost waypoint is victim position
        ghost_waypoints[i] = victim_positions[target_idx]
        direction = ghost_waypoints[i] - uav_positions[i]
        dist = np.linalg.norm(direction)
        if dist < step_size:
            uav_positions[i] = ghost_waypoints[i].copy()
        else:
            uav_positions[i] += (direction / dist) * step_size

        # Check if victim found
        if np.linalg.norm(uav_positions[i] - victim_positions[target_idx]) < 2.0:
            victim_found[target_idx] = True
            uav_targets[i] = -1

        # Record path
        uav_paths[i].append(uav_positions[i].copy())

    # --- Visualization ---
    ax.imshow(prob_map.T, origin='lower', cmap='coolwarm', alpha=0.5)
    ax.scatter(victim_positions[:,0], victim_positions[:,1], c='red', marker='X', s=100, label='Victims')
    ax.scatter(uav_positions[:,0], uav_positions[:,1], c='blue', label='UAVs')
    # Draw ghost waypoints arrows
    for i in range(num_uavs):
        if uav_targets[i] != -1:
            ax.arrow(uav_positions[i][0], uav_positions[i][1],
                     ghost_waypoints[i][0]-uav_positions[i][0],
                     ghost_waypoints[i][1]-uav_positions[i][1],
                     color='green', head_width=1.5, length_includes_head=True)
        path = np.array(uav_paths[i])
        ax.plot(path[:,0], path[:,1], 'b--', alpha=0.5)
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_title(f"Timestep {t}")
    plt.pause(0.1)

plt.ioff()
plt.show()

