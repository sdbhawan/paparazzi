import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import pymap3d as pm

PPRZ_HOME = os.getenv("PAPARAZZI_HOME", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

PPRZ_SRC = os.getenv("PAPARAZZI_SRC", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    '../../../..')))

# sys.path.append(PPRZ_HOME + "/var/lib/python")
sys.path.append(PPRZ_HOME + "/sw/ext/pprzlink/lib/v1.0/python")

from pprzlink.ivy import IvyMessagesInterface


def run_map():
    # --- Load XML ---
    xml_file = os.path.expanduser(
        "~/paparazzi2/paparazzi/conf/flight_plans/tudelft/rotwing_EHVB_Damian.xml"
    )
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # --- Flight plan reference origin ---
    lat0, lon0, alt0 = 52.1681551, 4.4126468, 0.0  # origin for ENU conversion

    # --- Parse waypoints ---
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

    # --- Polygons for zones ---
    ehvb_xy = np.array([waypoints[wp][:2] for wp in ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]])
    softgeo_xy = np.array([waypoints[wp][:2] for wp in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]])

    # --- Victim placement ---
    soft_poly = Polygon(softgeo_xy)
    victims = []
    while len(victims) < 10:
        x = np.random.uniform(soft_poly.bounds[0], soft_poly.bounds[2])
        y = np.random.uniform(soft_poly.bounds[1], soft_poly.bounds[3])
        if soft_poly.contains(Point(x, y)):
            victims.append([x, y])
    victims = np.array(victims)

    # --- Probability heatmap ---
    all_x = np.concatenate([
        [p[0] for p in waypoints.values()],
        ehvb_xy[:,0], softgeo_xy[:,0], victims[:,0]
    ])
    all_y = np.concatenate([
        [p[1] for p in waypoints.values()],
        ehvb_xy[:,1], softgeo_xy[:,1], victims[:,1]
    ])
    margin = 50
    min_x, max_x = np.min(all_x)-margin, np.max(all_x)+margin
    min_y, max_y = np.min(all_y)-margin, np.max(all_y)+margin

    grid_res = 2
    xv, yv = np.meshgrid(np.arange(min_x, max_x, grid_res),
                         np.arange(min_y, max_y, grid_res))
    prob_map = np.zeros_like(xv, dtype=float)
    for pos in victims:
        dist = np.sqrt((xv - pos[0])**2 + (yv - pos[1])**2)
        prob_map += np.exp(-(dist/20)**2)
    prob_map /= np.max(prob_map)

    # --- Plot setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(16,12))
    ax.set_title("UAV Mission Area (live ENU)")
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
    ax.legend()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # --- Track UAVs dynamically ---
    uav_dots = {}

    def message_recv(ac_id, msg):
        if msg.name == "GPS_INT":
            lat, lon, alt = float(msg['lat'])/1e7, float(msg['lon'])/1e7, float(msg['alt'])/1000.0
            x, y, z = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
            print(f"[GPS] AC {ac_id}: ENU=({x:.1f}, {y:.1f}, {z:.1f})")

            # Update or create UAV dot
            if ac_id in uav_dots:
                uav_dots[ac_id].set_offsets([x, y])
            else:
                uav_dots[ac_id] = ax.scatter(x, y, c='lime', marker='o', s=80, label=f'UAV {ac_id}')

            plt.draw()
            plt.pause(0.01)

    # --- Paparazzi interface ---
    interface = IvyMessagesInterface("rotwingframe", ivy_bus="127.255.255.255:2010")
    interface.subscribe(message_recv)

    print("✅ Live map running... (close the window to exit)")
    plt.show(block=True)
    # interface.loop()


if __name__ == "__main__":
    run_map()
