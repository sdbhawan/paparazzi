import numpy as np
import pandas as pd
import time
import os
from copy import deepcopy
from shapely.geometry import Point, Polygon

# --- IMPORT FROM YOUR MAIN SCRIPT ---
from multi_uav_old import (
    UAVAgent, 
    BeliefMap, 
    DriftModel, 
    get_vsqp_power,
    soft_poly  # The Real EHVB Polygon
)

# ==========================================
# 1. HARDCODED CONFIGURATIONS
# ==========================================

# Fixed Victim Locations (Scenario A uses index 0, Scenario B uses 0 & 1)
FIXED_VICTIMS = np.array([
    [50.0, -200.0, 0.0],
    [50.0, -400.0, 0.0]
], dtype=float)

# Fixed UAV Start Positions (N agents take the first N positions)
MASTER_START_POSITIONS = [
    np.array([400.0, 0.0, 80.0]),
    np.array([50.0, -300.0, 40.0]),
    np.array([0.0, -200.0, 40.0]), 
    np.array([300.0, -40.0, 40.0]),
    np.array([80.0, -100.0, 80.0]),
    np.array([10.0, -250.0, 40.0]),
    np.array([0.0, 0.0, 80.0])
]

# ==========================================
# 2. GOLDEN MASTER PARAMETERS (Table 5.1)
# ==========================================
minx, miny, maxx, maxy = soft_poly.bounds

PARAMS = {
    # Environment
    'GRID_RES': 10,
    'GRID_W': int(maxx),
    'GRID_H': int(maxy),
    'MAX_STEPS': 300,    # 300 s Timeout
    
    # Drift & Diffusion
    'GAMMA_CONE': 1.0,         
    'SIGMA_DIFF': 1.0,         
    
    # IPP Planner
    'HORIZON_EXPLORE': 3,
    'HORIZON_TRACK': 2,
    'D_STEP_EXPLORE': 40,
    'D_STEP_TRACK': 20,
    'E_SCALE_EXPLORE': 100.0,
    'E_SCALE_TRACK': 40.0,
    'LAMBDA': 0.5,
    
    # FSM & Task Allocation
    'LOCK_DURATION': 8.0,
    'GAMMA_WIND_COST': 5.0,    
    'CONF_THRESH_MEAN': 0.40,
    'CONF_THRESH_PEAK': 0.75,  
    
    # Constraints
    'ACCEL_MAX': 6.0,
    'MIN_SAFE_DIST': 15.0     
}

# ==========================================
# 3. EXPERIMENTAL CONFIGURATION
# ==========================================
NUM_TRIALS = 5           
SWARM_SIZES = [3, 5, 7]   
SCENARIOS = [1, 2]        

results_log = []

def run_single_trial(seed, n_uavs, n_victims):
    np.random.seed(seed) # Strict Paired Sampling
    
    # --- A. INITIALIZE ENVIRONMENT ---
    # Wind/Drift are randomized per seed (Testing robust weather handling)
    wind_mag = np.random.uniform(0, 2.5) 
    wind_dir = np.random.uniform(0, 2*np.pi)
    v_wind = np.array([wind_mag * np.cos(wind_dir), wind_mag * np.sin(wind_dir), 0.0])
    
    drift_mag = np.random.uniform(0.5, 2.0)
    drift_dir = np.random.uniform(0, 2*np.pi)
    v_drift = np.array([drift_mag * np.cos(drift_dir), drift_mag * np.sin(drift_dir), 0.0])
    
    # Victims: Use HARDCODED list
    # We copy() to avoid drift modifying the global constant
    active_victims_pos = [v.copy() for v in FIXED_VICTIMS[:n_victims]]
    victim_states = [{'pos': p, 'found': False, 'confirmed': False} for p in active_victims_pos]

    # --- B. INITIALIZE SWARM ---
    # Agents: Use HARDCODED list
    agents = []
    for i in range(n_uavs):
        # Safety check if N > 7 (loop back to start)
        template_pos = MASTER_START_POSITIONS[i % len(MASTER_START_POSITIONS)]
        start_pos = template_pos.copy()
        
        # Initialize Agent
        # Pass nominal_alt from the Z-coordinate of the start position
        agent = UAVAgent(id=i, start_pos=start_pos, nominal_alt=start_pos[2]) 
        agents.append(agent)
        
    # --- C. INITIALIZE MAP ---
    belief_map = BeliefMap(w=PARAMS['GRID_W'], h=PARAMS['GRID_H'], res=PARAMS['GRID_RES'])
    
    trajectory_energy = 0.0
    min_separation = 1000.0
    abort_count = 0
    success = False
    sim_time = 0
    
    # --- D. MAIN LOOP ---
    for t in range(PARAMS['MAX_STEPS']):
        sim_time = t
        
        # 1. Update Victims (Drift)
        for v in victim_states:
            v['pos'] += v_drift * 1.0 
            
        # 2. Update Belief
        belief_map.propagate(v_drift, PARAMS['SIGMA_DIFF'])
        
        # 3. Task Allocation (Simplified Centralized)
        peak_coords = belief_map.get_peak_coords()
        peak_val = np.max(belief_map.belief)
        
        if peak_val > 0.6: 
            already_tracked = False
            for a in agents:
                if a.tracker['active']:
                    dist = np.linalg.norm(a.tracker['pos'] - peak_coords[:2])
                    if dist < 100.0: already_tracked = True
            
            if not already_tracked:
                best_agent = None
                best_cost = np.inf
                for a in agents:
                    if a.mode == 'explore':
                        # D_eff calculation
                        r = peak_coords[:2] - a.state[:2]
                        dist = np.linalg.norm(r)
                        r_hat = r / (dist + 1e-6)
                        wind_cost = PARAMS['GAMMA_WIND_COST'] * abs(np.dot(v_wind[:2], r_hat))
                        cost = dist + wind_cost
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_agent = a
                
                if best_agent is not None:
                    best_agent.mode = 'track'
                    best_agent.tracker['active'] = True
                    best_agent.tracker['pos'] = peak_coords[:2].copy()
                    best_agent.tracker['time'] = t
                    best_agent.tracker['phase'] = 'to_detection'

        # 4. Agent Steps
        step_energy = 0
        team_positions = [a.state[:3] for a in agents]
        
        for i in range(n_uavs):
            for j in range(i+1, n_uavs):
                dist = np.linalg.norm(team_positions[i] - team_positions[j])
                if dist < min_separation:
                    min_separation = dist

        for agent in agents:
            # EXECUTE STEP
            agent.step(t, 1.0, belief_map, v_drift, v_wind, PARAMS)
            
            # Energy
            p_inst = get_vsqp_power(np.linalg.norm(agent.velocity))
            step_energy += p_inst 
            
            if agent.just_aborted_tracking:
                abort_count += 1
                
            if agent.tracker['confirmed']:
                for v in victim_states:
                    if np.linalg.norm(v['pos'][:2] - agent.tracker['pos']) < 50.0:
                        v['confirmed'] = True
                agent.tracker['confirmed'] = False
                
        trajectory_energy += step_energy
        
        # 5. Check Success
        confirmed_count = sum([v['confirmed'] for v in victim_states])
        if confirmed_count == n_victims:
            success = True
            break
            
        # 6. Check Map Clearance
        if belief_map.entropy < 0.1:
            break

    # --- E. COMPILE RESULTS ---
    delta_entropy = belief_map.initial_entropy - belief_map.entropy
    e_spec = trajectory_energy / max(delta_entropy, 0.001)
    
    drift_errors = []
    if success:
        peak_pos = belief_map.get_peak_coords()
        for v in victim_states:
             drift_errors.append(np.linalg.norm(peak_pos - v['pos']))
    
    avg_error = np.mean(drift_errors) if drift_errors else 0.0

    return {
        "Seed": seed,
        "N_UAVs": n_uavs,
        "N_Victims": n_victims,
        "Success": success,
        "Time": sim_time,
        "Energy_Total": trajectory_energy,
        "Energy_Spec": e_spec,
        "Min_Dist": min_separation,
        "Aborts": abort_count,
        "Drift_Error": avg_error,
        "Wind_Mag": wind_mag
    }

# ==========================================
# 4. EXECUTION
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Starting High-Fidelity Phase I Monte Carlo: {NUM_TRIALS} trials...")

for seed in range(NUM_TRIALS):
    for n in SWARM_SIZES:
        for v in SCENARIOS:
            print(f"Running: Seed {seed} | N={n} | Victims={v}")
            data = run_single_trial(seed, n, v)
            results_log.append(data)
            
            # Incremental Save
            if seed % 5 == 0:
                partial_path = os.path.join(script_dir, "phase1_results_partial.csv")
                pd.DataFrame(results_log).to_csv(partial_path, index=False)

print("Simulation Complete!")
final_path = os.path.join(script_dir, "phase1_results_final.csv")
df = pd.DataFrame(results_log)
df.to_csv(final_path, index=False)
print(f"Saved results to: {final_path}")