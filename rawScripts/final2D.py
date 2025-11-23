import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import math
import json
from decimal import Decimal
from scipy.special import jv

G_CONSTANT = 6.67430e-11
MASS_EARTH = 5.972e24       # kg
RADIUS_EARTH = 6.371e6      # Meters
GEO_RADIUS = 42164e3        # Meters
METEOR_MASS = 1.0e12        # kg
METEOR_RADIUS = 502.5       # Meters
SIM_BOUNDARY = 1.0e8        # Meters (Boundary for escape)


class DragForce:
    def __init__(self, c_t0=1e-8, c_r0=0.0, decay=0.0):
        self.c_t0 = float(c_t0)
        self.c_r0 = float(c_r0)
        self.decay = float(decay)

    def coeffs(self, t):
        factor = math.exp(-self.decay * t) if self.decay != 0.0 else 1.0
        return self.c_t0 * factor, self.c_r0 * factor

    def acceleration(self, r_vec, v_vec, t):
        c_t, c_r = self.coeffs(t)
        v = np.array(v_vec, dtype=float)
        r = np.array(r_vec, dtype=float)
        speed = np.linalg.norm(v)
        r_norm = np.linalg.norm(r)
        a_tangential = -c_t * v
        a_radial = np.zeros(2)
        if r_norm > 0 and c_r != 0.0:
            r_hat = r / r_norm
            a_radial = -c_r * r_hat * speed
        return a_tangential + a_radial

class PhysicsService:
    def __init__(self, G_val, Mass_primary, mass_secondary):
        self.G = Decimal(str(G_val))
        self.M = Decimal(str(Mass_primary))
        self.m = Decimal(str(mass_secondary))

class BesselAnomalySolver:
    def __init__(self, max_iterations=50):
        self.max_iter = max_iterations

    def solve_E(self, M, e):
        M_float = float(M)
        e_float = float(e)
        summation = Decimal(0)
        for s in range(1, self.max_iter + 1):
            bessel_term = jv(s, s * e_float)
            sin_term = math.sin(s * M_float)
            term = (Decimal(1) / Decimal(s)) * Decimal(bessel_term) * Decimal(sin_term)
            summation += term
        return M + (Decimal(2) * summation)

class TrajectoryComputer:
    def __init__(self, physics_service: PhysicsService, solver_service: BesselAnomalySolver, drag: DragForce):
        self.physics = physics_service
        self.solver = solver_service
        self.drag = drag        
        self.object_radius = 0.0 
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []
        self.state = None
        self.t = 0.0
        
        # End states
        self.destroyed = False # Hits Earth
        self.escaped = False   # Exits boundary
        self.has_collided = True # Default per instructions (fails if not escaped)

    def set_object_radius(self, radius):
        self.object_radius = float(radius)

    def set_initial_state_vectors(self, r_vec, v_vec):
        x0, y0 = r_vec
        vx0, vy0 = v_vec
        self.state = np.array([x0, y0, vx0, vy0], dtype=float)
        self.t = 0.0
        self._append_history_point()

    def set_initial_from_orbital_elements(self, a, b, start_true_anomaly=0.0):
        pass 

    def _append_history_point(self):
        x, y, vx, vy = self.state
        r = math.hypot(x, y)
        self.t_history.append(self.t)
        self.x_history.append(float(x))
        self.y_history.append(float(y))
        self.r_history.append(float(r))

    def _acceleration(self, state, t):
        x, y, vx, vy = state
        r_vec = np.array([x, y], dtype=float)
        v_vec = np.array([vx, vy], dtype=float)
        r_norm = np.linalg.norm(r_vec)
        GM = float(self.physics.G * (self.physics.M + self.physics.m))
        if r_norm == 0.0:
            a_grav = np.array([0.0, 0.0])
        else:
            a_grav = - (GM / (r_norm**3)) * r_vec
        a_drag = self.drag.acceleration(r_vec, v_vec, t)
        return a_grav + a_drag

    def rk4_step(self, dt):
        s = self.state.copy()
        t_local = self.t
        def deriv(s_local, tau):
            x, y, vx, vy = s_local
            a = self._acceleration(s_local, tau)
            return np.array([vx, vy, a[0], a[1]])
        k1 = deriv(s, t_local)
        k2 = deriv(s + 0.5 * dt * k1, t_local + 0.5 * dt)
        k3 = deriv(s + 0.5 * dt * k2, t_local + 0.5 * dt)
        k4 = deriv(s + dt * k3, t_local + dt)
        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t += dt
        self._append_history_point()

    def step(self, dt):
        if self.destroyed or self.escaped:
            return

        substeps = 50
        dt_sub = dt / substeps
        for _ in range(substeps):
            self.rk4_step(dt_sub)
            x, y, vx, vy = self.state
            r = math.hypot(x, y)
            
            if r <= (RADIUS_EARTH + self.object_radius):
                self.destroyed = True
                self.has_collided = True
                print("Meteor collided with Earth.")
                return

            if abs(x) > SIM_BOUNDARY or abs(y) > SIM_BOUNDARY:
                self.escaped = True
                self.has_collided = False
                print("Meteor escaped boundary.")
                return

        self._append_history_point()

    def get_history(self):
        return {
            "t": np.array(self.t_history),
            "x": np.array(self.x_history),
            "y": np.array(self.y_history),
            "r": np.array(self.r_history),
        }

class DeflectorSatellite:
    def __init__(self, x, y, vx, vy, target_id):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.target_id = target_id
        self.active = True
        self.thrust = 4000.0 
        self.fuel = 50.0
        self.deploy_timestamp = None
        self.detonation_timestamp = None
        self.history = [] # Stores {t, x, y, vx, vy}

    def update(self, dt, target_pos, target_vel, current_sim_time):
        if not self.active: return np.inf
        self.history.append({
            "t": float(current_sim_time),
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1])
        })
        
        v_norm = np.linalg.norm(target_vel)
        if v_norm > 0:
            v_hat = target_vel / v_norm
            aim_point = target_pos - (v_hat * 1200.0) 
        else:
            aim_point = target_pos

        r_vec = aim_point - self.pos
        dist = np.linalg.norm(r_vec)
        real_dist = np.linalg.norm(target_pos - self.pos)

        if self.fuel <= 0:
             self.active = False
             return real_dist
        
        closing_speed = np.linalg.norm(target_vel - self.vel)
        if closing_speed < 1.0: closing_speed = 1.0
        time_to_go = dist / closing_speed
        future_aim_point = aim_point + target_vel * time_to_go
        
        intercept_vec = future_aim_point - self.pos
        intercept_dist = np.linalg.norm(intercept_vec)
        
        if intercept_dist > 0:
            desired_dir = intercept_vec / intercept_dist
            self.vel += desired_dir * self.thrust * dt
            self.fuel -= dt
            
        self.pos += self.vel * dt
        return real_dist

class InterceptorRocket:
    def __init__(self):
        self.pos = np.array([RADIUS_EARTH * 0.7, RADIUS_EARTH * 0.7], dtype=float) 
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.active = False
        self.deployed_satellites = False
        self.acceleration_mag = 300.0         
        self.launch_timestamp = None
        self.history = [] 

    def launch(self, t):
        self.active = True
        self.launch_timestamp = float(t)

    def update(self, dt, target_pos, target_vel, current_sim_time):
        if not self.active: return None
        
        self.history.append({
            "t": float(current_sim_time),
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1])
        })
        
        # Guidance
        v_norm = np.linalg.norm(target_vel)
        aim_bias = np.array([0.0, 0.0])
        if v_norm > 0:
            aim_bias = - (target_vel / v_norm) * 5000.0

        r_vec = (target_pos + aim_bias) - self.pos
        dist = np.linalg.norm(r_vec)
        
        time_to_go = dist / 8000.0 
        future_pos = (target_pos + aim_bias) + target_vel * (time_to_go * 0.9)
        
        intercept_vec = future_pos - self.pos
        idist = np.linalg.norm(intercept_vec)
        if idist > 0:
            dir = intercept_vec / idist
            self.vel += dir * self.acceleration_mag * dt
            
        self.pos += self.vel * dt
        return dist

class DefenseCoordinator:
    def __init__(self):
        self.rocket = InterceptorRocket()
        self.satellites = []
        self.launched = False
        self.status_message = "Tracking"
        self.explosion_events = [] 
        
        self.deploy_distance_threshold = 5.0e6 
        self.impact_distance_threshold = 4.0e5 
        self.kick_delta_v_per_sat = 2000.0 

    def update(self, dt, sim_instance):
        mx, my, mvx, mvy = sim_instance.state
        m_pos = np.array([mx, my])
        m_vel = np.array([mvx, mvy])
        m_dist = np.linalg.norm(m_pos)
        current_time = sim_instance.t

        # Launch Logic
        if not self.launched and not self.rocket.active:
             if mx < -2.0e6 and m_dist < GEO_RADIUS * 1.3:
                self.rocket.launch(current_time)
                self.launched = True
                self.status_message = "INTERCEPTOR LAUNCHED"

        # Rocket Update
        rocket_dist = self.rocket.update(dt, m_pos, m_vel, current_time)

        # Deploy Trigger
        if self.rocket.active and not self.rocket.deployed_satellites:
            if rocket_dist is not None and rocket_dist < self.deploy_distance_threshold:
                self._deploy_satellites(current_time)
                self.rocket.active = False
                self.status_message = "SATELLITES DEPLOYED"

        # Satellite Updates
        detonation_occurred = False
        for sat in self.satellites:
            if sat.active:
                dist = sat.update(dt, m_pos, m_vel, current_time)
                
                if dist < self.impact_distance_threshold:
                    self._apply_impulse(sim_instance, m_pos, m_vel)
                    self.explosion_events.append(sat.pos.copy())
                    
                    # Log detonation time
                    sat.detonation_timestamp = float(current_time)
                    sat.active = False
                    detonation_occurred = True
                    print(f"*** BOOM *** Impact. Applied kick.")

        if detonation_occurred:
             self.status_message = "TRAJECTORY ALTERED"
        elif self.launched and self.rocket.deployed_satellites and all(not s.active for s in self.satellites):
             if not sim_instance.escaped:
                 self.status_message = "All Satellites Detonated"
             else:
                 self.status_message = "Target Escaped"

    def _deploy_satellites(self, t):
        self.rocket.deployed_satellites = True
        r_pos = self.rocket.pos
        r_vel = self.rocket.vel
        spread = 2000.0
        offsets = [
            np.array([spread, 0]), np.array([-spread, spread]), np.array([-spread, -spread])
        ]
        for i in range(3):
            sat = DeflectorSatellite(r_pos[0]+offsets[i][0], r_pos[1]+offsets[i][1], r_vel[0], r_vel[1], i)
            sat.deploy_timestamp = float(t)
            self.satellites.append(sat)

    def _apply_impulse(self, sim, m_pos, m_vel):
        v_norm = np.linalg.norm(m_vel)
        r_norm = np.linalg.norm(m_pos)
        if v_norm == 0 or r_norm == 0: return

        v_hat = m_vel / v_norm
        r_hat = m_pos / r_norm

        # Prograde + Outward mix
        impulse_dir = (v_hat * 0.7) + (r_hat * 0.3)
        impulse_dir = impulse_dir / np.linalg.norm(impulse_dir)
        
        delta_v = impulse_dir * self.kick_delta_v_per_sat
        
        sim.state[2] += delta_v[0]
        sim.state[3] += delta_v[1]
        sim._append_history_point()

    def get_rocket_pos(self):
        if self.rocket.active: return [self.rocket.pos[0]], [self.rocket.pos[1]]
        return [], []

    def get_satellite_positions(self):
        xs, ys = [], []
        for sat in self.satellites:
            if sat.active: xs.append(sat.pos[0]); ys.append(sat.pos[1])
        return xs, ys

class AnimationService:
    def __init__(self, simulator: TrajectoryComputer, dt=60.0, defense_system=None):
        self.sim = simulator
        self.dt = float(dt)
        self.defense_system = defense_system
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.earth_patch = Circle((0, 0), RADIUS_EARTH, color='dodgerblue', zorder=2, label='Earth')
        self.ax.add_patch(self.earth_patch)
        
        orbit_ref = Circle((0, 0), GEO_RADIUS, color='gray', fill=False, linestyle='--', alpha=0.3, label='GEO')
        self.ax.add_patch(orbit_ref)

        self.marker, = self.ax.plot([], [], 'ro', markersize=3, zorder=5, label='Meteor')
        self.trail, = self.ax.plot([], [], 'r-', alpha=0.5, linewidth=1, zorder=4)

        self.rocket_marker, = self.ax.plot([], [], 'o', color='purple', markersize=4, zorder=6, label='Interceptor')
        self.sat_markers, = self.ax.plot([], [], 'o', color='#00FF00', markeredgecolor='black', markersize=3, zorder=7, label='Deflectors')
        self.explosion_marker, = self.ax.plot([], [], 'o', color='orange', markeredgecolor='red', markersize=12, alpha=0.9, zorder=8, label='Detonation')

        self.limit = 1.0e8 # View matches boundary
        self.ax.set_xlim(-self.limit, self.limit)
        self.ax.set_ylim(-self.limit, self.limit)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend(loc='upper right')
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, verticalalignment='top')

    def update(self, frame):
        if not self.sim.destroyed and not self.sim.escaped:
            for _ in range(5):
                self.sim.step(self.dt)
                if self.sim.destroyed or self.sim.escaped: break
                if self.defense_system:
                    self.defense_system.update(self.dt, self.sim)

        hist = self.sim.get_history()
        x = hist["x"][-1]
        y = hist["y"][-1]
        r = hist["r"][-1]
        t = hist["t"][-1]
        
        if self.sim.destroyed:
            self.marker.set_data([], [])
            self.info_text.set_text("Meteor collided with Earth.")
            return self.marker, self.trail, self.info_text, self.earth_patch, self.rocket_marker, self.sat_markers, self.explosion_marker
        
        if self.sim.escaped:
             self.marker.set_data([x], [y])
             self.info_text.set_text("Target Escaped Boundary.\nSafe.")
             return self.marker, self.trail, self.info_text, self.earth_patch, self.rocket_marker, self.sat_markers, self.explosion_marker

        self.marker.set_data([x], [y])
        self.trail.set_data(hist["x"], hist["y"])

        exp_x, exp_y = [], []
        if self.defense_system:
            rx, ry = self.defense_system.get_rocket_pos()
            sx, sy = self.defense_system.get_satellite_positions()
            self.rocket_marker.set_data(rx, ry)
            self.sat_markers.set_data(sx, sy)
            
            if self.defense_system.explosion_events:
                for pos in self.defense_system.explosion_events:
                    exp_x.append(pos[0])
                    exp_y.append(pos[1])
                self.defense_system.explosion_events = [] 

        self.explosion_marker.set_data(exp_x, exp_y)
        
        hours = t / 3600.0
        dist_km = r / 1000.0
        status = self.defense_system.status_message if self.defense_system else ""
        vx, vy = self.sim.state[2], self.sim.state[3]
        v_tot = math.hypot(vx, vy)
        
        self.info_text.set_text(f"Time: {hours:.2f} h\nDist: {dist_km:.0f} km\nVel: {v_tot:.0f} m/s\nStatus: {status}")

        return self.marker, self.trail, self.info_text, self.earth_patch, self.rocket_marker, self.sat_markers, self.explosion_marker

    def start(self, frames=20000, interval=20):
        ani = FuncAnimation(self.fig, self.update, frames=frames, interval=interval, blit=True, repeat=False)
        plt.show()

if __name__ == "__main__":
    drag = DragForce(c_t0=4.0e-5, c_r0=5e-8, decay=1e-5)
    physics = PhysicsService(G_CONSTANT, MASS_EARTH, METEOR_MASS) 
    solver = BesselAnomalySolver(max_iterations=40)
    sim = TrajectoryComputer(physics, solver, drag)
    sim.set_object_radius(METEOR_RADIUS)
    
    initial_position = [3e7+5e7, -3.5e7]
    initial_velocity = [-4949.74, 4949.747]
    sim.set_initial_state_vectors(initial_position, initial_velocity)
    
    defense = DefenseCoordinator()
    
    dt = 5.0 
    animator = AnimationService(simulator=sim, dt=dt, defense_system=defense)

    try:
        animator.start(frames=8000, interval=10)
    finally:
        hist = sim.get_history()
        t_arr, x_arr, y_arr, r_arr = hist["t"], hist["x"], hist["y"], hist["r"]
        
        total_points = len(t_arr)
        points_to_save = min(1000, total_points)
        output_list = []
        if total_points > 0:
            indices = np.round(np.linspace(0, total_points - 1, num=points_to_save)).astype(int)
            output_list = [{"time_sec": float(t_arr[i]), "x_m": float(x_arr[i]), "y_m": float(y_arr[i]), "r_m": float(r_arr[i])} for i in indices]

        rocket_data = {
            "launch_timestamp": defense.rocket.launch_timestamp,
            "trajectory": defense.rocket.history
        }

        sat_data_list = []
        for sat in defense.satellites:
            sat_data_list.append({
                "id": sat.target_id,
                "deploy_timestamp": sat.deploy_timestamp,
                "detonation_timestamp": sat.detonation_timestamp,
                "trajectory": sat.history
            })

        output_data = {
            "metadata": {
                "meteor_mass_kg": METEOR_MASS,
                "has_collided": bool(sim.has_collided),
                "termination_reason": "Escaped" if sim.escaped else ("Impact" if sim.destroyed else "Timeout")
            },
            "data": output_list,
            "rocket": rocket_data,
            "satellites": sat_data_list
        }
        
        filename = "simulation_data_deflected.json"
        with open(filename, "w") as f: json.dump(output_data, f, indent=4)
        print(f"Simulation ended. has_collided={output_data['metadata']['has_collided']}. Data saved to {filename}")