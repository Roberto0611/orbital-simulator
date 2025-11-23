import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import math
from decimal import Decimal
from scipy.special import jv

G_CONSTANT = 6.67430e-11
MASS_EARTH = 5.972e24       # kg
RADIUS_EARTH = 6.371e6      # Meters (approx 6,371 km)
GEO_RADIUS = 42164e3        # Geostationary orbit radius (approx 42,164 km)
METEOR_MASS = 1.0e12        # kg (approx 100 tons)
METEOR_RADIUS = 502.5       # Meters (approx 10m diameter)

class DragForce:
    """
    Tangential + radial drag with exponential coefficient decay.
    """
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

    def calculate_mean_motion(self, a_semimajor):
        a = Decimal(str(a_semimajor))
        numerator = self.G * (self.M + self.m)
        denominator = a ** 3
        return (numerator / denominator).sqrt()

    def get_mean_anomaly(self, n, t):
        return n * Decimal(str(t))


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
        self.destroyed = False

    def set_object_radius(self, radius):
        self.object_radius = float(radius)

    # --- NEW METHOD: Initialize with explicit Position and Velocity vectors ---
    def set_initial_state_vectors(self, r_vec, v_vec):
        """
        Manually set the initial state with vectors.
        r_vec: [x, y] (meters)
        v_vec: [vx, vy] (m/s)
        """
        x0, y0 = r_vec
        vx0, vy0 = v_vec
        
        self.state = np.array([x0, y0, vx0, vy0], dtype=float)
        self.t = 0.0
        # Clear previous history if any
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []
        self._append_history_point()

    def set_initial_from_orbital_elements(self, a, b, start_true_anomaly=0.0):
        a = float(a)
        b = float(b)
        e = math.sqrt(max(0.0, a**2 - b**2)) / a
        v_true = float(start_true_anomaly)
        r_mag = a * (1 - e**2) / (1 + e * math.cos(v_true))
        x0 = r_mag * math.cos(v_true)
        y0 = r_mag * math.sin(v_true)

        GM = float(self.physics.G * (self.physics.M + self.physics.m))
        v0 = math.sqrt(GM * (2.0 / r_mag - 1.0 / a))

        vx0 = -v0 * math.sin(v_true)
        vy0 = v0 * math.cos(v_true)

        self.state = np.array([x0, y0, vx0, vy0], dtype=float)
        self.t = 0.0
        self._append_history_point()

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


    def step(self, dt):
        if self.destroyed:
            return
        x, y, vx, vy = self.state
        r = math.hypot(x, y)
        if r > 2 * RADIUS_EARTH:
            substeps = 5
        elif r > 1.2 * RADIUS_EARTH:
            substeps = 10
        else:
            substeps = 40
        dt_sub = dt / substeps

        for _ in range(substeps):
            self.rk4_step(dt_sub)

            x, y, vx, vy = self.state
            r = math.hypot(x, y)

            if r <= (RADIUS_EARTH + self.object_radius):
                print("Meteor collided with Earth. Object destroyed.")
                self.destroyed = True
                return

        # Save only ONE point per dt step
        self._append_history_point()

    def get_history(self):
        return {
            "t": np.array(self.t_history),
            "x": np.array(self.x_history),
            "y": np.array(self.y_history),
            "r": np.array(self.r_history),
        }

# # -------------------------
# # Animation Service
# # -------------------------
# class AnimationService:
#     def __init__(self, simulator: TrajectoryComputer, dt=60.0):
#         self.sim = simulator
#         self.dt = float(dt)

#         self.fig, self.ax = plt.subplots(figsize=(10, 10))

#         # Earth Patch
#         self.earth_patch = Circle((0, 0), RADIUS_EARTH, color='dodgerblue', zorder=2, label='Earth')
#         self.ax.add_patch(self.earth_patch)

#         # Reference Orbit (Geostationary)
#         orbit_ref = Circle((0, 0), GEO_RADIUS, color='gray', fill=False, linestyle='--', alpha=0.3, label='GEO')
#         self.ax.add_patch(orbit_ref)

#         self.marker, = self.ax.plot([], [], 'ro', markersize=5, zorder=5)
#         self.trail, = self.ax.plot([], [], 'b-', alpha=0.5, linewidth=1, zorder=4)

#         # Set limits based on GEO radius
#         self.limit = GEO_RADIUS * 2.0
#         self.ax.set_xlim(-self.limit, self.limit)
#         self.ax.set_ylim(-self.limit, self.limit)
#         self.ax.set_aspect('equal')
#         self.ax.grid(True, linestyle=':', alpha=0.6)
#         self.ax.legend(loc='upper right')

#         self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, verticalalignment='top')

#     def update(self, frame):
#         for _ in range(5):
#             self.sim.step(self.dt)

#         hist = self.sim.get_history()
#         x = hist["x"][-1]
#         y = hist["y"][-1]
#         r = hist["r"][-1]
#         t = hist["t"][-1]

#         # ----------------------------
#         # Destroyed? Hide & stop.
#         # ----------------------------
#         if self.sim.destroyed:
#             self.marker.set_data([], [])
#             self.info_text.set_text("Meteor destroyed (impacted Earth).")
#             return self.marker, self.trail, self.info_text, self.earth_patch

#         self.marker.set_data([x], [y])
#         self.trail.set_data(hist["x"], hist["y"])

#         hours = t / 3600.0
#         dist_km = r / 1000.0
#         self.info_text.set_text(f"Time: {hours:.2f} hours\nDist: {dist_km:.1f} km")

#         return self.marker, self.trail, self.info_text, self.earth_patch

#     def start(self, frames=20000, interval=20):
#         ani = FuncAnimation(self.fig, self.update, frames=frames, interval=interval, blit=False, repeat=False)
#         plt.show()

def mainKepler2D():
    drag = DragForce(c_t0=4.0e-5, c_r0=5e-8, decay=1e-5)
    physics = PhysicsService(G_CONSTANT, MASS_EARTH, METEOR_MASS) 
    solver = BesselAnomalySolver(max_iterations=40)
    sim = TrajectoryComputer(physics, solver, drag)
    sim.set_object_radius(METEOR_RADIUS)
    initial_position = [3e7+5e7, -3.5e7]
    initial_velocity = [-4949.74, 4949.747]
    sim.set_initial_state_vectors(initial_position, initial_velocity)
    dt = 10

    try:
        for step in range(5000):
            sim.step(dt)
            if sim.destroyed:
                print(f"Simulation ended at step {step} due to destruction.")
                break
    finally:
        hist = sim.get_history()
        t_arr = hist["t"]
        x_arr = hist["x"]
        y_arr = hist["y"]
        r_arr = hist["r"]

        total_points = len(t_arr)
        max_save = 1000
        points_to_save = min(max_save, total_points)

        if total_points == 0:
            print("No history to save.")
        else:
            indices = np.linspace(0, total_points - 1, num=points_to_save)
            indices = np.round(indices).astype(int)
            indices = np.clip(indices, 0, total_points - 1)

            output_list = []
            for idx in indices:
                output_list.append({
                    "time_sec": float(t_arr[idx]),
                    "x_m": float(x_arr[idx]),
                    "y_m": float(y_arr[idx]),
                    "r_m": float(r_arr[idx])
                })

        output_data = {
            "metadata": {
                "saved_points": len(output_list) if 'output_list' in locals() else 0,
                "total_recorded_points": total_points,
                "meteor_mass_kg": METEOR_MASS,
                "meteor_radius_m": METEOR_RADIUS,
                "destroyed": bool(sim.destroyed),
                "central_body": "Earth"
            },
            "data": output_list if 'output_list' in locals() else []
        }

        return output_data

if __name__ == "__main__":
    mainKepler2D()