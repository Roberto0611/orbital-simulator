import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math
import json
from decimal import Decimal
from scipy.special import jv

G_CONSTANT = 6.67430e-11
MASS_EARTH = 5.972e24       # kg
RADIUS_EARTH = 6.371e6      # Meters (approx 6,371 km)
GEO_RADIUS = 42164e3        # Geostationary orbit radius (approx 42,164 km)
METEOR_MASS = 1.0e12        # kg (approx 100 tons)
METEOR_RADIUS = 502.5       # Meters (approx 10m diameter -> actually 502.5m radius here)

class DragForce:
    """
    Tangential + radial drag with exponential coefficient decay (3D).
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
        a_radial = np.zeros(3)
        if r_norm > 0 and c_r != 0.0:
            r_hat = r / r_norm
            a_radial = -c_r * r_hat * speed

        return a_tangential + a_radial

# -------------------------
# Numerical integrator
# -------------------------
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
    def __init__(self, physics_service: PhysicsService, solver_service: BesselAnomalySolver, drag: DragForce, asteroid_mass=1e9, asteroid_radius=50.0):
        self.physics = physics_service
        self.solver = solver_service
        self.drag = drag
        
        self.asteroid_mass = float(asteroid_mass)
        self.asteroid_radius = float(asteroid_radius)

        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.r_history = []
        self.vx_history = []
        self.vy_history = []
        self.vz_history = []

        self.state = None
        self.t = 0.0
        self.destroyed = False

    def set_object_radius(self, radius):
        self.asteroid_radius = float(radius)

    def set_initial_state_vectors(self, r_vec, v_vec):
        """
        Manually set the initial state with vectors.
        r_vec: [x, y, z] (meters) - if 2D provided, z=0 assumed
        v_vec: [vx, vy, vz] (m/s) - if 2D provided, vz=0 assumed
        """
        # Handle 2D inputs by appending 0 for Z
        if len(r_vec) == 2:
            r_vec = list(r_vec) + [0.0]
        if len(v_vec) == 2:
            v_vec = list(v_vec) + [0.0]

        x0, y0, z0 = r_vec
        vx0, vy0, vz0 = v_vec
        
        self.state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)
        self.t = 0.0
        
        # Clear history
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.z_history = []
        self.r_history = []
        self.vx_history = []
        self.vy_history = []
        self.vz_history = []
        
        self._append_history_point()

    def set_initial_from_orbital_elements(self, a, b, start_true_anomaly=0.0,
                                          incl_deg=0.0, raan_deg=0.0, argp_deg=0.0):
        a = float(a)
        b = float(b)
        e = math.sqrt(max(0.0, a**2 - b**2)) / a
        theta = float(start_true_anomaly)
        r_mag = a * (1 - e**2) / (1 + e * math.cos(theta))
        r_pf = np.array([r_mag * math.cos(theta), r_mag * math.sin(theta), 0.0], dtype=float)
        mu = float(self.physics.G * (self.physics.M + self.physics.m))
        h = math.sqrt(mu * a * (1 - e**2))
        v_pf = (mu / h) * np.array([-math.sin(theta), e + math.cos(theta), 0.0], dtype=float)
        i = math.radians(incl_deg)
        Omega = math.radians(raan_deg)
        omega = math.radians(argp_deg)

        def Rz(angle):
            c = math.cos(angle); s = math.sin(angle)
            return np.array([[c, -s, 0.0],[s, c, 0.0],[0.0,0.0,1.0]])

        def Rx(angle):
            c = math.cos(angle); s = math.sin(angle)
            return np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]])

        R = Rz(Omega) @ Rx(i) @ Rz(omega)
        r_eci = R @ r_pf
        v_eci = R @ v_pf
        x0, y0, z0 = r_eci
        vx0, vy0, vz0 = v_eci
        self.state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)
        self.t = 0.0
        self._append_history_point()

    def _append_history_point(self):
        x, y, z, vx, vy, vz = self.state
        r = math.sqrt(x*x + y*y + z*z)
        self.t_history.append(self.t)
        self.x_history.append(float(x))
        self.y_history.append(float(y))
        self.z_history.append(float(z))
        self.r_history.append(float(r))
        self.vx_history.append(float(vx))
        self.vy_history.append(float(vy))
        self.vz_history.append(float(vz))

    def _acceleration(self, state, t):
        x, y, z, vx, vy, vz = state
        r_vec = np.array([x, y, z], dtype=float)
        v_vec = np.array([vx, vy, vz], dtype=float)
        r_norm = np.linalg.norm(r_vec)

        mu = float(self.physics.G * (self.physics.M + self.physics.m))
        if r_norm == 0.0:
            a_grav = np.zeros(3)
        else:
            a_grav = - (mu / (r_norm**3)) * r_vec

        a_drag = self.drag.acceleration(r_vec, v_vec, t)
        return a_grav + a_drag

    def _rk4_no_history(self, dt):
        s = self.state.copy()
        t_local = self.t

        def deriv(s_local, tau):
            x, y, z, vx, vy, vz = s_local
            a = self._acceleration(s_local, tau)
            return np.array([vx, vy, vz, a[0], a[1], a[2]])

        k1 = deriv(s, t_local)
        k2 = deriv(s + 0.5 * dt * k1, t_local + 0.5 * dt)
        k3 = deriv(s + 0.5 * dt * k2, t_local + 0.5 * dt)
        k4 = deriv(s + dt * k3, t_local + dt)

        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t += dt


    def step(self, dt):
        if self.destroyed:
            return
        x, y, z, vx, vy, vz = self.state
        r = math.sqrt(x*x + y*y + z*z)
        if r > 2 * RADIUS_EARTH:
            substeps = 5
        elif r > 1.2 * RADIUS_EARTH:
            substeps = 10
        else:
            substeps = 40

        dt_sub = dt / substeps

        for _ in range(substeps):
            self._rk4_no_history(dt_sub)

            x, y, z, vx, vy, vz = self.state
            r = math.sqrt(x*x + y*y + z*z)

            if r <= RADIUS_EARTH + self.asteroid_radius:
                print("Meteor collided with Earth. Object destroyed.")
                self.destroyed = True
                return

        self.t_history.append(self.t)
        self.x_history.append(self.state[0])
        self.y_history.append(self.state[1])
        self.z_history.append(self.state[2])
        self.r_history.append(r)
        self.vx_history.append(self.state[3])
        self.vy_history.append(self.state[4])
        self.vz_history.append(self.state[5])


    def get_history(self):
        return {
            "t": np.array(self.t_history),
            "x": np.array(self.x_history),
            "y": np.array(self.y_history),
            "z": np.array(self.z_history),
            "r": np.array(self.r_history),
            "vx": np.array(self.vx_history),
            "vy": np.array(self.vy_history),
            "vz": np.array(self.vz_history),
        }

# -------------------------
# Animation Service (3D)
# -------------------------
# class AnimationService:
#     def __init__(self, simulator: TrajectoryComputer, dt=60.0):
#         self.sim = simulator
#         self.dt = float(dt)

#         self.fig = plt.figure(figsize=(10, 10))
#         self.ax = self.fig.add_subplot(111, projection='3d')

#         # Earth sphere
#         u = np.linspace(0, 2 * np.pi, 40)
#         v = np.linspace(0, np.pi, 20)
#         x_s = RADIUS_EARTH * np.outer(np.cos(u), np.sin(v))
#         y_s = RADIUS_EARTH * np.outer(np.sin(u), np.sin(v))
#         z_s = RADIUS_EARTH * np.outer(np.ones_like(u), np.cos(v))
#         self.ax.plot_surface(x_s, y_s, z_s, color='dodgerblue', alpha=0.6)

#         max_lim = GEO_RADIUS * 2.0
#         self.ax.set_xlim(-max_lim, max_lim)
#         self.ax.set_ylim(-max_lim, max_lim)
#         self.ax.set_zlim(-max_lim, max_lim)
#         self.ax.set_box_aspect([1,1,1])

#         self.trail_line, = self.ax.plot([], [], [], 'b-', alpha=0.6)
#         self.marker_line, = self.ax.plot([], [], [], 'ro')
#         self.info_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes)

#     def update(self, frame):
#         for _ in range(5):
#             self.sim.step(self.dt)

#         hist = self.sim.get_history()
#         x = hist["x"][-1]
#         y = hist["y"][-1]
#         z = hist["z"][-1]
#         r = hist["r"][-1]
#         t = hist["t"][-1]

#         if self.sim.destroyed:
#             self.marker_line.set_data([], [])
#             self.marker_line.set_3d_properties([])
#             self.info_text.set_text("Meteor destroyed (impacted Earth).")
#             return self.marker_line, self.trail_line, self.info_text

#         self.marker_line.set_data([x], [y])
#         self.marker_line.set_3d_properties([z])
#         self.trail_line.set_data(hist["x"], hist["y"])
#         self.trail_line.set_3d_properties(hist["z"])

#         hours = t / 3600.0
#         dist_km = r / 1000.0
#         self.info_text.set_text(f"Time: {hours:.2f} hours\nDist: {dist_km:.1f} km")

#         return self.marker_line, self.trail_line, self.info_text

#     def start(self, frames=20000, interval=20):
#         ani = FuncAnimation(self.fig, self.update, frames=frames, interval=interval, blit=False, repeat=False)
#         plt.show()

def mainKepler3D():
    drag = DragForce(c_t0=4.0e-5, c_r0=5e-8, decay=1e-5)
    physics = PhysicsService(G_CONSTANT, MASS_EARTH, METEOR_MASS) 
    solver = BesselAnomalySolver(max_iterations=40)
    sim = TrajectoryComputer(physics, solver, drag, asteroid_mass=METEOR_MASS, asteroid_radius=METEOR_RADIUS)
    sim.set_object_radius(METEOR_RADIUS)
    initial_position = [3e7+5e7, -3.5e7, -3.5e7]
    initial_velocity = [-4330.12, 4330.12, 4330.12]
    sim.set_initial_state_vectors(initial_position, initial_velocity)
    dt = 10.0

    try:
        while not sim.destroyed:
            sim.step(dt)
    finally:
        hist = sim.get_history()
        t_arr = hist["t"]
        x_arr = hist["x"]
        y_arr = hist["y"]
        z_arr = hist["z"]
        r_arr = hist["r"]
        vx_arr = hist["vx"]
        vy_arr = hist["vy"]
        vz_arr = hist["vz"]

        total_points = len(t_arr)
        max_save = 1000
        points_to_save = min(max_save, total_points)

        if total_points == 0:
            print("No history to save.")
            output_list = []
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
                    "z_m": float(z_arr[idx]),
                    "r_m": float(r_arr[idx]),
                    "vx_m_s": float(vx_arr[idx]),
                    "vy_m_s": float(vy_arr[idx]),
                    "vz_m_s": float(vz_arr[idx])
                })

        output_data = {
            "metadata": {
                "saved_points": len(output_list),
                "total_recorded_points": total_points,
                "meteor_mass_kg": METEOR_MASS,
                "meteor_radius_m": METEOR_RADIUS,
                "destroyed": bool(sim.destroyed),
                "central_body": "Earth"
            },
            "data": output_list
        }

        return output_data

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    mainKepler3D()