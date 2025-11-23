import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import math
import json
from decimal import Decimal
from scipy.special import jv

# -------------------------
# Constants (SI Units)
# -------------------------
G_CONSTANT = 6.67430e-11
MASS_EARTH = 5.972e24       # kg
RADIUS_EARTH = 6.371e6      # Meters (approx 6,371 km)
GEO_RADIUS = 42164e3        # Geostationary orbit radius (approx 42,164 km)

# -------------------------
# Drag class (now 3D)
# -------------------------
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

        # Tangential Drag (opposite to velocity)
        a_tangential = -c_t * v

        # Radial Drag (along -r_hat, scaled by speed)
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
        # Keep Decimal for any long-precision parts, but use floats for the integrator
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
        self.asteroid_mass = float(asteroid_mass)
        self.asteroid_radius = float(asteroid_radius)
        # original code below
        self.physics = physics_service
        self.solver = solver_service
        self.drag = drag

        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []

        self.state = None
        self.t = 0.0
        self.destroyed = False
        self.physics = physics_service
        self.solver = solver_service
        self.drag = drag

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

    def set_initial_from_orbital_elements(self, a, b, start_true_anomaly=0.0,
                                          incl_deg=0.0, raan_deg=0.0, argp_deg=0.0):
        """
        Set initial state from orbital elements. Uses perifocal -> ECI transform to allow 3D.
        a, b: semi-major and semi-minor axes (meters)
        start_true_anomaly: theta (radians)
        incl_deg, raan_deg, argp_deg: inclination, RAAN, argument of periapsis in degrees
        """
        a = float(a)
        b = float(b)
        e = math.sqrt(max(0.0, a**2 - b**2)) / a
        theta = float(start_true_anomaly)

        # radial distance
        r_mag = a * (1 - e**2) / (1 + e * math.cos(theta))

        # Perifocal position
        r_pf = np.array([r_mag * math.cos(theta), r_mag * math.sin(theta), 0.0], dtype=float)

        # Standard gravitational parameter (float)
        mu = float(self.physics.G * (self.physics.M + self.physics.m))

        # Specific angular momentum magnitude
        h = math.sqrt(mu * a * (1 - e**2))

        # Perifocal velocity components (classical formula)
        v_pf = (mu / h) * np.array([-math.sin(theta), e + math.cos(theta), 0.0], dtype=float)

        # Rotation from perifocal to ECI: r_eci = Rz(Omega) * Rx(i) * Rz(omega) * r_pf
        i = math.radians(incl_deg)
        Omega = math.radians(raan_deg)
        omega = math.radians(argp_deg)

        # Rotation matrices
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

    def rk4_step(self, dt):
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
        self._append_history_point()

    def step(self, dt):
        if self.destroyed:
            return

        # Substepping for stability
        substeps = 50
        dt_sub = dt / substeps

        for _ in range(substeps):
            self.rk4_step(dt_sub)

            x, y, z, vx, vy, vz = self.state
            r = math.sqrt(x*x + y*y + z*z)

            if r <= RADIUS_EARTH + self.asteroid_radius:
                print("Meteor collided with Earth. Object destroyed.")
                self.destroyed = True
                return

        # Save one snapshot per real simulation step (already done in rk4)

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

def mainKepler3D():
    # Orbital parameters for an asteroid near Earth
    a_axis = 50000e3  # 50,000 km semi-major axis
    b_axis = 30000e3  # 30,000 km semi-minor axis

    # Provide some inclination and orientation so it's actually 3D
    inclination_deg = 45.0
    raan_deg = 10.0
    argp_deg = 20.0

    drag = DragForce(c_t0=1.5e-6, c_r0=0.0, decay=0.0)

    physics = PhysicsService(G_CONSTANT, MASS_EARTH, 0.0)
    solver = BesselAnomalySolver(max_iterations=40)
    sim = TrajectoryComputer(physics, solver, drag, asteroid_mass=1e9, asteroid_radius=1000.0)

    sim.set_initial_from_orbital_elements(a_axis, b_axis, start_true_anomaly=0.0,
                                         incl_deg=inclination_deg, raan_deg=raan_deg, argp_deg=argp_deg)

    # Time step: 120 seconds per high-level step
    dt = 120.0

    try:
        for step in range(0, 2000):
            sim.step(dt)
            if sim.destroyed:
                break
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
                "requested_max_points": max_save,
                "destroyed": bool(sim.destroyed),
                "asteroid_mass_kg": sim.asteroid_mass,
                "asteroid_radius_m": sim.asteroid_radius,
                "central_body": "Earth",
                "inclination_deg": inclination_deg,
                "raan_deg": raan_deg,
                "argp_deg": argp_deg
            },
            "data": output_list
        }

        return output_data

# -------------------------
# MAIN (3D output JSON)
# -------------------------
if __name__ == "__main__":
    mainKepler3D()