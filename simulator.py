import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import math
from decimal import Decimal
from scipy.special import jv

# -------------------------
# Constants (SI Units)
# -------------------------
G_CONSTANT = 6.67430e-11
MASS_SUN = 1.989e30
RADIUS_SUN = 6.9634e8       # Meters (approx 700,000 km)
AU = 1.496e11               # Astronomical Unit in meters

# -------------------------
# Drag class
# -------------------------
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

        # Tangential Drag
        a_tangential = -c_t * v

        # Radial Drag
        a_radial = np.zeros(2)
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
    def __init__(self, physics_service: PhysicsService, solver_service: BesselAnomalySolver, drag: DragForce):
        self.physics = physics_service
        self.solver = solver_service
        self.drag = drag

        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []

        self.state = None
        self.t = 0.0
        self.destroyed = False     # <-- NEW FLAG

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
        self._append_history_point()

    def step(self, dt):
        if self.destroyed:
            return

        substeps = 50
        dt_sub = dt / substeps

        for _ in range(substeps):
            self.rk4_step(dt_sub)

            x, y, vx, vy = self.state
            r = math.hypot(x, y)

            if r <= RADIUS_SUN:
                print("Meteor collided with the Sun. Object destroyed.")
                self.destroyed = True
                return

        # Save one snapshot per real simulation step
        self._append_history_point()


    def get_history(self):
        return {
            "t": np.array(self.t_history),
            "x": np.array(self.x_history),
            "y": np.array(self.y_history),
            "r": np.array(self.r_history),
        }

# -------------------------
# Animation Service
# -------------------------


def mainKepler():
    a_axis = 2.0 * AU
    b_axis = 1.5 * AU

    drag = DragForce(c_t0=1.5e-7, c_r0=0.0, decay=0.0)

    physics = PhysicsService(G_CONSTANT, MASS_SUN, 0.0)
    solver = BesselAnomalySolver(max_iterations=40)
    sim = TrajectoryComputer(physics, solver, drag)
    sim.set_initial_from_orbital_elements(a_axis, b_axis, start_true_anomaly=0.0)

    dt = 3600.0 * 8

    try:
        total_simulation_time = 3600.0 * 24 * 365 * 2
        steps = int(total_simulation_time / dt)

        for step in range(steps):
            if sim.destroyed:
                break
            sim.step(dt)
    finally:
        hist = sim.get_history()
        t_arr = hist["t"]
        x_arr = hist["x"]
        y_arr = hist["y"]
        r_arr = hist["r"]

        total_points = len(t_arr)
        # Desired maximum points to save
        max_save = 1000
        points_to_save = min(max_save, total_points)

        if total_points == 0:
            print("No history to save.")
            output_list = []
        else:
            # Evenly spaced indices from 0 .. total_points-1 (inclusive)
            indices = np.linspace(0, total_points - 1, num=points_to_save)
            # round to nearest integer index and convert to int
            indices = np.round(indices).astype(int)
            # safety: ensure indices are within bounds
            indices = np.clip(indices, 0, total_points - 1)

            # Build list of JSON-friendly dicts (preserves temporal order)
            output_list = []
            for idx in indices:
                output_list.append({
                    "time_sec": float(t_arr[idx]),
                    "x_m": float(x_arr[idx]),
                    "y_m": float(y_arr[idx]),
                    "r_m": float(r_arr[idx])
                })

        # Add small metadata wrapper so the file is self-describing
        output_data = {
            "metadata": {
                "saved_points": len(output_list),
                "total_recorded_points": total_points,
                "requested_max_points": max_save,
                "destroyed": bool(sim.destroyed)
            },
            "data": output_list
        }

        return output_data
# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    mainKepler()