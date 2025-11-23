import numpy as np
import math
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

        # history arrays
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []

        self.state = None
        self.t = 0.0

        # Precompute GM as float for speed
        self.GM_float = float(self.physics.G * (self.physics.M + self.physics.m))

        # End states
        self.destroyed = False
        self.escaped = False
        self.has_collided = False  # start false; set True on collision

        # NEW: pending delta-v (2D) applied to the state at start of the next RK4 step
        self.pending_delta_v = np.zeros(2, dtype=float)

    def set_object_radius(self, radius):
        self.object_radius = float(radius)

    def set_initial_state_vectors(self, r_vec, v_vec):
        """
        Set initial state and reset histories. Immediately logs the first history point.
        """
        x0, y0 = r_vec
        vx0, vy0 = v_vec
        self.state = np.array([x0, y0, vx0, vy0], dtype=float)
        self.t = 0.0

        # reset history (important when rerunning multiple sims in same process)
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []

        # Append initial history point
        self._append_history_point()

    def set_initial_from_orbital_elements(self, a, b, start_true_anomaly=0.0):
        # simplified orbital initializer (kept for compatibility)
        a = float(a)
        b = float(b)
        e = math.sqrt(max(0.0, a**2 - b**2)) / a
        v_true = float(start_true_anomaly)
        r_mag = a * (1 - e**2) / (1 + e * math.cos(v_true))
        x0 = r_mag * math.cos(v_true)
        y0 = r_mag * math.sin(v_true)

        GM = self.GM_float
        v0 = math.sqrt(GM * (2.0 / r_mag - 1.0 / a))

        vx0 = -v0 * math.sin(v_true)
        vy0 = v0 * math.cos(v_true)

        self.state = np.array([x0, y0, vx0, vy0], dtype=float)
        self.t = 0.0
        self.t_history = []
        self.x_history = []
        self.y_history = []
        self.r_history = []
        self._append_history_point()

    def _append_history_point(self):
        """
        Append a single snapshot (called once per main dt).
        """
        x, y, vx, vy = self.state
        r = math.sqrt(x * x + y * y)
        self.t_history.append(float(self.t))
        self.x_history.append(float(x))
        self.y_history.append(float(y))
        self.r_history.append(float(r))

    def _acceleration(self, state, t):
        x, y, vx, vy = state
        r_vec = np.array([x, y], dtype=float)
        v_vec = np.array([vx, vy], dtype=float)
        r_norm = math.sqrt(x * x + y * y)

        if r_norm == 0.0:
            a_grav = np.array([0.0, 0.0])
        else:
            a_grav = - (self.GM_float / (r_norm ** 3)) * r_vec

        a_drag = self.drag.acceleration(r_vec, v_vec, t)
        return a_grav + a_drag

    def rk4_step(self, dt):
        """
        One RK4 micro-step. We apply any pending_delta_v to the local integrator
        state `s` before computing the k1..k4 so the impulse is respected.
        """
        s = self.state.copy()
        t_local = self.t

        # APPLY pending delta-v *to the local state copy* before integration.
        if np.any(self.pending_delta_v):
            # add instantaneous velocity change
            s[2] += float(self.pending_delta_v[0])
            s[3] += float(self.pending_delta_v[1])
            # clear pending; it should only apply once
            self.pending_delta_v[:] = 0.0

        def deriv(s_local, tau):
            x, y, vx, vy = s_local
            a = self._acceleration(s_local, tau)
            return np.array([vx, vy, a[0], a[1]])

        k1 = deriv(s, t_local)
        k2 = deriv(s + 0.5 * dt * k1, t_local + 0.5 * dt)
        k3 = deriv(s + 0.5 * dt * k2, t_local + 0.5 * dt)
        k4 = deriv(s + dt * k3, t_local + dt)

        # Update true state at end of micro-step
        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += dt
        # NOTE: intentionally do NOT append history here (we only want one record per main time step)

    def step(self, dt):
        """
        Perform one main step. Internally performs `substeps` RK4 micro-steps.
        Only one history snapshot is appended per call to step() (saves memory & speeds up).
        """
        if self.destroyed or self.escaped:
            return

        substeps = 5
        dt_sub = dt / substeps

        for _ in range(substeps):
            self.rk4_step(dt_sub)

            # After each substep, check destruction / escape
            x, y, vx, vy = self.state

            # Quick bounding checks to avoid expensive hypot when obvious
            if abs(x) < RADIUS_EARTH * 2 and abs(y) < RADIUS_EARTH * 2:
                r = math.sqrt(x * x + y * y)
                if r <= (RADIUS_EARTH + self.object_radius):
                    print("Meteor collided with Earth. Object destroyed.")
                    self.destroyed = True
                    self.has_collided = True
                    return

            # Escape boundary check (fast absolute checks)
            if abs(x) > SIM_BOUNDARY or abs(y) > SIM_BOUNDARY:
                self.escaped = True
                self.has_collided = False
                print("Meteor escaped boundary.")
                return

        # After finishing micro-steps, append one history snapshot
        self._append_history_point()

    def get_history(self):
        # Convert to numpy arrays only when requested (keeps runtime fast)
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
        # store history efficiently
        self.hist_t = []
        self.hist_x = []
        self.hist_y = []
        self.hist_vx = []
        self.hist_vy = []

    def update(self, dt, target_pos, target_vel, current_sim_time):
        if not self.active:
            return np.inf

        # record
        self.hist_t.append(float(current_sim_time))
        self.hist_x.append(float(self.pos[0]))
        self.hist_y.append(float(self.pos[1]))
        self.hist_vx.append(float(self.vel[0]))
        self.hist_vy.append(float(self.vel[1]))

        v_norm = np.linalg.norm(target_vel)
        if v_norm > 0:
            v_hat = target_vel / v_norm
            aim_point = target_pos - (v_hat * 1200.0)
        else:
            aim_point = target_pos

        r_vec = aim_point - self.pos
        real_dist = np.linalg.norm(target_pos - self.pos)

        if self.fuel <= 0:
            self.active = False
            return real_dist

        closing_speed = np.linalg.norm(target_vel - self.vel)
        if closing_speed < 1.0:
            closing_speed = 1.0
        dist = np.linalg.norm(r_vec)
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

    def get_history_dict(self):
        return [
            {"t": t, "x": x, "y": y, "vx": vx, "vy": vy}
            for t, x, y, vx, vy in zip(self.hist_t, self.hist_x, self.hist_y, self.hist_vx, self.hist_vy)
        ]


class InterceptorRocket:
    def __init__(self):
        self.pos = np.array([RADIUS_EARTH * 0.7, RADIUS_EARTH * 0.7], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.active = False
        self.deployed_satellites = False
        self.acceleration_mag = 300.0
        self.launch_timestamp = None
        self.hist_t = []
        self.hist_x = []
        self.hist_y = []
        self.hist_vx = []
        self.hist_vy = []

    def launch(self, t):
        self.active = True
        self.launch_timestamp = float(t)

    def update(self, dt, target_pos, target_vel, current_sim_time):
        if not self.active:
            return None

        self.hist_t.append(float(current_sim_time))
        self.hist_x.append(float(self.pos[0]))
        self.hist_y.append(float(self.pos[1]))
        self.hist_vx.append(float(self.vel[0]))
        self.hist_vy.append(float(self.vel[1]))

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

    def get_history_dict(self):
        return [
            {"t": t, "x": x, "y": y, "vx": vx, "vy": vy}
            for t, x, y, vx, vy in zip(self.hist_t, self.hist_x, self.hist_y, self.hist_vx, self.hist_vy)
        ]


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
                    # APPLY IMPULSE: accumulate on sim.pending_delta_v (robust vs RK4 timing)
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
            sat = DeflectorSatellite(r_pos[0] + offsets[i][0], r_pos[1] + offsets[i][1], r_vel[0], r_vel[1], i)
            sat.deploy_timestamp = float(t)
            self.satellites.append(sat)

    def _apply_impulse(self, sim, m_pos, m_vel):
        v_norm = np.linalg.norm(m_vel)
        r_norm = np.linalg.norm(m_pos)
        if v_norm == 0 or r_norm == 0:
            return

        v_hat = m_vel / v_norm
        r_hat = m_pos / r_norm

        # Prograde + Outward mix
        impulse_dir = (v_hat * 0.7) + (r_hat * 0.3)
        impulse_dir = impulse_dir / np.linalg.norm(impulse_dir)

        delta_v = impulse_dir * self.kick_delta_v_per_sat

        # IMPORTANT: accumulate into pending_delta_v so RK4 will apply it at the start of the next micro-step
        sim.pending_delta_v += delta_v
        # Do NOT call sim._append_history_point() here — RK4 & step() manage history snapshots

    def get_rocket_pos(self):
        if self.rocket.active:
            return [self.rocket.pos[0]], [self.rocket.pos[1]]
        return [], []

    def get_satellite_positions(self):
        xs, ys = [], []
        for sat in self.satellites:
            if sat.active:
                xs.append(sat.pos[0])
                ys.append(sat.pos[1])
        return xs, ys


def Simulation2D():
    drag = DragForce(c_t0=4.0e-5, c_r0=5e-8, decay=1e-5)
    physics = PhysicsService(G_CONSTANT, MASS_EARTH, METEOR_MASS)
    solver = BesselAnomalySolver(max_iterations=40)
    sim = TrajectoryComputer(physics, solver, drag)
    sim.set_object_radius(METEOR_RADIUS)

    initial_position = [3e7 + 5e7, -3.5e7]
    # try a light object to test deflection, e.g. 1 kg; (Note: METEOR_MASS constant not used by impulse)
    initial_velocity = [-4949.74, 4949.747]
    sim.set_initial_state_vectors(initial_position, initial_velocity)

    defense = DefenseCoordinator()

    dt = 5.0

    try:
        while not sim.destroyed and not sim.escaped:
            sim.step(dt)
            # defense.update after main step (works too). If you call it during micro-steps, the pending_delta_v mechanism still protects impulses.
            defense.update(dt, sim)
    finally:
        hist = sim.get_history()
        t_arr, x_arr, y_arr, r_arr = hist["t"], hist["x"], hist["y"], hist["r"]

        total_points = len(t_arr)
        points_to_save = min(1000, total_points)
        output_list = []
        if total_points > 0:
            indices = np.round(np.linspace(0, total_points - 1, num=points_to_save)).astype(int)
            output_list = [
                {"time_sec": float(t_arr[i]), "x_m": float(x_arr[i]), "y_m": float(y_arr[i]), "r_m": float(r_arr[i])}
                for i in indices
            ]

        rocket_data = {
            "launch_timestamp": defense.rocket.launch_timestamp,
            "trajectory": defense.rocket.get_history_dict()
        }

        sat_data_list = []
        for sat in defense.satellites:
            sat_data_list.append({
                "id": sat.target_id,
                "deploy_timestamp": sat.deploy_timestamp,
                "detonation_timestamp": sat.detonation_timestamp,
                "trajectory": sat.get_history_dict()
            })

        result = {
            "metadata": {
                "meteor_mass_kg": METEOR_MASS,
                "has_collided": bool(sim.has_collided),
                "termination_reason": "Escaped" if sim.escaped else ("Impact" if sim.destroyed else "Timeout")
            },
            "data": output_list,
            "rocket": rocket_data,
            "satellites": sat_data_list
        }

        return result

if __name__ == "__main__":
    Simulation2D()