"""Quasi-steady-state lap time simulation.

Method: compute the curvature of a line round the circuit, find the maximum
cornering speed at each point from the grip available, then sweep forward
under acceleration limits and backward under braking limits. The lowest of
the three at each point is the speed the car can actually carry.

Geometry and car are described separately from the physics, so the same
simulator runs any car on any line. Track geometry currently comes from a
driven telemetry line, so the simulator answers "how fast could this car go
round the line this driver took" rather than "what is the optimal lap".

MODEL ASSUMPTIONS AND LIMITATIONS

This is a point mass with two refinements: load-sensitive tyres, and a
driven-axle limit on acceleration. Not modelled:
  - explicit weight transfer between individual wheels
  - aerodynamic balance shifting with ride height
  - a torque curve and gear ratios; power is a single figure
  - tyre temperature, wear, camber and track surface
  - elevation change, banking and kerbs

Grip sharing between cornering and acceleration uses a friction ellipse,
which assumes the same peak grip laterally and longitudinally.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

GRAVITY = 9.81
AIR_DENSITY = 1.225      # kg/m3 at sea level, 15C

@dataclass
class Car:
    """Everything the point-mass simulation needs to know about a car."""
    name: str
    mass: float              # kg, including driver and fuel
    power: float             # W, at the wheels
    cla: float               # downforce coefficient x area, m2
    cda: float               # drag coefficient x area, m2
    mu: float                # tyre friction coefficient at reference load
    brake_limit: float       # maximum braking deceleration, g
    load_sensitivity: float = 0.0    # exponent k; 0 means constant mu
    reference_load: float = 2100.0   # N per tyre at which mu applies
    drive_fraction: float = 1.0      # share of grip the driven axle can use
    drs_cda_delta: float = 0.0       # drag reduction when DRS is open

    def downforce(self, speed):
        return 0.5 * AIR_DENSITY * self.cla * speed ** 2

    def drag(self, speed, drs=False):
        cda = self.cda - (self.drs_cda_delta if drs else 0.0)
        return 0.5 * AIR_DENSITY * cda * speed ** 2

    def normal_load(self, speed):
        return self.mass * GRAVITY + self.downforce(speed)

    def effective_mu(self, speed):
        """Friction coefficient at the load this speed produces.

        Real tyres lose grip as vertical load rises. Modelled as a power
        law: mu = mu0 * (F/F_ref)^-k.
        """
        if self.load_sensitivity <= 0:
            return self.mu

        load_per_tyre = self.normal_load(speed) / 4.0
        ratio = np.maximum(load_per_tyre / self.reference_load, 1e-6)
        return self.mu * ratio ** (-self.load_sensitivity)

    def grip_force(self, speed):
        return self.effective_mu(speed) * self.normal_load(speed)

# 2024-spec F1 car. Mass and power are anchored to known figures;
# the rest are starting points for fitting.
F1_2024 = Car(
    name="F1 2024",
    mass=850.0,
    power=760_000.0,
    cla=4.5,
    cda=1.48,
    mu=1.75,
    brake_limit=6.0,
    load_sensitivity=0.15,
    reference_load=2100.0,
    drive_fraction=0.55,
    drs_cda_delta=0.30,
)

@dataclass
class Track:
    """Geometry of a line round a circuit, at even spacing."""
    name: str
    distance: np.ndarray
    curvature: np.ndarray
    source: str

    @property
    def radius(self):
        return np.divide(1.0, self.curvature,
                         out=np.full_like(self.curvature, np.inf),
                         where=self.curvature > 1e-9)

    @property
    def length(self):
        return float(self.distance[-1])

    @property
    def step(self):
        return float(self.distance[1] - self.distance[0])

@dataclass
class LapResult:
    """Output of a simulation run."""
    track: Track
    car: Car
    speed: np.ndarray
    lap_time: float
    limit: np.ndarray

    @property
    def speed_kph(self):
        return self.speed * 3.6

def track_from_telemetry(x, y, name="unknown", spacing=2.0,
                         smoothing_window=15, scale=0.1):
    """Build a Track from a driven telemetry line.

    Resamples to even spacing before smoothing, so the filter spans a
    consistent length of track everywhere.
    """
    x = np.asarray(x, dtype=float) * scale
    y = np.asarray(y, dtype=float) * scale

    step = np.hypot(np.diff(x), np.diff(y))
    raw_distance = np.concatenate([[0], np.cumsum(step)])

    keep = np.concatenate([[True], step > 1e-6])
    x, y, raw_distance = x[keep], y[keep], raw_distance[keep]

    distance = np.arange(0, raw_distance[-1], spacing)
    rx = np.interp(distance, raw_distance, x)
    ry = np.interp(distance, raw_distance, y)

    if smoothing_window > 1:
        window = np.ones(smoothing_window) / smoothing_window
        rx = np.convolve(rx, window, mode='same')
        ry = np.convolve(ry, window, mode='same')
        edge = smoothing_window
        distance = distance[edge:-edge]
        rx, ry = rx[edge:-edge], ry[edge:-edge]
        distance = distance - distance[0]

    dx = np.gradient(rx, spacing)
    dy = np.gradient(ry, spacing)
    ddx = np.gradient(dx, spacing)
    ddy = np.gradient(dy, spacing)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    curvature = np.divide(numerator, denominator,
                          out=np.zeros_like(numerator),
                          where=denominator > 1e-9)

    return Track(name=name, distance=distance, curvature=curvature,
                 source="driven telemetry line")

def cornering_limit(track, car, iterations=25):
    """Maximum speed at each point, set by lateral grip.

    With load-sensitive tyres mu depends on speed, so the balance is
    implicit. Solved by damped fixed-point iteration.
    """
    radius = track.radius
    finite = np.isfinite(radius) & (radius > 0)

    speed = np.full_like(radius, 100.0)
    speed[~finite] = np.inf

    for _ in range(iterations):
        current = np.where(finite, speed, 0.0)
        grip = car.grip_force(current)

        new_squared = np.divide(grip * radius, car.mass,
                                out=np.zeros_like(radius),
                                where=finite)
        new = np.sqrt(np.maximum(new_squared, 0.0))

        speed = np.where(finite, 0.5 * speed + 0.5 * new, np.inf)

    return speed

def available_longitudinal(car, speed, curvature, braking=False):
    """Longitudinal acceleration available, in m/s2.

    One grip budget shared between cornering and accelerating or braking,
    via a friction ellipse.
    """
    grip = car.grip_force(speed)

    lateral_force = car.mass * speed ** 2 * curvature
    used = np.clip(lateral_force / np.maximum(grip, 1e-9), 0.0, 1.0)
    remaining = np.sqrt(np.maximum(0.0, 1.0 - used ** 2))

    drag = car.drag(speed)

    if braking:
        tyre_limit = grip * remaining
        mechanical_limit = car.brake_limit * car.mass * GRAVITY
        force = np.minimum(tyre_limit, mechanical_limit) + drag
        return force / car.mass

    # Only the driven axle puts power down.
    traction_limit = grip * remaining * car.drive_fraction
    power_limit = car.power / np.maximum(speed, 1.0)
    force = np.minimum(traction_limit, power_limit) - drag
    return force / car.mass

def _terminal_speed(car):
    """Top speed where power equals drag."""
    return (car.power / (0.5 * AIR_DENSITY * car.cda)) ** (1 / 3)

def simulate(track, car, initial_speed=None):
    """Run a quasi-steady-state lap simulation."""
    step = track.step
    curvature = track.curvature
    points = len(track.distance)

    corner_speed = cornering_limit(track, car)
    corner_speed = np.minimum(corner_speed, _terminal_speed(car))

    forward = np.empty(points)
    forward[0] = corner_speed[0] if initial_speed is None else initial_speed

    for i in range(1, points):
        acceleration = available_longitudinal(
            car, forward[i - 1], curvature[i - 1], braking=False)
        squared = forward[i - 1] ** 2 + 2 * acceleration * step
        forward[i] = min(np.sqrt(max(squared, 1.0)), corner_speed[i])

    backward = np.empty(points)
    backward[-1] = corner_speed[-1]

    for i in range(points - 2, -1, -1):
        deceleration = available_longitudinal(
            car, backward[i + 1], curvature[i + 1], braking=True)
        squared = backward[i + 1] ** 2 + 2 * deceleration * step
        backward[i] = min(np.sqrt(max(squared, 1.0)), corner_speed[i])

    speed = np.minimum(np.minimum(forward, backward), corner_speed)

    limit = np.full(points, 'power', dtype=object)
    at_corner = np.isclose(speed, corner_speed, rtol=0.01)
    at_brake = np.isclose(speed, backward, rtol=0.01) & ~at_corner
    limit[at_corner] = 'corner'
    limit[at_brake] = 'brake'

    lap_time = float(np.sum(step / np.maximum(speed, 1.0)))

    return LapResult(track=track, car=car, speed=speed,
                     lap_time=lap_time, limit=limit)

def fit_car(track, actual_speed_kph, actual_lap_time, base_car,
            fit=('cda', 'cla', 'mu', 'load_sensitivity', 'drive_fraction'),
            bounds=None, verbose=True):
    """Fit car parameters so the simulation matches a real lap.

    Fits against the whole speed trace rather than just lap time. A single
    lap time can be produced by many different parameter sets; thousands of
    speed samples constrain the fit far more tightly.
    """
    default_bounds = {
        'cda': (0.8, 2.2),
        'cla': (2.5, 7.0),
        'mu': (1.0, 2.5),
        'load_sensitivity': (0.0, 0.4),
        'drive_fraction': (0.4, 1.0),
        'brake_limit': (3.0, 8.0),
        'mass': (700.0, 950.0),
        'power': (500_000.0, 900_000.0),
    }
    bounds = bounds or default_bounds

    start = [getattr(base_car, name) for name in fit]
    limits = [bounds[name] for name in fit]

    actual_ms = np.asarray(actual_speed_kph, dtype=float) / 3.6

    def build(values):
        return replace(base_car, **dict(zip(fit, values)))

    def error(values):
        car = build(values)
        try:
            result = simulate(track, car)
        except Exception:
            return 1e6

        speed_error = np.sqrt(np.mean(
            ((result.speed - actual_ms) / np.maximum(actual_ms, 1.0)) ** 2))
        time_error = abs(result.lap_time - actual_lap_time) / actual_lap_time

        return speed_error + 2.0 * time_error

    outcome = minimize(error, start, bounds=limits, method='L-BFGS-B',
                       options={'maxiter': 200})

    fitted = build(outcome.x)

    if verbose:
        print(f"\nFitted parameters ({outcome.nit} iterations):")
        for name, value in zip(fit, outcome.x):
            print(f"  {name}: {getattr(base_car, name):.3f} -> {value:.3f}")
        print(f"  final error: {outcome.fun:.4f}")

    return fitted, outcome

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import style
    import telemetry

    style.apply()

    OFFICIAL_LENGTH = 5793.0

    lap, tel, session = telemetry.get_lap_telemetry(2024, 'Monza', 'NOR')
    track = track_from_telemetry(tel['X'], tel['Y'], name="Monza")

    print(f"Lap length: {track.length:.0f}m at {track.step:.1f}m spacing")
    print(f"Tightest radius: {track.radius.min():.0f}m")

    actual = lap['LapTime'].total_seconds()
    real_speed_kph = np.interp(track.distance, tel['Distance'], tel['Speed'])

    result = simulate(track, F1_2024)
    print(f"\nBefore fitting: {result.lap_time:.3f}s "
          f"({result.lap_time - actual:+.3f}s), "
          f"top speed {result.speed_kph.max():.0f} km/h")

    fitted, outcome = fit_car(track, real_speed_kph, actual, F1_2024)
    fitted_result = simulate(track, fitted)

    print(f"\nAfter fitting: {fitted_result.lap_time:.3f}s "
          f"({fitted_result.lap_time - actual:+.3f}s), "
          f"top speed {fitted_result.speed_kph.max():.0f} km/h "
          f"vs actual {tel['Speed'].max():.0f} km/h")

    real_ms = real_speed_kph / 3.6
    real_accel = np.gradient(real_ms) * real_ms / track.step
    sim_accel = np.gradient(fitted_result.speed) * fitted_result.speed / track.step

    print(f"\nAcceleration after fitting:")
    print(f"  accelerating: sim {np.nanmax(sim_accel) / GRAVITY:.1f}g "
          f"vs real {np.nanmax(real_accel) / GRAVITY:.1f}g")
    print(f"  braking:      sim {np.nanmin(sim_accel) / GRAVITY:.1f}g "
          f"vs real {np.nanmin(real_accel) / GRAVITY:.1f}g")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(tel['Distance'], tel['Speed'], color=style.DRIVER_A,
            label='Actual', linewidth=1.5)
    ax.plot(track.distance, fitted_result.speed_kph, color=style.DRIVER_B,
            label='Simulated (fitted)', linewidth=1.5)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.legend()

    style.title(fig, f"Lap simulation — {track.name}",
                f"Fitted {fitted_result.lap_time:.3f}s vs actual "
                f"{actual:.3f}s  ·  {fitted_result.lap_time - actual:+.3f}s")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()