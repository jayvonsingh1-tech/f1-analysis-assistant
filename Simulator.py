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

Grip sharing uses a friction ellipse, which assumes equal peak grip
laterally and longitudinally. Validation shows this is the model's main
weakness: fits that match lap time tend to underpredict acceleration and
overpredict braking.

Fitting a single circuit is underdetermined - many parameter sets produce
the same lap time. fit_multi() fits several circuits at once, sharing the
tyre parameters while letting aero vary per circuit, which is both more
physically honest (teams change wing level) and better constrained.
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
    max_tractive_force: float = 1e9   # N, torque limit at low speed

    def downforce(self, speed):
        return 0.5 * AIR_DENSITY * self.cla * speed ** 2

    def drag(self, speed, drs=False):
        cda = self.cda - (self.drs_cda_delta if drs else 0.0)
        return 0.5 * AIR_DENSITY * cda * speed ** 2

    def normal_load(self, speed):
        return self.mass * GRAVITY + self.downforce(speed)

    def effective_mu(self, speed):
        """Friction coefficient at the load this speed produces.

        Real tyres lose grip as vertical load rises. Power law:
        mu = mu0 * (F/F_ref)^-k.
        """
        if self.load_sensitivity <= 0:
            return self.mu

        load_per_tyre = self.normal_load(speed) / 4.0
        ratio = np.maximum(load_per_tyre / self.reference_load, 1e-6)
        return self.mu * ratio ** (-self.load_sensitivity)

    def grip_force(self, speed):
        return self.effective_mu(speed) * self.normal_load(speed)

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
    max_tractive_force=18_000.0,
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

def track_from_telemetry(x, y, name="unknown", spacing=1.0,
                         smoothing_metres=20.0, scale=0.1):
    """Build a Track from a driven telemetry line.

    Resamples to even spacing before smoothing, so the filter spans a
    consistent length of track everywhere.

    spacing: metres between simulation points. Smaller is more accurate
        and slower.
    smoothing_metres: how much track the smoothing filter averages over.
        Must be short enough not to blur the tightest corners: Monaco
        needs less than Monza.
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

    window_points = max(3, int(round(smoothing_metres / spacing)))
    if window_points > 1:
        window = np.ones(window_points) / window_points
        rx = np.convolve(rx, window, mode='same')
        ry = np.convolve(ry, window, mode='same')
        edge = window_points
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
                 source=f"driven line, {spacing}m steps, "
                        f"{smoothing_metres}m smoothing")

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
    """Longitudinal acceleration available, in m/s2."""
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
    # Power gives F = P/v, which is unbounded as v approaches zero. Real
    # cars are torque-limited at low speed, so cap the tractive force.
    traction_limit = grip * remaining * car.drive_fraction
    power_limit = np.minimum(car.power / np.maximum(speed, 1.0),
                             car.max_tractive_force)
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

@dataclass
class Reference:
    """One real lap to fit against."""
    track: Track
    speed_kph: np.ndarray    # actual speed at each of track.distance
    lap_time: float          # actual, seconds

def lap_error(track, car, reference):
    """Dimensionless error between a simulated and a real lap.

    Weighted towards the speed trace rather than lap time. A single lap
    time can be hit by many wrong parameter sets; matching the trace at
    every point is far harder to fake. Top speed gets its own term
    because it cleanly isolates the drag and power balance.
    """
    try:
        result = simulate(track, car)
    except Exception:
        return 1e6, None

    actual_ms = reference.speed_kph / 3.6

    speed_error = np.sqrt(np.mean(
        ((result.speed - actual_ms) / np.maximum(actual_ms, 1.0)) ** 2))

    top_error = abs(result.speed.max() - actual_ms.max()) / actual_ms.max()

    time_error = abs(result.lap_time - reference.lap_time) / reference.lap_time

    return speed_error + 1.0 * top_error + 0.5 * time_error, result

SHARED_BOUNDS = {
    'mu': (1.0, 2.5),
    'load_sensitivity': (0.0, 0.4),
    'drive_fraction': (0.3, 1.0),
    'brake_limit': (3.0, 9.0),
}
AERO_BOUNDS = {
    'cla': (2.5, 7.5),
    'cda': (0.8, 2.5),
}

def fit_multi(references, base_car,
              shared=('mu', 'load_sensitivity', 'drive_fraction',
                      'brake_limit'),
              per_circuit=('cla', 'cda'),
              verbose=True, maxiter=300):
    """Fit one car across several circuits at once.

    Tyre and drivetrain parameters are shared, because they don't change
    between races. Aero is fitted per circuit, because teams genuinely run
    different wing levels - a Monza car has far less downforce than a
    Monaco one, so forcing one ClA across both would be wrong.

    Fitting several circuits simultaneously constrains the shared
    parameters far better than any single lap can, because each circuit
    stresses a different part of the model.

    references: list of Reference
    Returns (shared_values, per_circuit_cars, outcome)
    """
    n = len(references)

    start = ([getattr(base_car, name) for name in shared]
             + [getattr(base_car, name) for _ in range(n)
                for name in per_circuit])

    limits = ([SHARED_BOUNDS[name] for name in shared]
              + [AERO_BOUNDS[name] for _ in range(n)
                 for name in per_circuit])

    def unpack(values):
        shared_values = dict(zip(shared, values[:len(shared)]))
        cars = []
        cursor = len(shared)
        for _ in range(n):
            aero = dict(zip(per_circuit,
                            values[cursor:cursor + len(per_circuit)]))
            cursor += len(per_circuit)
            cars.append(replace(base_car, **shared_values, **aero))
        return shared_values, cars

    def total_error(values):
        _, cars = unpack(values)
        total = 0.0
        for reference, car in zip(references, cars):
            error, _ = lap_error(reference.track, car, reference)
            total += error
        return total / n

    outcome = minimize(total_error, start, bounds=limits,
                       method='L-BFGS-B', options={'maxiter': maxiter})

    shared_values, cars = unpack(outcome.x)

    if verbose:
        print(f"\nMulti-circuit fit ({outcome.nit} iterations, "
              f"mean error {outcome.fun:.4f})")
        print("\n  Shared (tyres and drivetrain):")
        for name, value in shared_values.items():
            print(f"    {name}: {getattr(base_car, name):.3f} "
                  f"-> {value:.3f}")
        print("\n  Per circuit (aero):")
        for reference, car in zip(references, cars):
            print(f"    {reference.track.name}: "
                  + ", ".join(f"{name} {getattr(car, name):.3f}"
                              for name in per_circuit))

    return shared_values, cars, outcome

def build_reference(year, race, driver, spacing=1.0, smoothing_metres=20.0):
    """Load a real lap and prepare it for fitting."""
    import telemetry

    lap, tel, session = telemetry.get_lap_telemetry(year, race, driver)
    track = track_from_telemetry(tel['X'], tel['Y'], name=race,
                                 spacing=spacing,
                                 smoothing_metres=smoothing_metres)
    speed = np.interp(track.distance, tel['Distance'], tel['Speed'])

    return Reference(track=track, speed_kph=speed,
                     lap_time=lap['LapTime'].total_seconds())

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import style

    style.apply()

    # Circuits chosen to stress different parts of the model:
    # Monza low downforce, Monaco low speed, Silverstone high speed,
    # Barcelona mixed. Monaco gets tighter smoothing because its corners
    # are short enough that a 20m filter would blur them.
    setups = [
        (2024, 'Monza', 'NOR', 20.0),
        (2024, 'Monaco', 'LEC', 10.0),
        (2024, 'Silverstone', 'HAM', 20.0),
        (2024, 'Barcelona', 'VER', 15.0),
    ]

    references = []
    for year, race, driver, smoothing in setups:
        print(f"Loading {race} {year} ({driver})...")
        reference = build_reference(year, race, driver,
                                    smoothing_metres=smoothing)
        print(f"  {reference.track.length:.0f}m, "
              f"tightest radius {reference.track.radius.min():.0f}m, "
              f"lap {reference.lap_time:.3f}s")
        references.append(reference)

    print("\nFitting...")
    shared_values, cars, outcome = fit_multi(references, F1_2024)

    print("\nPer-circuit results:")
    for reference, car in zip(references, cars):
        result = simulate(reference.track, car)
        error = result.lap_time - reference.lap_time
        print(f"  {reference.track.name}: {result.lap_time:.3f}s "
              f"vs {reference.lap_time:.3f}s ({error:+.3f}s), "
              f"top {result.speed_kph.max():.0f} "
              f"vs {reference.speed_kph.max():.0f} km/h")

    fig, axes = plt.subplots(len(references), 1,
                             figsize=(13, 3 * len(references)))
    for ax, reference, car in zip(axes, references, cars):
        result = simulate(reference.track, car)
        ax.plot(reference.track.distance, reference.speed_kph,
                color=style.DRIVER_A, label='Actual', linewidth=1.2)
        ax.plot(reference.track.distance, result.speed_kph,
                color=style.DRIVER_B, label='Simulated', linewidth=1.2)
        ax.set_ylabel('km/h')
        ax.set_title(f"{reference.track.name}  "
                     f"{result.lap_time:.3f}s vs {reference.lap_time:.3f}s",
                     color=style.TEXT, fontsize=11)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Distance (m)')

    style.title(fig, "Multi-circuit fit",
                "Shared tyre parameters, per-circuit aero")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()