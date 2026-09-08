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
  - aerodynamic balance shifting with ride height, and its effect on
    understeer and oversteer
  - a torque curve and gear ratios; power is a single figure
  - tyre temperature, wear, camber and track surface
  - elevation change, banking and kerbs

Grip sharing between cornering and acceleration uses a friction ellipse,
which assumes the same peak grip laterally and longitudinally.
"""

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
AIR_DENSITY = 1.225      # kg/m3 at sea level, 15C

@dataclass
class Car:
    """Everything the point-mass simulation needs to know about a car.

    Aero coefficients are given as coefficient x area (ClA, CdA), which is
    how they are usually quoted, so no separate frontal area is needed.
    """
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
        """Aerodynamic downforce in newtons at a given speed (m/s)."""
        return 0.5 * AIR_DENSITY * self.cla * speed ** 2

    def drag(self, speed, drs=False):
        """Aerodynamic drag in newtons at a given speed (m/s)."""
        cda = self.cda - (self.drs_cda_delta if drs else 0.0)
        return 0.5 * AIR_DENSITY * cda * speed ** 2

    def normal_load(self, speed):
        """Total vertical load: weight plus downforce."""
        return self.mass * GRAVITY + self.downforce(speed)

    def effective_mu(self, speed):
        """Friction coefficient at the load this speed produces.

        Real tyres lose grip as vertical load rises. Modelled as a power
        law: mu = mu0 * (F/F_ref)^-k, with load per tyre taken as the
        total normal load divided by four.
        """
        if self.load_sensitivity <= 0:
            return self.mu

        load_per_tyre = self.normal_load(speed) / 4.0
        ratio = np.maximum(load_per_tyre / self.reference_load, 1e-6)
        return self.mu * ratio ** (-self.load_sensitivity)

    def grip_force(self, speed):
        """Maximum horizontal force all four tyres can produce."""
        return self.effective_mu(speed) * self.normal_load(speed)

# 2024-spec F1 car. Mass, power and top speed are anchored to known
# figures; mu, load sensitivity and drive fraction are fitted.
F1_2024 = Car(
    name="F1 2024",
    mass=850.0,          # 798kg minimum plus part-race fuel
    power=760_000.0,     # ~1020hp at the wheels
    cla=4.5,
    cda=1.48,            # fitted to match 339 km/h top speed
    mu=1.75,
    brake_limit=6.0,
    load_sensitivity=0.15,
    reference_load=2100.0,   # roughly the static load per tyre
    drive_fraction=0.55,     # rear axle share under acceleration
    drs_cda_delta=0.30,
)

@dataclass
class Track:
    """Geometry of a line round a circuit, at even spacing.

    Deliberately independent of where the geometry came from: a driven
    telemetry line now, a computed racing line later.
    """
    name: str
    distance: np.ndarray     # metres from the start, evenly spaced
    curvature: np.ndarray    # 1/m at each point
    source: str              # how this geometry was derived

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
    speed: np.ndarray        # m/s at each track point
    lap_time: float          # seconds
    limit: np.ndarray        # 'corner', 'power' or 'brake' at each point

    @property
    def speed_kph(self):
        return self.speed * 3.6

def track_from_telemetry(x, y, name="unknown", spacing=2.0,
                         smoothing_window=15, scale=0.1):
    """Build a Track from a driven telemetry line.

    Resamples to even spacing before smoothing. Telemetry points are dense
    in slow corners and sparse on straights, so smoothing raw samples
    treats the two very differently; resampling first makes the filter
    uniform along the track.
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

    With load-sensitive tyres, mu depends on downforce which depends on
    speed, so the balance is implicit. Solved by damped fixed-point
    iteration: guess a speed, compute the grip at that speed, find the
    speed that grip supports, repeat until it settles.
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

    The tyres have one grip budget shared between cornering and
    accelerating or braking. Using a friction ellipse: if a fraction f of
    the grip is committed laterally, sqrt(1 - f^2) remains for
    longitudinal use.
    """
    grip = car.grip_force(speed)

    lateral_force = car.mass * speed ** 2 * curvature
    used = np.clip(lateral_force / np.maximum(grip, 1e-9), 0.0, 1.0)
    remaining = np.sqrt(np.maximum(0.0, 1.0 - used ** 2))

    drag = car.drag(speed)

    if braking:
        # All four tyres brake, and drag helps.
        tyre_limit = grip * remaining
        mechanical_limit = car.brake_limit * car.mass * GRAVITY
        force = np.minimum(tyre_limit, mechanical_limit) + drag
        return force / car.mass

    # Only the driven axle puts power down. Weight transfer under
    # acceleration loads the rear, so it carries rather more than half.
    traction_limit = grip * remaining * car.drive_fraction
    power_limit = car.power / np.maximum(speed, 1.0)
    force = np.minimum(traction_limit, power_limit) - drag
    return force / car.mass

def _terminal_speed(car):
    """Top speed where power equals drag: P = 0.5 * rho * CdA * v^3."""
    return (car.power / (0.5 * AIR_DENSITY * car.cda)) ** (1 / 3)

def simulate(track, car, initial_speed=None):
    """Run a quasi-steady-state lap simulation.

    Three passes:
      1. Cornering limit at every point from lateral grip
      2. Forward from the start under acceleration limits
      3. Backward from the end under braking limits

    The speed the car can carry is the lowest of the three.
    """
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

    result = simulate(track, F1_2024)

    actual = lap['LapTime'].total_seconds()
    error = result.lap_time - actual

    print(f"\nSimulated: {result.lap_time:.3f}s")
    print(f"Actual:    {actual:.3f}s")
    print(f"Error:     {error:+.3f}s ({error / actual * 100:+.1f}%)")
    print(f"Top speed: {result.speed_kph.max():.0f} km/h "
          f"vs actual {tel['Speed'].max():.0f} km/h")

    sim_pace = result.lap_time / track.length
    real_pace = actual / OFFICIAL_LENGTH
    print(f"\nScaled to official length: "
          f"{sim_pace * OFFICIAL_LENGTH:.3f}s "
          f"({sim_pace * OFFICIAL_LENGTH - actual:+.3f}s)")

    real_speed = np.interp(track.distance, tel['Distance'],
                           tel['Speed']) / 3.6
    real_accel = np.gradient(real_speed) * real_speed / track.step
    sim_accel = np.gradient(result.speed) * result.speed / track.step

    print(f"\nAcceleration (m/s2):")
    print(f"  Real peak accelerating:  {np.nanmax(real_accel):.1f} "
          f"({np.nanmax(real_accel) / GRAVITY:.1f}g)")
    print(f"  Sim  peak accelerating:  {np.nanmax(sim_accel):.1f} "
          f"({np.nanmax(sim_accel) / GRAVITY:.1f}g)")
    print(f"  Real peak braking:      {np.nanmin(real_accel):.1f} "
          f"({np.nanmin(real_accel) / GRAVITY:.1f}g)")
    print(f"  Sim  peak braking:      {np.nanmin(sim_accel):.1f} "
          f"({np.nanmin(sim_accel) / GRAVITY:.1f}g)")

    print(f"\nEffective mu:")
    for v in (30, 60, 90):
        print(f"  at {v * 3.6:.0f} km/h: {F1_2024.effective_mu(v):.3f}")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(tel['Distance'], tel['Speed'], color=style.DRIVER_A,
            label='Actual', linewidth=1.5)
    ax.plot(track.distance, result.speed_kph, color=style.DRIVER_B,
            label='Simulated', linewidth=1.5)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.legend()

    style.title(fig, f"Lap simulation — {track.name}",
                f"Simulated {result.lap_time:.3f}s vs actual "
                f"{actual:.3f}s  ·  {error:+.3f}s")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()