"""Quasi-steady-state lap time simulation.

Method: compute the curvature of a line round the circuit, find the maximum
cornering speed at each point from the grip available, then sweep forward
under acceleration limits and backward under braking limits. The lowest of
the three at each point is the speed the car can actually carry.

Geometry and car are described separately from the physics, so the same
simulator runs any car on any line. Track geometry currently comes from
driven laps, so the simulator answers "how fast could this car go round the
line drivers took" rather than "what is the optimal lap".

MODEL ASSUMPTIONS AND LIMITATIONS

This is a point mass with three refinements: load-sensitive tyres, a
driven-axle limit on acceleration, and a tractive force cap at low speed.
Not modelled:
  - explicit weight transfer between individual wheels
  - aerodynamic balance shifting with ride height
  - a torque curve and gear ratios; power is a single figure
  - tyre temperature, wear, camber and track surface
  - elevation change, banking and kerbs

Grip sharing uses a friction ellipse, which assumes equal peak grip
laterally and longitudinally.

GEOMETRY

Geometry quality dominates everything else. FastF1 position data arrives
about four times a second, which is too sparse within a single lap to
compute curvature reliably. track_from_laps() pools many laps, which adds
genuine spatial resolution, then fits a penalised smoothing spline: it
penalises bending, so it curves only where the data demands. Fixed-knot
splines were tried and rejected - they overfit the lap-to-lap differences
in line and produced curvature noise everywhere.

Always check implied lateral acceleration (v^2 * curvature) before
fitting. Anything well above 6g is noise, and the fit will bend the car
parameters to compensate for it.

FITTING

Fitting a single circuit is underdetermined - many parameter sets produce
the same lap time. fit_multi() fits several circuits at once, sharing tyre
parameters while letting aero vary per circuit.

SPEED

A fit runs thousands of laps, so simulate() has to be quick. Its two
sweeps step through the lap one point at a time, and numpy is fast on
whole arrays but slow when called on a single number. So the sweeps and
available_longitudinal() work on plain Python floats, and the Car methods
use only plain arithmetic, which works on one float or a whole array
alike. Keep numpy calls out of that path. cornering_limit() handles the
whole lap at once, so it uses numpy arrays.
"""

import math
from dataclasses import dataclass, field, fields, replace

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

GRAVITY = 9.81
AIR_DENSITY = 1.225      # kg/m3 at sea level, 15C

# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

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
    load_sensitivity: float = 0.0     # exponent k; 0 means constant mu
    reference_load: float = 2100.0    # N per tyre at which mu applies
    drive_fraction: float = 1.0       # share of grip the driven axle can use
    max_tractive_force: float = 1e9   # N, torque limit at low speed
    drs_cda_delta: float = 0.0        # drag reduction when DRS is open

    def __post_init__(self):
        # Store plain Python numbers. The optimiser hands over numpy ones,
        # which give the same answers but make simulate() far slower.
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.generic):
                setattr(self, item.name, value.item())

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

        # Load is weight plus downforce, so this ratio is always positive
        load_per_tyre = self.normal_load(speed) / 4.0
        ratio = load_per_tyre / self.reference_load
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
    max_tractive_force=18_000.0,
    drs_cda_delta=0.30,
)

# ---------------------------------------------------------------------------
# Track geometry
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Geometry of a line round a circuit, at even spacing.

    channels holds any other data resampled onto the same distance grid,
    such as the real speed along a driven line. Computed geometry has
    none.
    """
    name: str
    distance: np.ndarray
    curvature: np.ndarray
    source: str
    channels: dict = field(default_factory=dict)

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

def track_from_telemetry(x, y, time, name="unknown", spacing=1.0,
                         smoothing_seconds=1.0, sample_rate=10.0,
                         scale=0.1, channels=None):
    """Build a Track from a single driven lap.

    Kept for quick single-lap use, but one lap's position data is too
    sparse for reliable curvature. Prefer track_from_laps() for anything
    that matters.
    """
    x = np.asarray(x, dtype=float) * scale
    y = np.asarray(y, dtype=float) * scale
    t = np.asarray(time, dtype=float)
    channels = {key: np.asarray(value, dtype=float)
                for key, value in (channels or {}).items()}

    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    channels = {key: value[order] for key, value in channels.items()}

    unique = np.concatenate([[True], np.diff(t) > 1e-6])
    t, x, y = t[unique], x[unique], y[unique]
    channels = {key: value[unique] for key, value in channels.items()}

    dt = 1.0 / sample_rate
    grid_t = np.arange(t[0], t[-1], dt)
    gx = np.interp(grid_t, t, x)
    gy = np.interp(grid_t, t, y)
    grid_channels = {key: np.interp(grid_t, t, value)
                     for key, value in channels.items()}

    window = max(int(round(smoothing_seconds / dt)) | 1, 5)

    def smooth(values, order):
        return savgol_filter(values, window, polyorder=3, deriv=order,
                             delta=dt, mode='interp')

    sx, sy = smooth(gx, 0), smooth(gy, 0)
    vx, vy = smooth(gx, 1), smooth(gy, 1)
    accx, accy = smooth(gx, 2), smooth(gy, 2)

    numerator = np.abs(vx * accy - vy * accx)
    denominator = (vx ** 2 + vy ** 2) ** 1.5
    curvature_t = np.divide(numerator, denominator,
                            out=np.zeros_like(numerator),
                            where=denominator > 1e-9)

    along = np.concatenate(
        [[0], np.cumsum(np.hypot(np.diff(sx), np.diff(sy)))])
    distance = np.arange(0, along[-1], spacing)
    curvature = np.interp(distance, along, curvature_t)
    resampled = {key: np.interp(distance, along, value)
                 for key, value in grid_channels.items()}

    return Track(name=name, distance=distance, curvature=curvature,
                 source=f"single lap, {smoothing_seconds}s smoothing",
                 channels=resampled)

def _resample_line(x, y, spacing=1.0, smooth_metres=15.0):
    """Evenly spaced, lightly smoothed copy of a line.

    Used only to align laps against each other, so it needs to be in the
    right place, not to have accurate curvature.
    """
    moved = np.hypot(np.diff(x), np.diff(y)) > 1e-6
    keep = np.concatenate([[True], moved])
    x, y = x[keep], y[keep]

    raw = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s = np.arange(0, raw[-1], spacing)
    rx = np.interp(s, raw, x)
    ry = np.interp(s, raw, y)

    window = max(int(round(smooth_metres / spacing)) | 1, 5)
    rx = savgol_filter(rx, window, polyorder=2, mode='interp')
    ry = savgol_filter(ry, window, polyorder=2, mode='interp')
    return s, rx, ry

def track_from_laps(positions, name="unknown", spacing=1.0, lam=None,
                    scale=0.1, max_offset=8.0, bin_width=0.5,
                    speed_line=None):
    """Build a Track by pooling position data from many laps.

    One lap's position data has about four samples a second - too few per
    corner to compute a second derivative reliably. Pooling laps adds
    genuine resolution: each lap is sampled at different points round the
    track, so twenty laps give roughly twenty times the samples, and
    random position noise averages out.

    Method:
      1. Use the first lap as a reference line, and give every sample
         from every lap a continuous distance along it
      2. Average the samples into short bins along the track, each bin
         weighted by how many samples it holds
      3. Fit a penalised smoothing spline through the bin averages, x and
         y each as a function of distance
      4. Differentiate the spline analytically for curvature

    Binning matters for two reasons. The smoothing spline puts a knot at
    every data point, so samples from different laps landing millimetres
    apart make it numerically singular. And a bin average of n samples
    has 1/n the variance of one sample, so weighting by n is the correct
    way to combine them.

    The smoothing spline penalises bending, so it curves only where the
    data demands. Straights stay straight; tight corners are still
    followed because the car is slow there and the samples are dense.
    Fixed-knot splines were rejected because they overfit the small
    lap-to-lap differences in line, producing curvature noise everywhere.

    The cost: a uniform penalty slightly rounds off the fastest, most
    sparsely sampled direction changes, so the sim may be a little quick
    through those.

    positions: list of (x, y) raw position arrays, one per lap. The first
        is the alignment reference, so use the fastest lap.
    lam: smoothing strength. None chooses it automatically by generalised
        cross-validation. Raise it to smooth harder.
    max_offset: metres. Samples further than this from the reference line
        are dropped, which removes off-track moments and data glitches.
    bin_width: metres of track per averaging bin.
    speed_line: optional (x, y, speed_kph) from one lap, projected onto
        the fitted line so its speed is aligned with the geometry.

    The result is an average line across laps, slightly smoother than any
    single lap a driver actually drove.
    """
    # 1. Reference line from the first lap, plus its unit tangent
    ref_s, ref_x, ref_y = _resample_line(
        np.asarray(positions[0][0], dtype=float) * scale,
        np.asarray(positions[0][1], dtype=float) * scale)
    ref_tree = cKDTree(np.column_stack([ref_x, ref_y]))

    tangent_x = np.gradient(ref_x)
    tangent_y = np.gradient(ref_y)
    norm = np.maximum(np.hypot(tangent_x, tangent_y), 1e-9)
    tangent_x, tangent_y = tangent_x / norm, tangent_y / norm

    pooled_s, pooled_x, pooled_y = [], [], []
    for lap_x, lap_y in positions:
        lap_x = np.asarray(lap_x, dtype=float) * scale
        lap_y = np.asarray(lap_y, dtype=float) * scale
        offset, index = ref_tree.query(np.column_stack([lap_x, lap_y]))

        near = offset < max_offset
        idx = index[near]
        px, py = lap_x[near], lap_y[near]

        # Continuous distance: the nearest reference point plus the
        # along-track part of the gap to it
        along = ((px - ref_x[idx]) * tangent_x[idx]
                 + (py - ref_y[idx]) * tangent_y[idx])

        pooled_s.append(ref_s[idx] + along)
        pooled_x.append(px)
        pooled_y.append(py)

    s = np.concatenate(pooled_s)
    x = np.concatenate(pooled_x)
    y = np.concatenate(pooled_y)

    # 2. Average into bins along the track, weighted by sample count.
    # Bin centres are exactly bin_width apart, which keeps the spline
    # well conditioned.
    start = s.min()
    bins = np.floor((s - start) / bin_width).astype(int)
    counts = np.bincount(bins)
    occupied = counts > 0

    bin_s = start + (np.flatnonzero(occupied) + 0.5) * bin_width
    bin_x = np.bincount(bins, weights=x)[occupied] / counts[occupied]
    bin_y = np.bincount(bins, weights=y)[occupied] / counts[occupied]
    weights = counts[occupied].astype(float)

    # 3. Penalised smoothing spline through the bin averages
    spline_x = make_smoothing_spline(bin_s, bin_x, w=weights, lam=lam)
    spline_y = make_smoothing_spline(bin_s, bin_y, w=weights, lam=lam)

    # 4. Analytic derivatives on a fine grid
    fine = np.arange(bin_s[0], bin_s[-1], spacing / 4)
    fx, fy = spline_x(fine), spline_y(fine)
    dx, dy = spline_x.derivative(1)(fine), spline_y.derivative(1)(fine)
    ddx, ddy = spline_x.derivative(2)(fine), spline_y.derivative(2)(fine)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    curvature_fine = np.divide(numerator, denominator,
                               out=np.zeros_like(numerator),
                               where=denominator > 1e-9)

    # Even spacing in true arc length along the fitted line
    arc = np.concatenate(
        [[0], np.cumsum(np.hypot(np.diff(fx), np.diff(fy)))])
    distance = np.arange(0, arc[-1], spacing)
    curvature = np.interp(distance, arc, curvature_fine)

    channels = {}
    if speed_line is not None:
        line_x = np.asarray(speed_line[0], dtype=float) * scale
        line_y = np.asarray(speed_line[1], dtype=float) * scale
        line_v = np.asarray(speed_line[2], dtype=float)
        fitted_tree = cKDTree(np.column_stack([fx, fy]))
        _, index = fitted_tree.query(np.column_stack([line_x, line_y]))
        along = arc[index]
        order = np.argsort(along, kind='stable')
        channels['speed_kph'] = np.interp(distance, along[order],
                                          line_v[order])

    smoothing = "automatic" if lam is None else f"lam {lam}"
    return Track(name=name, distance=distance, curvature=curvature,
                 source=f"pooled from {len(positions)} laps, "
                        f"smoothing spline ({smoothing})",
                 channels=channels)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

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
    """Longitudinal acceleration available at one point, in m/s2.

    One grip budget shared between cornering and accelerating or braking,
    via a friction ellipse.

    speed and curvature are single numbers, not arrays. simulate() calls
    this for every point of the lap in both directions, so it uses
    Python's own min, max and math.sqrt, which are far quicker than the
    numpy versions on one number.
    """
    grip = car.grip_force(speed)

    lateral_force = car.mass * speed ** 2 * curvature
    used = min(max(lateral_force / max(grip, 1e-9), 0.0), 1.0)
    remaining = math.sqrt(max(1.0 - used ** 2, 0.0))

    drag = car.drag(speed)

    if braking:
        # All four tyres brake, and drag helps
        tyre_limit = grip * remaining
        mechanical_limit = car.brake_limit * car.mass * GRAVITY
        force = min(tyre_limit, mechanical_limit) + drag
        return force / car.mass

    # Only the driven axle puts power down. P/v is unbounded as speed
    # approaches zero, but real cars are torque-limited there, so cap it.
    traction_limit = grip * remaining * car.drive_fraction
    power_limit = min(car.power / max(speed, 1.0), car.max_tractive_force)
    force = min(traction_limit, power_limit) - drag
    return force / car.mass

def _terminal_speed(car):
    """Top speed where power equals drag."""
    return (car.power / (0.5 * AIR_DENSITY * car.cda)) ** (1 / 3)

def simulate(track, car, initial_speed=None, periodic=False):
    """Run a quasi-steady-state lap simulation.

    The starting speed matters. Without it the car begins every lap at
    its theoretical top speed instead of the speed it actually crossed
    the line at. The start speed comes from, in order:
      1. initial_speed, if given
      2. the real speed at the line, if the track carries a measured
         speed channel (geometry built from driven laps does)
      3. periodic=True: run the lap twice, starting the second pass at
         the first pass's finishing speed - correct for a flying lap
         with no measured data, at twice the cost
      4. otherwise the cornering limit at the start
    """
    step = track.step
    points = len(track.distance)

    corner_speed = cornering_limit(track, car)
    corner_speed = np.minimum(corner_speed, _terminal_speed(car))

    if initial_speed is None and 'speed_kph' in track.channels:
        initial_speed = float(track.channels['speed_kph'][0]) / 3.6

    # The sweeps below step through the lap one point at a time, so they
    # read from plain Python lists rather than numpy arrays (see SPEED in
    # the notes at the top of this file).
    curvature = track.curvature.tolist()
    corner_cap = corner_speed.tolist()

    def one_pass(start):
        forward = [0.0] * points
        forward[0] = min(float(start), corner_cap[0])
        for i in range(1, points):
            previous = forward[i - 1]
            acceleration = available_longitudinal(
                car, previous, curvature[i - 1], braking=False)
            squared = previous ** 2 + 2 * acceleration * step
            forward[i] = min(math.sqrt(max(squared, 1.0)), corner_cap[i])

        backward = [0.0] * points
        backward[-1] = corner_cap[-1]
        for i in range(points - 2, -1, -1):
            ahead = backward[i + 1]
            deceleration = available_longitudinal(
                car, ahead, curvature[i + 1], braking=True)
            squared = ahead ** 2 + 2 * deceleration * step
            backward[i] = min(math.sqrt(max(squared, 1.0)), corner_cap[i])

        return np.array(forward), np.array(backward)

    start = corner_speed[0] if initial_speed is None else initial_speed
    forward, backward = one_pass(start)

    if periodic and initial_speed is None:
        lap_end = min(forward[-1], backward[-1])
        forward, backward = one_pass(lap_end)

    speed = np.minimum(np.minimum(forward, backward), corner_speed)

    limit = np.full(points, 'power', dtype=object)
    at_corner = np.isclose(speed, corner_speed, rtol=0.01)
    at_brake = np.isclose(speed, backward, rtol=0.01) & ~at_corner
    limit[at_corner] = 'corner'
    limit[at_brake] = 'brake'

    lap_time = float(np.sum(step / np.maximum(speed, 1.0)))

    return LapResult(track=track, car=car, speed=speed,
                     lap_time=lap_time, limit=limit)

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """One real lap to fit against."""
    track: Track
    speed_kph: np.ndarray    # real speed at each of track.distance
    lap_time: float          # actual, seconds
    drag_limited: bool = True    # does the car reach terminal speed here?

def lap_error(track, car, reference):
    """Dimensionless error between a simulated and a real lap.

    Weighted towards the speed trace, which is far harder to fake than a
    single lap time. Top speed gets its own term only where the car
    actually reaches terminal speed; on a short-straight circuit like
    Monaco top speed is set by straight length, not drag, and using it
    would force a nonsense drag value.
    """
    try:
        result = simulate(track, car)
    except Exception:
        return 1e6, None

    actual_ms = reference.speed_kph / 3.6

    speed_error = np.sqrt(np.mean(
        ((result.speed - actual_ms) / np.maximum(actual_ms, 1.0)) ** 2))

    time_error = abs(result.lap_time - reference.lap_time) / reference.lap_time

    error = speed_error + 0.5 * time_error

    if reference.drag_limited:
        top_error = abs(result.speed.max() - actual_ms.max()) / actual_ms.max()
        error += top_error

    return error, result

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
    different wing levels.

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
            low, high = SHARED_BOUNDS[name]
            flag = ("  <- at bound"
                    if min(value - low, high - value) < 1e-3 else "")
            print(f"    {name}: {getattr(base_car, name):.3f} "
                  f"-> {value:.3f}{flag}")
        print("\n  Per circuit (aero):")
        for reference, car in zip(references, cars):
            print(f"    {reference.track.name}: "
                  + ", ".join(f"{name} {getattr(car, name):.3f}"
                              for name in per_circuit))

    return shared_values, cars, outcome

def build_reference(year, race, driver, spacing=1.0, lam=None,
                    max_laps=20, drag_limited=True):
    """Load a driver's laps and prepare a reference for fitting.

    Geometry is pooled from up to max_laps clean laps on the same compound
    as the fastest lap, so the lines are comparable. Speed and lap time
    come from the fastest lap alone.
    """
    import telemetry

    session = telemetry.load_session(year, race)
    laps = session.laps.pick_drivers(driver).pick_quicklaps().pick_wo_box()
    fastest = laps.pick_fastest()
    if fastest is None:
        raise ValueError(f"{driver} has no clean laps at {race} {year}.")

    # The fastest lap goes first, because it is the alignment reference
    fastest_pos = fastest.get_pos_data()
    positions = [(fastest_pos['X'].to_numpy(), fastest_pos['Y'].to_numpy())]

    same_compound = laps[laps['Compound'] == fastest['Compound']]
    for _, lap in same_compound.iterlaps():
        if len(positions) >= max_laps:
            break
        if lap['LapNumber'] == fastest['LapNumber']:
            continue
        try:
            pos = lap.get_pos_data()
        except Exception:
            continue
        if pos is None or len(pos) < 50:
            continue
        positions.append((pos['X'].to_numpy(), pos['Y'].to_numpy()))

    tel = fastest.get_telemetry()
    track = track_from_laps(positions, name=race, spacing=spacing, lam=lam,
                            speed_line=(tel['X'], tel['Y'], tel['Speed']))

    return Reference(track=track,
                     speed_kph=track.channels['speed_kph'],
                     lap_time=fastest['LapTime'].total_seconds(),
                     drag_limited=drag_limited)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import style

    style.apply()

    # Geometry check only. Set to False once every circuit passes.
    CHECK_ONLY = False

    # (year, race, driver, smoothing - None for automatic, terminal speed?)
    setups = [
        (2024, 'Monza', 'NOR', None, True),
        (2024, 'Monaco', 'LEC', 100000, False),
        (2024, 'Silverstone', 'HAM', None, True),
        (2024, 'Barcelona', 'VER', None, True),
    ]

    references = []
    for year, race, driver, lam, drag_limited in setups:
        print(f"Loading {race} {year} ({driver})...")
        reference = build_reference(year, race, driver, lam=lam,
                                    drag_limited=drag_limited)
        print(f"  {reference.track.length:.0f}m, "
              f"tightest radius {reference.track.radius.min():.0f}m, "
              f"lap {reference.lap_time:.3f}s")
        print(f"  {reference.track.source}")
        references.append(reference)

    print("\nGeometry check (real F1 cars peak around 5-6g):")
    lateral = {}
    for reference in references:
        track = reference.track
        speed_ms = reference.speed_kph / 3.6
        g = speed_ms ** 2 * track.curvature / GRAVITY
        lateral[track.name] = g
        print(f"  {track.name}: peak {np.nanmax(g):.1f}g, "
              f"points above 6g: {int((g > 6).sum())}")

    if CHECK_ONLY:
        fig, axes = plt.subplots(len(references), 1,
                                 figsize=(13, 2.6 * len(references)))
        for axis, reference in zip(axes, references):
            g = lateral[reference.track.name]
            axis.plot(reference.track.distance, g,
                      color=style.DRIVER_B, linewidth=1)
            axis.axhline(6, color=style.ACCENT, linewidth=0.8,
                         linestyle='--')
            axis.set_ylabel('lateral g')
            axis.set_ylim(0, max(8, min(np.nanmax(g), 15)))
            axis.set_title(f"{reference.track.name}  peak "
                           f"{np.nanmax(g):.1f}g",
                           color=style.TEXT, fontsize=11)
        axes[-1].set_xlabel('Distance (m)')
        style.title(fig, "Geometry check",
                    "Dashed line is 6g. Narrow spikes are noise; "
                    "sustained plateaus are real corners.")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
        raise SystemExit("\nGeometry check only. "
                         "Set CHECK_ONLY = False to run the fit.")

    print("\nFitting...")
    shared_values, cars, outcome = fit_multi(
        references, F1_2024, shared=('mu', 'drive_fraction'))

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
    for axis, reference, car in zip(axes, references, cars):
        result = simulate(reference.track, car)
        axis.plot(reference.track.distance, reference.speed_kph,
                  color=style.DRIVER_A, label='Actual', linewidth=1.2)
        axis.plot(reference.track.distance, result.speed_kph,
                  color=style.DRIVER_B, label='Simulated', linewidth=1.2)
        axis.set_ylabel('km/h')
        axis.set_title(f"{reference.track.name}  "
                       f"{result.lap_time:.3f}s vs "
                       f"{reference.lap_time:.3f}s",
                       color=style.TEXT, fontsize=11)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel('Distance (m)')

    style.title(fig, "Multi-circuit fit",
                "Shared tyre parameters, per-circuit aero")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()