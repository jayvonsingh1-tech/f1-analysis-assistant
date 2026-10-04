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

Two choices in that fit matter more than anything else in this file. The
spline is fitted against lap time, not distance, so it smooths over more
metres where the car is fast. And the smoothing strength is chosen by
agreement between two halves of the laps, not by cross-validation on
position, which leaves too much noise in the curvature. Curvature noise
only ever slows the simulated car, so with noisy geometry the fit
inflates grip to compensate. Tested on made-up laps with a known car:
fitting on the old geometry got grip and downforce wrong by 10 to 30%,
fitting on this one recovers them within about 7%.

On real laps the gain was much smaller, and for a reason worth knowing.
Those tests assumed random position noise. In real data the two halves
of the laps already agree with very little smoothing, so the bumps left
in the curvature are the same on every lap. They are errors in the
position data that repeat at the same place, or real features of the
line, and no smoothing chosen from the data can remove those. Monaco's
two Swimming Pool chicanes are the worst case: quick left-right flicks
at speed, where a metre of position error doubles the lateral g.

Always check implied lateral acceleration (v^2 * curvature) before
fitting. Anything well above 6g is noise, and the fit will bend the car
parameters to compensate for it.

SPEED AND POSITION CLOCKS

FastF1 gets speed and position as two separate streams, each with its
own time stamps. If one clock runs a quarter of a second behind the
other, the real speed trace sits 20m out of place at speed. Every
braking point then looks early or late, and the fit bends the tyre grip
and the drive fraction to explain it. Tested on made-up laps with a
known car: a quarter of a second moved the fitted grip by 8 to 10% and
the drive fraction by 0.13 to 0.3. build_reference() measures the offset
on every lap, and corrects it when the laps agree.

FITTING

Fitting a single circuit is underdetermined - many parameter sets produce
the same lap time. fit_multi() fits several circuits at once, sharing tyre
parameters while letting aero vary per circuit.

Even then tyre grip and downforce trade off against each other: more of
one and less of the other gives nearly the same laps. So the error has a
long shallow valley, and an optimiser that stops anywhere along it looks
converged when it is not. fit_multi() solves the fit as a least-squares
problem, which follows the valley to its lowest point, and checks itself
from a second starting point.

STEPPING

simulate() works along the lap in steps of one grid spacing, and the
acceleration changes within a step: the corner tightens or opens, and
the car gains or loses speed. Using the acceleration at the start of
each step, as this file once did, is the obvious way and it is biased:
every lap came out 0.1 to 0.35s too slow at 1m spacing, more on a
twisty circuit, purely from the arithmetic. Using the acceleration
half-way along each step removes nearly all of that for the same amount
of work. Against the same lap computed with 0.05m steps, 1m steps are
now within 0.01 to 0.08s. What a setup change is worth was never much
affected, because the bias was nearly the same for both cars compared.

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
from scipy.optimize import least_squares
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
    brake_fraction: float = 1.0       # share of grip usable under braking
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

# Smoothing strengths tried when one is chosen automatically, weakest first
_SMOOTHING_CANDIDATES = np.logspace(1.0, 7.0, 19)

def _fit_line(clock, x, y, weights, lam):
    """Smoothing splines for x and y, both against the same clock."""
    return (make_smoothing_spline(clock, x, w=weights, lam=lam),
            make_smoothing_spline(clock, y, w=weights, lam=lam))

def _signed_curvature(spline_x, spline_y, clock):
    """Curvature of a fitted line, positive turning one way and negative
    the other. The formula gives the same answer whatever the splines
    are fitted against."""
    dx = spline_x.derivative(1)(clock)
    dy = spline_y.derivative(1)(clock)
    ddx = spline_x.derivative(2)(clock)
    ddy = spline_y.derivative(2)(clock)

    numerator = dx * ddy - dy * ddx
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    return np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator > 1e-9)

def _agreed_smoothing(halves, clock, speed):
    """Choose the smoothing strength the data itself supports.

    halves: binned samples from two separate sets of laps
    clock, speed: points both halves cover, and the pace at each

    The two halves drove the same corners but carry different noise. For
    each candidate strength, smooth one half with it and compare its
    lateral acceleration with the other half's, smoothed only lightly.
    Too little smoothing and the first half's noise shows up as
    disagreement. Too much and it flattens corners the other half still
    has. The strength with the least disagreement is the best the data
    can identify.

    Each half holds half the samples, so smoothing the full set by the
    same amount takes twice the strength.
    """
    lightly = []
    for half in halves:
        spline_x, spline_y = _fit_line(*half, lam=None)
        lightly.append(
            _signed_curvature(spline_x, spline_y, clock) * speed ** 2)

    disagreement = []
    for lam in _SMOOTHING_CANDIDATES:
        total = 0.0
        for half, other in zip(halves, reversed(lightly)):
            spline_x, spline_y = _fit_line(*half, lam=lam)
            lateral = (_signed_curvature(spline_x, spline_y, clock)
                       * speed ** 2)
            total += np.mean((lateral - other) ** 2)
        disagreement.append(total)

    return 2.0 * _SMOOTHING_CANDIDATES[int(np.argmin(disagreement))]

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
      2. Read the fastest lap's clock at each point of that line
      3. Average the samples into short bins along the track, each bin
         weighted by how many samples it holds
      4. Fit a penalised smoothing spline through the bin averages, x and
         y each as a function of the clock
      5. Differentiate the spline analytically for curvature

    Why the clock and not distance. Curvature is a second derivative, so
    it amplifies noise, and how much that matters depends on speed:
    lateral acceleration is speed squared times curvature, so a small
    curvature error at 250 km/h is a large error in g. Smoothing against
    distance treats every metre alike, which either leaves fast corners
    noisy or rounds off hairpins. Smoothing against time reaches over
    more metres where the car is fast and fewer where it is slow, which
    is what both need. It needs speed_line; without one the fit falls
    back to distance.

    How strong the smoothing is. Left to cross-validation on position,
    the spline is tuned to reproduce positions, and that leaves far too
    much noise in the curvature. Noise only ever slows the simulated car
    (it brakes for wiggles that are not there), so the fit then inflates
    grip to compensate. _agreed_smoothing() chooses the strength from the
    curvature itself, by building the line from two halves of the laps
    and finding where they agree best.

    Binning matters for two reasons. The smoothing spline puts a knot at
    every data point, so samples from different laps landing millimetres
    apart make it numerically singular. And a bin average of n samples
    has 1/n the variance of one sample, so weighting by n is the correct
    way to combine them.

    positions: list of (x, y) raw position arrays, one per lap. The first
        is the alignment reference, so use the fastest lap.
    lam: smoothing strength. None chooses it automatically. Raise it to
        smooth harder.
    max_offset: metres. Samples further than this from the reference line
        are dropped, which removes off-track moments and data glitches.
    bin_width: metres of track per averaging bin.
    speed_line: optional (x, y, speed_kph) from the fastest lap. Sets the
        clock, and is projected onto the fitted line so its speed is
        aligned with the geometry.

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

    # 2. The fastest lap's clock at each reference point. It is rescaled
    # so a whole lap of clock equals the lap length, which keeps smoothing
    # strengths comparable with and without a speed line.
    if speed_line is None:
        pace = np.ones_like(ref_s)
        ref_clock = ref_s
    else:
        line_x = np.asarray(speed_line[0], dtype=float) * scale
        line_y = np.asarray(speed_line[1], dtype=float) * scale
        line_v = np.asarray(speed_line[2], dtype=float)
        usable = (np.isfinite(line_x) & np.isfinite(line_y)
                  & np.isfinite(line_v))
        line_x, line_y, line_v = line_x[usable], line_y[usable], line_v[usable]

        _, index = ref_tree.query(np.column_stack([line_x, line_y]))
        order = np.argsort(ref_s[index], kind='stable')
        pace = np.interp(ref_s, ref_s[index][order], line_v[order] / 3.6)
        pace = savgol_filter(pace, 61, polyorder=2, mode='interp')
        pace = np.maximum(pace, 5.0)

        slowness = 1.0 / pace
        ref_clock = np.concatenate(
            [[0.0], np.cumsum(0.5 * (slowness[1:] + slowness[:-1])
                              * np.diff(ref_s))])
        ref_clock = ref_clock * (ref_s[-1] - ref_s[0]) / ref_clock[-1]

    def clock_at(distance):
        """Clock reading at a distance along the reference line. Past
        either end it carries on at the rate it had there."""
        reading = np.interp(distance, ref_s, ref_clock)
        before = distance < ref_s[0]
        after = distance > ref_s[-1]
        reading[before] = ref_clock[0] + (
            (distance[before] - ref_s[0])
            * (ref_clock[1] - ref_clock[0]) / (ref_s[1] - ref_s[0]))
        reading[after] = ref_clock[-1] + (
            (distance[after] - ref_s[-1])
            * (ref_clock[-1] - ref_clock[-2]) / (ref_s[-1] - ref_s[-2]))
        return reading

    pooled_s, pooled_x, pooled_y, pooled_lap = [], [], [], []
    for number, (lap_x, lap_y) in enumerate(positions):
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
        pooled_lap.append(np.full(len(px), number))

    s = np.concatenate(pooled_s)
    x = np.concatenate(pooled_x)
    y = np.concatenate(pooled_y)
    lap = np.concatenate(pooled_lap)

    # 3. Average into bins along the track, weighted by sample count.
    # Bins are exactly bin_width apart in distance, which keeps the spline
    # well conditioned.
    start = s.min()
    bins = np.floor((s - start) / bin_width).astype(int)

    def binned(keep):
        """Bin averages of the chosen samples: clock, x, y and weight."""
        counts = np.bincount(bins[keep])
        occupied = counts > 0
        centre = start + (np.flatnonzero(occupied) + 0.5) * bin_width
        return (clock_at(centre),
                np.bincount(bins[keep], weights=x[keep])[occupied]
                / counts[occupied],
                np.bincount(bins[keep], weights=y[keep])[occupied]
                / counts[occupied],
                counts[occupied].astype(float))

    # 4. Penalised smoothing spline through the bin averages
    chosen = lam
    if chosen is None and len(positions) >= 2:
        # Compare the halves away from the very ends of the lap, where a
        # half may have no samples yet
        common = np.arange(start + 20.0, s.max() - 20.0, 1.0)
        chosen = _agreed_smoothing(
            [binned(lap % 2 == 0), binned(lap % 2 == 1)],
            clock_at(common), np.interp(common, ref_s, pace))

    spline_x, spline_y = _fit_line(*binned(np.ones(len(s), dtype=bool)),
                                   lam=chosen)

    # 5. Analytic derivatives on a fine grid
    fine_s = np.arange(start + 0.5 * bin_width,
                       start + (bins.max() + 0.5) * bin_width, spacing / 4)
    fine = clock_at(fine_s)
    fx, fy = spline_x(fine), spline_y(fine)
    curvature_fine = np.abs(_signed_curvature(spline_x, spline_y, fine))

    # Even spacing in true arc length along the fitted line
    arc = np.concatenate(
        [[0], np.cumsum(np.hypot(np.diff(fx), np.diff(fy)))])
    distance = np.arange(0, arc[-1], spacing)
    curvature = np.interp(distance, arc, curvature_fine)

    channels = {}
    if speed_line is not None:
        fitted_tree = cKDTree(np.column_stack([fx, fy]))
        _, index = fitted_tree.query(np.column_stack([line_x, line_y]))
        along = arc[index]
        order = np.argsort(along, kind='stable')
        channels['speed_kph'] = np.interp(distance, along[order],
                                          line_v[order])

    against = "distance" if speed_line is None else "lap time"
    if lam is not None:
        strength = f"lam {lam:g}"
    elif chosen is None:
        strength = "automatic"
    else:
        strength = f"automatic, lam {chosen:.3g}"
        # Landing on either end of the candidates means the best
        # strength may lie outside them
        if chosen <= 2.0 * _SMOOTHING_CANDIDATES[0]:
            strength += ", the weakest tried"
        elif chosen >= 2.0 * _SMOOTHING_CANDIDATES[-1]:
            strength += ", the strongest tried"
    return Track(name=name, distance=distance, curvature=curvature,
                 source=f"pooled from {len(positions)} laps, smoothed "
                        f"against {against} ({strength})",
                 channels=channels)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _terminal_speed(car):
    """Top speed, where all the power goes into beating drag."""
    return (car.power / (0.5 * AIR_DENSITY * car.cda)) ** (1 / 3)

def cornering_limit(track, car, halvings=30):
    """Maximum speed at each point, set by lateral grip.

    The car can hold a speed through a point if its grip is at least
    the force the turn needs: grip(v) >= mass * v^2 * curvature. Grip
    depends on speed too (downforce, and load-sensitive tyres), so there
    is no formula for the answer. It is found by halving: take the range
    from standstill to top speed, test the middle, keep the half the
    answer lies in, and repeat. Thirty halvings pin it to a ten-millionth
    of a metre per second.

    Where the car could corner faster than it can ever go, the answer
    is its top speed.
    """
    curvature = track.curvature
    low = np.zeros_like(curvature)
    high = np.full_like(curvature, _terminal_speed(car))

    for _ in range(halvings):
        middle = 0.5 * (low + high)
        holds = car.grip_force(middle) >= car.mass * middle ** 2 * curvature
        low = np.where(holds, middle, low)
        high = np.where(holds, high, middle)

    return 0.5 * (low + high)

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
        # All four tyres brake, and drag helps. brake_fraction below 1
        # allows for them not all reaching their limit together.
        tyre_limit = grip * remaining * car.brake_fraction
        mechanical_limit = car.brake_limit * car.mass * GRAVITY
        force = min(tyre_limit, mechanical_limit) + drag
        return force / car.mass

    # Only the driven axle puts power down. P/v is unbounded as speed
    # approaches zero, but real cars are torque-limited there, so cap it.
    traction_limit = grip * remaining * car.drive_fraction
    power_limit = min(car.power / max(speed, 1.0), car.max_tractive_force)
    force = min(traction_limit, power_limit) - drag
    return force / car.mass

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

    if initial_speed is None and 'speed_kph' in track.channels:
        initial_speed = float(track.channels['speed_kph'][0]) / 3.6

    # The sweeps below step through the lap one point at a time, so they
    # read from plain Python lists rather than numpy arrays (see SPEED in
    # the notes at the top of this file).
    corner_cap = corner_speed.tolist()

    # Curvature half-way between each point and the next
    halfway_curvature = (0.5 * (track.curvature[1:]
                                + track.curvature[:-1])).tolist()

    def one_pass(start):
        # Each step uses the acceleration available half-way along it
        # (see STEPPING in the notes at the top of this file). The
        # speed there is not known yet, so it is predicted: v^2 changes
        # by 2 x acceleration x distance, and over half a step the car
        # is assumed to keep the acceleration it had on the last one.
        # previous and ahead hold speed squared.
        forward = [0.0] * points
        forward[0] = min(float(start), corner_cap[0])
        acceleration = 0.0
        for i in range(1, points):
            previous = forward[i - 1] ** 2
            halfway = math.sqrt(max(previous + acceleration * step, 1.0))
            acceleration = available_longitudinal(
                car, halfway, halfway_curvature[i - 1], braking=False)
            squared = previous + 2 * acceleration * step
            forward[i] = min(math.sqrt(max(squared, 1.0)), corner_cap[i])

        # The same thing backwards from the end of the lap, under
        # braking: how fast could the car have been going one step
        # earlier and still be down to this speed here?
        backward = [0.0] * points
        backward[-1] = corner_cap[-1]
        deceleration = 0.0
        for i in range(points - 2, -1, -1):
            ahead = backward[i + 1] ** 2
            halfway = math.sqrt(max(ahead + deceleration * step, 1.0))
            deceleration = available_longitudinal(
                car, halfway, halfway_curvature[i], braking=True)
            squared = ahead + 2 * deceleration * step
            backward[i] = min(math.sqrt(max(squared, 1.0)), corner_cap[i])

        return np.array(forward), np.array(backward)

    start = corner_speed[0] if initial_speed is None else initial_speed
    forward, backward = one_pass(start)

    if periodic and initial_speed is None:
        lap_end = min(forward[-1], backward[-1])
        forward, backward = one_pass(lap_end)

    speed = np.minimum(np.minimum(forward, backward), corner_speed)

    # What sets the speed at each point: the corner itself, braking for
    # a corner ahead, or how hard the car can accelerate. In a long
    # corner the car sits a shade under the limit, holding its speed
    # against drag, so 'corner' allows 1%. A point where the car could
    # corner at top speed is not limited by the corner.
    limit = np.full(points, 'power', dtype=object)
    limit[backward < forward] = 'brake'
    limit[np.isclose(speed, corner_speed, rtol=0.01)
          & (corner_speed < 0.999 * _terminal_speed(car))] = 'corner'

    lap_time = float(np.sum(step / np.maximum(speed, 1.0)))

    return LapResult(track=track, car=car, speed=speed,
                     lap_time=lap_time, limit=limit)

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """One real lap to fit against.

    lap_time is the time the real car took along the fitted line, worked
    out from its speed at each point. That is the fair target for a
    simulation along the same line. The fitted line is a little shorter
    than the distance the car really covered, so lap_time comes out
    slightly under the timed lap, which is kept in official_time.
    """
    track: Track
    speed_kph: np.ndarray    # real speed at each of track.distance
    lap_time: float          # seconds along this line at the real speed
    drag_limited: bool = True    # does the car reach terminal speed here?
    official_time: float = None  # the timed lap, seconds
    stream_offset: float = None  # seconds the speed clock was corrected by
    clock: str = ""              # what was found about the two clocks
    notes: list = field(default_factory=list)   # problems found in the data

def lap_error(track, car, reference):
    """One number for how far a simulated lap is from the real one.

    Used to report a fit, and kept the same so results stay comparable
    from one run to the next. The fit itself minimises lap_residuals().

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

def lap_residuals(track, car, reference, time_weight=1.0, top_weight=3.0):
    """Every way a simulated lap differs from the real one, as one array.

    A least-squares fit makes the sum of these squared as small as it
    can, so each kind of difference is scaled to count the right amount:
      - the relative speed error at every point, scaled so that together
        they add up to the mean squared speed error
      - the relative lap-time error, times time_weight
      - the relative top-speed error, times top_weight, only where the
        car reaches terminal speed (see lap_error)

    The speed trace carries most of the weight. The top-speed term is
    weighted up because top speed is what pins the drag.
    """
    actual = reference.speed_kph / 3.6
    try:
        result = simulate(track, car)
    except Exception:
        return np.ones(len(actual) + 2)

    speed = ((result.speed - actual) / np.maximum(actual, 1.0)
             / math.sqrt(len(actual)))
    timing = (time_weight * (result.lap_time - reference.lap_time)
              / reference.lap_time)
    top = 0.0
    if reference.drag_limited:
        top = (top_weight * (result.speed.max() - actual.max())
               / actual.max())

    return np.concatenate([speed, [timing, top]])

SHARED_BOUNDS = {
    'mu': (1.0, 2.5),
    'load_sensitivity': (0.0, 0.4),
    'drive_fraction': (0.3, 1.0),
    'brake_limit': (3.0, 9.0),
    'brake_fraction': (0.5, 1.2),
}
AERO_BOUNDS = {
    'cla': (2.5, 7.5),
    'cda': (0.8, 2.5),
}

def fit_multi(references, base_car,
              shared=('mu', 'drive_fraction'),
              per_circuit=('cla', 'cda'),
              verbose=True):
    """Fit one car across several circuits at once.

    Tyre and drivetrain parameters are shared, because they don't change
    between races. Aero is fitted per circuit, because teams genuinely run
    different wing levels.

    Solved as a least-squares problem on lap_residuals(), with a
    trust-region method. The choice matters. A general-purpose minimiser
    working on one error number stalled part-way, stopped at a different
    place for every starting point, and used thousands of laps doing it.
    This one reaches the same answer from any start in a couple of
    hundred laps. To prove it on the day, the fit is run from two very
    different starting points and the two answers are compared.

    Only fit what the laps can pin down. Tested on made-up laps with a
    known car: load_sensitivity trades off against downforce and comes
    out badly wrong, and brake_limit does nothing for a car like this
    (its tyres give up before its brakes do), so both are better held
    at an assumed value than fitted. brake_fraction can be pinned down;
    see braking_check().

    Returns (shared_values, per_circuit_cars, outcome)
    """
    n = len(references)
    fitted = tuple(shared) + tuple(per_circuit)

    lower = np.array([SHARED_BOUNDS[name][0] for name in shared]
                     + [AERO_BOUNDS[name][0] for _ in range(n)
                        for name in per_circuit])
    upper = np.array([SHARED_BOUNDS[name][1] for name in shared]
                     + [AERO_BOUNDS[name][1] for _ in range(n)
                        for name in per_circuit])

    def unpack(values):
        shared_values = {name: float(value) for name, value
                         in zip(shared, values[:len(shared)])}
        cars = []
        cursor = len(shared)
        for _ in range(n):
            aero = {name: float(value) for name, value
                    in zip(per_circuit,
                           values[cursor:cursor + len(per_circuit)])}
            cursor += len(per_circuit)
            cars.append(replace(base_car, **shared_values, **aero))
        return shared_values, cars

    # A circuit only needs simulating again when one of its own numbers
    # changes. The solver nudges one number at a time, so remembering
    # each circuit's laps saves most of the work.
    seen = {}

    def residuals(values):
        _, cars = unpack(values)
        parts = []
        for index, (reference, car) in enumerate(zip(references, cars)):
            key = (index,) + tuple(getattr(car, name) for name in fitted)
            if key not in seen:
                seen[key] = lap_residuals(reference.track, car, reference)
            parts.append(seen[key])
        return np.concatenate(parts)

    def solve(start):
        # The solver needs to start strictly inside the bounds
        margin = 1e-6 * (upper - lower)
        start = np.clip(start, lower + margin, upper - margin)
        return least_squares(residuals, start, bounds=(lower, upper),
                             x_scale='jac', diff_step=1e-3)

    outcome = solve(np.array(
        [getattr(base_car, name) for name in shared]
        + [getattr(base_car, name) for _ in range(n)
           for name in per_circuit]))

    # The same fit again from somewhere very different: plenty of tyre
    # grip and little downforce. Agreement shows the answer comes from
    # the data and not from where the search began.
    span = upper - lower
    far = lower + 0.2 * span
    far[:len(shared)] = (lower + 0.8 * span)[:len(shared)]
    second = solve(far)

    gap = float(np.max(np.abs(second.x - outcome.x) / span))
    if second.cost < outcome.cost:
        outcome = second

    shared_values, cars = unpack(outcome.x)

    if verbose:
        mean_error = np.mean([lap_error(reference.track, car, reference)[0]
                              for reference, car in zip(references, cars)])
        print(f"\nMulti-circuit fit ({len(seen)} laps simulated, "
              f"mean error {mean_error:.4f})")
        if gap < 0.01:
            print("  A second starting point gave the same answer.")
        else:
            print(f"  WARNING: a second starting point gave a different "
                  f"answer, up to {gap:.0%} of a parameter's range away. "
                  f"The better of the two is shown. Treat it with care.")

        def at_bound(value, bounds):
            low, high = bounds
            return ("  <- at bound"
                    if min(value - low, high - value) < 1e-3 else "")

        print("\n  Shared (tyres and drivetrain):")
        for name, value in shared_values.items():
            print(f"    {name}: {getattr(base_car, name):.3f} "
                  f"-> {value:.3f}{at_bound(value, SHARED_BOUNDS[name])}")
        print("\n  Per circuit (aero):")
        for reference, car in zip(references, cars):
            print(f"    {reference.track.name}: "
                  + ", ".join(f"{name} {getattr(car, name):.3f}"
                              + at_bound(getattr(car, name),
                                         AERO_BOUNDS[name])
                              for name in per_circuit))

    return shared_values, cars, outcome

def _stream_offset(pos_time, x, y, car_time, speed_kph, scale=0.1,
                   reach=1.0, window=2.0, longest_gap=1.0):
    """How far the speed data's clock is from the position data's.

    FastF1 gets speed and position as two separate streams, each with
    its own time stamps, and nothing guarantees the two clocks agree. A
    quarter of a second is 20m at speed, enough to put every braking
    point in the wrong place.

    Both streams say how far the car went in any stretch of time: the
    positions directly, the speed trace by adding up speed x time. With
    the clocks apart, the two disagree wherever the car is gaining or
    losing speed. This slides the speed trace in time until they agree
    best over every stretch of `window` seconds, and returns the shift:
    the speed that belongs with a position stamped t is the one stamped
    t + shift. Returns None if it cannot tell.

    Distances over a couple of seconds are compared, not speeds between
    neighbouring samples, because a small error in one time stamp badly
    upsets a speed worked out across a quarter of a second, and hardly
    touches a distance covered in two.

    reach: seconds, the largest shift looked for either way
    longest_gap: seconds. Stretches with a longer hole in either stream
        are left out.
    """
    pos_time = np.asarray(pos_time, dtype=float)
    x = np.asarray(x, dtype=float) * scale
    y = np.asarray(y, dtype=float) * scale
    keep = np.isfinite(pos_time) & np.isfinite(x) & np.isfinite(y)
    order = np.argsort(pos_time[keep], kind='stable')
    pos_time, x, y = pos_time[keep][order], x[keep][order], y[keep][order]

    car_time = np.asarray(car_time, dtype=float)
    speed = np.asarray(speed_kph, dtype=float) / 3.6
    keep = np.isfinite(car_time) & np.isfinite(speed)
    order = np.argsort(car_time[keep], kind='stable')
    car_time, speed = car_time[keep][order], speed[keep][order]

    if len(pos_time) < 50 or len(car_time) < 50:
        return None

    # Distance covered since the first sample, according to each stream
    path = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    covered = np.concatenate(
        [[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1])
                          * np.diff(car_time))])

    def holes_before(time):
        """How many long gaps a stream has before each of its samples."""
        return np.concatenate(
            [[0], np.cumsum(np.diff(time) > longest_gap)])

    # One stretch per position sample: from it to the sample about
    # `window` seconds later. The partner is found by counting samples,
    # not by reading the clock, so that errors in the time stamps have
    # no say in which samples are compared. The speed data has to cover
    # the stretch with `reach` to spare at both ends.
    ahead = max(int(round(window / np.median(np.diff(pos_time)))), 1)
    first = np.arange(len(pos_time) - ahead)
    last = first + ahead

    low = np.searchsorted(car_time, pos_time[first] - reach,
                          side='right') - 1
    high = np.searchsorted(car_time, pos_time[last] + reach, side='left')
    usable = (low >= 0) & (high < len(car_time))
    first, last, low, high = (first[usable], last[usable],
                              low[usable], high[usable])

    pos_holes, car_holes = holes_before(pos_time), holes_before(car_time)
    usable = ((pos_holes[last] == pos_holes[first])
              & (car_holes[high] == car_holes[low]))
    first, last = first[usable], last[usable]
    if len(first) < 50 or covered[-1] <= 0:
        return None

    travelled = path[last] - path[first]

    shifts = np.arange(-reach, reach + 1e-9, 0.01)
    miss = []
    for shift in shifts:
        predicted = (np.interp(pos_time[last] + shift, car_time, covered)
                     - np.interp(pos_time[first] + shift, car_time,
                                 covered))
        # A speed sensor reading slightly high or low would look like a
        # clock offset, so the speed trace is allowed one scale factor
        factor = np.sum(predicted * travelled) / np.sum(predicted ** 2)
        miss.append(np.mean((travelled - factor * predicted) ** 2))

    best = int(np.argmin(miss))
    if best in (0, len(shifts) - 1) or not np.isfinite(miss[best]):
        return None

    # Good data agrees to within a few percent of the distance covered.
    # If even the best shift leaves the two far apart, they are not
    # describing the same lap, and no shift means anything.
    if math.sqrt(miss[best]) > 0.15 * np.mean(travelled):
        return None

    # A parabola through the lowest three points finds the minimum
    # between the 0.01s steps
    before, lowest, after = miss[best - 1:best + 2]
    bend = before - 2.0 * lowest + after
    if bend <= 0:
        return float(shifts[best])
    return float(shifts[best] + 0.005 * (before - after) / bend)

def build_reference(year, race, driver, spacing=1.0, lam=None,
                    max_laps=20, drag_limited=True):
    """Load a driver's laps and prepare a reference for fitting.

    Geometry is pooled from up to max_laps clean laps on the same compound
    as the fastest lap, so the lines are comparable. Speed comes from the
    fastest lap alone, with its clock lined up with the position data's
    first (see _stream_offset).
    """
    import telemetry

    session = telemetry.load_session(year, race)
    laps = session.laps.pick_drivers(driver).pick_quicklaps().pick_wo_box()
    fastest = laps.pick_fastest()
    if fastest is None:
        raise ValueError(f"{driver} has no clean laps at {race} {year}.")

    def seconds(column):
        return column.dt.total_seconds().to_numpy()

    def clock_offset(lap, pos):
        """The speed clock's offset measured on one lap, or None."""
        try:
            car = lap.get_car_data()
            return _stream_offset(seconds(pos['SessionTime']), pos['X'],
                                  pos['Y'], seconds(car['SessionTime']),
                                  car['Speed'])
        except Exception:
            return None

    def longest_gap(times):
        """Longest wait between two samples, in seconds."""
        return float(np.max(np.diff(times))) if len(times) > 1 else 0.0

    # The fastest lap goes first, because it is the alignment reference
    fastest_pos = fastest.get_pos_data()
    positions = [(fastest_pos['X'].to_numpy(), fastest_pos['Y'].to_numpy())]
    own_offset = clock_offset(fastest, fastest_pos)
    offsets = [own_offset]

    # Samples normally arrive four or five times a second. A long hole
    # in the fastest lap means part of it is guesswork.
    notes = []
    pos_time = seconds(fastest_pos['SessionTime'])
    if longest_gap(pos_time) > 1.0:
        notes.append(f"the fastest lap has a {longest_gap(pos_time):.1f}s "
                     f"hole in its position data, so the line there is "
                     f"a guess")

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
        offsets.append(clock_offset(lap, pos))

    # Every lap gives its own measurement of the clock offset. Laps of
    # one session should agree, so the correction is only made when
    # they do, and it uses the middle value so one odd lap cannot move
    # it. The spread is the range the middle half of the laps fall in.
    offsets = np.array([value for value in offsets if value is not None])
    stream_offset = None
    if len(offsets) < 3:
        clock = ("speed and position clocks could not be compared: "
                 "not corrected")
    else:
        quarter, three_quarters = np.percentile(offsets, [25, 75])
        spread = three_quarters - quarter
        if spread >= 0.05:
            own = ("not measured" if own_offset is None
                   else f"{own_offset:+.2f}s")
            clock = (f"speed and position clocks are {offsets.min():+.2f}s "
                     f"to {offsets.max():+.2f}s apart depending on the lap "
                     f"(fastest lap {own}): not corrected")
        else:
            stream_offset = float(np.median(offsets))
            if abs(stream_offset) < 0.005:
                clock = "speed and position clocks agree"
            else:
                side = "later" if stream_offset > 0 else "earlier"
                clock = (f"speed is time-stamped {abs(stream_offset):.2f}s "
                         f"{side} than position (laps agree within "
                         f"{spread:.2f}s): corrected")

    speed_line = None
    if stream_offset is not None:
        # Keep every speed sample exactly as measured, and place each at
        # the position the car had at that moment, read between the
        # position samples either side
        try:
            car = fastest.get_car_data(pad=5, pad_side='both')
            car_time = seconds(car['SessionTime']) - stream_offset
            inside = (car_time >= pos_time[0]) & (car_time <= pos_time[-1])
            if inside.sum() < 50:
                raise ValueError("too little speed data")
            speed_line = (
                np.interp(car_time[inside], pos_time, fastest_pos['X']),
                np.interp(car_time[inside], pos_time, fastest_pos['Y']),
                car['Speed'].to_numpy()[inside])
            if longest_gap(car_time[inside]) > 1.0:
                notes.append(f"the fastest lap has a "
                             f"{longest_gap(car_time[inside]):.1f}s hole "
                             f"in its speed data, so the real speed "
                             f"there is a guess")
        except Exception:
            speed_line = stream_offset = None
            clock = ("the fastest lap's speed data could not be read on "
                     "its own: clocks not corrected")

    if speed_line is None:
        # Fall back to FastF1's own merge of the two streams
        tel = fastest.get_telemetry()
        speed_line = (tel['X'], tel['Y'], tel['Speed'])

    track = track_from_laps(positions, name=race, spacing=spacing, lam=lam,
                            speed_line=speed_line)

    speed_kph = track.channels['speed_kph']
    along_line = float(np.sum(track.step / np.maximum(speed_kph / 3.6, 1.0)))

    return Reference(track=track,
                     speed_kph=speed_kph,
                     lap_time=along_line,
                     drag_limited=drag_limited,
                     official_time=fastest['LapTime'].total_seconds(),
                     stream_offset=stream_offset,
                     clock=clock,
                     notes=notes)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def braking_check(references, base_car, outcome,
                  shared=('mu', 'drive_fraction')):
    """Does the real car brake as hard as the model says it could?

    The model lets all four tyres brake at their limit at once. A real
    car cannot quite manage that: its brakes are split front to rear in
    a fixed proportion, while the load on each axle keeps changing. If
    the real car brakes well inside the model's limit and the fit is not
    told, it lowers the tyre grip to explain it, and downforce then
    comes out too high to make up for the grip.

    This runs the fit again with brake_fraction free, and reports where
    it settles and how much better the laps then match.

    outcome: what fit_multi() returned for the same references with
        brake_fraction held at 1

    How to read the result. Tested on made-up laps with a known car:
      - where the car braked at the full limit, brake_fraction still
        came out at 0.83 to 0.96 and the squared differences fell by 14
        to 27%, because it also soaks up geometry error at corner
        entries
      - where the car braked at 75 to 85% of the limit, it came out at
        0.68 to 0.82 and they fell by 60 to 90%
    The size of the fall separates the two cases more cleanly than the
    value does.

    Returns (shared_values, improvement). improvement is the fraction
    by which the sum of squared differences fell.
    """
    shared_values, _, freed = fit_multi(
        references, base_car, shared=tuple(shared) + ('brake_fraction',),
        verbose=False)
    return shared_values, 1.0 - freed.cost / outcome.cost

def biggest_gaps(result, reference, count=3, window=150.0):
    """Where a simulated lap differs most from the real one.

    Finds the points with the largest speed difference, at least `window`
    metres apart. Returns one dict per point:
      distance   metres from the start of the lap
      sim_kph    simulated speed there
      real_kph   real speed there
      time       seconds the sim loses (+) or gains (-) within `window`
                 metres either side
      needs_g    most lateral acceleration the geometry demands in that
                 stretch, at the speed the real car carried
      has_g      lateral acceleration the simulated car can generate at
                 that same speed

    When needs_g is well above has_g the car cannot take the corner at
    the real speed, so the sim has to slow down. No real car pulls much
    more than 6g, so a needs_g above that points at the geometry, not at
    the car.
    """
    track, car = result.track, result.car
    real_kph = np.asarray(reference.speed_kph, dtype=float)
    real = np.maximum(real_kph / 3.6, 1.0)
    simulated = np.maximum(result.speed, 1.0)

    lost = track.step * (1.0 / simulated - 1.0 / real)
    needs = real ** 2 * track.curvature / GRAVITY
    span = int(round(window / track.step))

    rows = []
    unclaimed = np.abs(result.speed_kph - real_kph)
    for _ in range(count):
        worst = int(np.argmax(unclaimed))
        if unclaimed[worst] <= 0:
            break
        low = max(worst - span, 0)
        high = min(worst + span + 1, len(real))
        tightest = low + int(np.argmax(needs[low:high]))
        rows.append({
            'distance': float(track.distance[worst]),
            'sim_kph': float(result.speed_kph[worst]),
            'real_kph': float(real_kph[worst]),
            'time': float(lost[low:high].sum()),
            'needs_g': float(needs[tightest]),
            'has_g': float(car.grip_force(real[tightest])
                           / (car.mass * GRAVITY)),
        })
        unclaimed[low:high] = 0.0

    return rows

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import style

    style.apply()

    # Geometry check only. Set to False once every circuit passes.
    CHECK_ONLY = False

    # After the fit, also test whether the real car brakes as hard as
    # the model allows (see braking_check). Costs one more fit.
    CHECK_BRAKING = True

    # (year, race, driver, smoothing - None for automatic, terminal speed?)
    # Monaco is left out: it fails the geometry check at 11g, in the two
    # Swimming Pool chicanes, and smoothing does not remove it.
    setups = [
        (2024, 'Monza', 'NOR', None, True),
        # (2024, 'Monaco', 'LEC', None, False),
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
              f"lap {reference.official_time:.3f}s, "
              f"{reference.lap_time:.3f}s along this line")
        print(f"  {reference.track.source}")
        print(f"  {reference.clock}")
        for note in reference.notes:
            print(f"  WARNING: {note}")
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

    results = [simulate(reference.track, car)
               for reference, car in zip(references, cars)]

    print("\nPer-circuit results (against the real lap along the same "
          "line):")
    for reference, result in zip(references, results):
        error = result.lap_time - reference.lap_time
        print(f"  {reference.track.name}: {result.lap_time:.3f}s "
              f"vs {reference.lap_time:.3f}s ({error:+.3f}s), "
              f"top {result.speed_kph.max():.0f} "
              f"vs {reference.speed_kph.max():.0f} km/h")

    print("\nBiggest gaps (time is what the sim loses within 150m "
          "either side):")
    for reference, result in zip(references, results):
        print(f"  {reference.track.name}:")
        for row in biggest_gaps(result, reference):
            # A car on the limit needs about what it has, so only flag
            # a clear shortfall
            flag = ("  <- more than the car has"
                    if row['needs_g'] > 1.1 * row['has_g'] else "")
            print(f"    {row['distance']:5.0f}m: sim {row['sim_kph']:3.0f} "
                  f"vs real {row['real_kph']:3.0f} km/h, "
                  f"{row['time']:+.2f}s, corner needs "
                  f"{row['needs_g']:.1f}g, car has "
                  f"{row['has_g']:.1f}g{flag}")

    if CHECK_BRAKING:
        print("\nBraking check (the same fit with brake_fraction free):")
        freed, improvement = braking_check(references, F1_2024, outcome)
        print(f"  brake_fraction {freed['brake_fraction']:.3f}, "
              f"mu {freed['mu']:.3f}, "
              f"drive_fraction {freed['drive_fraction']:.3f}, "
              f"squared differences down {improvement:.0%}")
        print("  A car braking at the tyres' full limit reads 0.83 to "
              "0.96 and 14 to 27% here.")
        print("  One braking at 75 to 85% of it reads 0.68 to 0.82 and "
              "60 to 90%.")

    fig, axes = plt.subplots(len(references), 1,
                             figsize=(13, 3 * len(references)))
    for axis, reference, result in zip(axes, references, results):
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