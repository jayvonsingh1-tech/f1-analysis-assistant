"""Quasi-steady-state lap time simulation.

Method: compute the curvature of a line round the circuit, find the maximum
cornering speed at each point from the grip available, then sweep forward
under acceleration limits and backward under braking limits. The lowest of
the three at each point is the speed the car can actually carry.

Geometry and car are described separately from the physics, so the same
simulator runs any car on any line. Track geometry currently comes from
FastF1's position data, which turns out to be one map of the circuit,
the same for every car, and not the line each car drove. The map itself
follows a racing line (see GEOMETRY).

MODEL ASSUMPTIONS AND LIMITATIONS

This is a point mass with three refinements: load-sensitive tyres, a
driven-axle limit on acceleration, and a tractive force cap at low speed.
Not modelled:
  - explicit weight transfer between individual wheels
  - aerodynamic balance shifting with ride height
  - a torque curve and gear ratios; power is a single figure
  - DRS and energy deployment (the Car has a DRS figure that nothing
    uses yet; build_reference() fits a lap driven with DRS shut where
    there is one, and warns where there is not)
  - tyre temperature, wear, camber and track surface
  - elevation change, banking and kerbs
  - the car's weight and the air on the day: every lap is run at 850kg
    in sea-level air. A real car weighs its minimum plus the fuel left,
    and on four real race days the air was 3.5 to 7% thinner. Tested on
    made-up laps with a car 60kg lighter and 60kW weaker than assumed:
    what a setup change is worth hardly moved, but the fitted downforce
    came out 8 to 19% too high and the fitted drag 8 to 9% too high

Grip sharing uses a friction ellipse, which assumes equal peak grip
laterally and longitudinally.

GEOMETRY

Geometry quality dominates everything else, and the first thing to know
is what FastF1's position data is. X and Y are not where each car was on
the road. Every car, on every lap, lies on one and the same path, to
within the 10cm the numbers are rounded to. Checked on a real race: 19
drivers, about 20,000 samples each, nine in ten within 16cm of one
driver's line and none more than a metre from it. X and Y are the car's
progress along a fixed map of the circuit.

Three things follow.

There is no noise to average out. More laps put more points on the same
path, which helps, but the data cannot say how much to smooth, because
every lap repeats the same thing. Physics has to: a car cannot change
direction arbitrarily fast, so track_from_laps() makes its spline just
stiff enough to iron out anything that comes and goes more than about
1.5 times a second. The spline is fitted against lap time, not distance,
which is what lets one figure in seconds work at every speed.

The map has faults. In places it steps sideways by 20 to 40cm over a few
metres, and the odd sample lies metres off the path. Curvature is a
second derivative, so a 30cm step looks like a 10g corner. Points more
than 15cm from the fitted line are left out and the line is fitted
again. At Spa in 2020 that took the worst point from 7.2g to 6.1g.

The map follows a racing line, not the middle of the road. Checked in
October 2026 against real track edges (the TU Munich racetrack
database, github.com/TUMFTM/racetrack-database) at Monza, Spa and
Bahrain. Laid over the database's track, the map stays inside the edges
and runs out, in and out through the corners. It leans off the middle
of the road the same way the database's own racing line does, about two
thirds as far.

Even so, the real laps need more grip in the corners than the model car
has. That shows without any model, by working out how much grip the
real lap uses at each point: the force at the tyres over the load on
them (weight plus an assumed downforce). At the hardest braking of the
lap the cars use 1.5 to 1.7. Cornering round the map at the real speed
takes about 2 at many corners and up to 2.7. So the fit carries one
more number for each circuit, line, which scales the map's curvature
(see FITTING). It comes out near 0.8.

What line stands for is not settled, and the laps cannot settle it (see
FITTING): a car with more cornering grip than the tyre figure used here
laps exactly like a car taking wider arcs than the map. A racing line
worked out inside the edges of the road does not account for it. With
the real laps placed on the database's own racing line the fit still
wanted 0.80 at Monza and 0.84 at Spa, against 0.78 and 0.82 on the map,
and fitted no better. (On the middle of the road it wanted 0.45 to
0.50.) So the map is about as good as a line inside the edges gets.

What evidence there is points to real cars taking wider arcs than that,
over the kerbs and beyond the edges. Cornering on the map takes a
quarter more grip than the hardest braking shows the tyres to have. On
a 2024 race lap the map asks for 6.5g at Barcelona, more than F1 cars
are credited with. At Abbey, the first corner at Silverstone, even the
database's racing line asks for 7.2g at the 297 km/h the cars take it
flat out. And corners that turn through a smaller angle tend to need a
lower line (22 corners, correlation 0.4), which is what using more road
would do: width buys far more radius in a short bend than a long one.

In places the map asks for the impossible. At Abbey the 2024 map asks
for 10.2g at the real speed. Left alone, one kink like that spoils a
whole circuit's fit: the simulated car slowed from 297 to 191 km/h
there and lost 0.8s, and the fit is pulled towards more downforce to
get the car through it. Jeddah in 2021 has six such kinks, of 7 to
11g. So build_reference() eases them. Wherever a stretch of 50m or less
asks for more than MOST_LATERAL_G at the real speed, the bend is eased
to the tightest one that speed allows (see _eased), and the loading
report says where. At Jeddah that is 159m in all. Fitted together with
Bahrain, the fit's squared error falls by 37%, and Jeddah's lap comes
out 0.36s from the real one where it was 0.85s. At Monza and Bahrain
nothing is eased, and at Spa 9m.

Longer stretches are left alone and reported. A long stretch over the
limit is a whole corner the map has too tight, and easing whole corners
hides them from the fit. Easing also only catches what is impossible
outright. A corner the map has a fifth too tight, as at Ascari, passes.

SPEED AND POSITION CLOCKS

FastF1 gets speed and position as two separate streams, each with its
own time stamps, and they do not agree. Measured on four real races
from 2020 to 2022, speed is stamped 0.04 to 0.09s later than position.
At speed that puts the real speed trace 3 to 8m out of place, so every
braking point looks late, and the fit bends the tyre grip and the drive
fraction to explain it. Tested on made-up laps with a known car: a
quarter of a second moved the fitted grip by 8 to 10% and the drive
fraction by 0.13 to 0.3. build_reference() measures the offset on every
lap and corrects it.

WHERE THE CAR IS

The positions are not measured afresh at every sample. Checked hop by
hop on real laps: from one sample to the next the position moves on by
what the car's own speed says, to within 10 or 20cm, until a timing
loop, where it is put right in a single jump of anything from 2 to 10m.
There are about a dozen such jumps a lap. Now and then a lap goes wrong
for longer: on Hamilton's fastest lap at Monza in 2020 the positions
sit 15m behind for half a kilometre, through both Lesmo corners. In
2022 the time stamps are rough as well. Four hops in ten disagree with
the speed by more than 2m, because a stamp is up to 0.15s out.

Two things follow. The jumps upset the clock measurement, so it leaves
them out (see _stream_offset). And the positions are a poor way to say
where each speed sample belongs on the track. The car's own speed,
added up, is a far better one: it has no jumps, and the positions are
then needed only to pin down where the count starts (see
_speed_by_distance).

Tested on made-up laps built the same way, with the truth known. Placed
by the positions, the real speed trace was out by 2.3 to 4.1 km/h on
average and by 19 to 31 km/h at its worst point. Placed by the car's own
distance it is out by 1.1 to 2.1 and 12 to 14. On the real Monza lap the
slowest point of each Lesmo moved 13 to 14m: it had been 21m earlier
than on the other 43 laps, and is now 7 to 9m earlier.

FITTING

One circuit on its own pins the car down poorly: many sets of numbers
give nearly the same lap. fit_multi() fits several circuits at once,
sharing what belongs to the car and its driver while letting aero vary
per circuit.

Even then some numbers trade off against each other: a lower line with
less downforce gives nearly the same laps as a higher one with more. So
the error has a long shallow valley, and an optimiser that
stops anywhere along it looks converged when it is not. fit_multi()
solves the fit as a least-squares problem, which follows the valley to
its lowest point, and checks itself from a second starting point.

The fit also carries line for each circuit, a factor on the map's
curvature (see GEOMETRY), and two shares of the tyre's grip:
brake_fraction, how much of it the car uses under braking, and
drive_fraction, how much it can put down under power. Without them one
grip figure has to serve the corners, the braking and the drive out of
the corners, and the real laps want far more for the first than for the
other two. With them the fit's sum of squared differences on real laps
(Monza and Spa 2020, Bahrain 2022) falls by 40 to 65%, and they come
out at line 0.77 to 0.83, brake_fraction 0.6 to 0.74 and drive_fraction
0.47 to 0.49.

The tyre's own grip, mu, is not fitted, because it cannot be. A car
with a grip of 1.75 and a line of 0.8, braking with 0.6 of its grip and
driving with 0.48, laps exactly like one with a grip of 2.19 and a line
of 1 (the map as it is), braking with 0.48 and driving with 0.38. The
laps pin down three things: grip over line (the corners), grip times
brake_fraction, and grip times drive_fraction. Four numbers cannot be
had from three. So mu is held at 1.75 at its reference load, which
agrees with the hardest braking seen on the real laps, and line takes
up the rest. Which of the two cars is nearer the truth is not settled
(see GEOMETRY). The laps, and what a setup change is worth, come out
exactly the same either way.

A brake_fraction of 0.6 does not mean weak brakes. The model brakes at
one steady share of the grip all the way into every corner. A real
driver brakes late and hard, then eases off, and the fitted share is
the average. So the simulated car starts braking too early: on three
real races it was on the brakes for 4 to 5% of the lap where the real
car was not yet slowing hard.

Where the fit is still out. On three real races (Monza and Spa 2020,
Bahrain 2022) the simulated speed is about 10 km/h from the real one,
rms. Split by what limits the simulated car at each point: the 83% of
the lap where it is the engine holds 64% of the squared error, the 13%
under braking holds 25%, and the 4% at the cornering limit holds 11%.
About half of it all lies in the last 200m before the braking points,
where the simulated car is 4 to 8 km/h too fast. It gains about half a
second a lap on the straights and loses it again elsewhere. At the
cornering limit it is 3 to 9 km/h too slow on average and 16 to 18 km/h
out rms: too slow at some corners, too fast at others.

One line number for a whole circuit is a simplification, and those
corner speeds show it. Two ways of doing better through the geometry
have been tried, and neither is in this file. The smoothest line within
a couple of metres of the map fitted four real laps 8 to 50% better,
but the laps did not say how wide to make it (1.5m was best at Monza,
4m or more at Bahrain), and the downforce found swung with the width. A
racing line worked out from real track edges fitted no better than the
map (see GEOMETRY). Why the corners differ is not yet known.

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
from scipy.signal import medfilt, savgol_filter
from scipy.spatial import cKDTree
from scipy.stats import theilslopes

GRAVITY = 9.81
AIR_DENSITY = 1.225      # kg/m3 at sea level, 15C

# The most sideways acceleration the kind of car being fitted can have,
# in g. An F1 car peaks at 5 to 6g. Where a short stretch of the map
# asks for more than this at the speed the real car went,
# build_reference() takes the map to be wrong and eases it (see GEOMETRY
# in the notes above). Any other kind of car has far less, so set this
# for the car.
MOST_LATERAL_G = 6.0

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

    distance is in metres and has to be evenly spaced, because
    simulate() steps along it one spacing at a time. curvature is 1 /
    radius in 1/m. Left and right turns count the same, so its sign is
    dropped.
    """
    name: str
    distance: np.ndarray
    curvature: np.ndarray
    source: str
    channels: dict = field(default_factory=dict)

    def __post_init__(self):
        # Plain arrays of numbers, whatever was handed over (a list, a
        # pandas column), so that everything after this can count on it
        self.distance = np.asarray(self.distance, dtype=float)
        self.curvature = np.abs(np.asarray(self.curvature, dtype=float))

        points = len(self.distance)
        if points < 2 or points != len(self.curvature):
            raise ValueError("A Track needs a distance and a curvature for "
                             "each of its points, and at least two points.")
        spacing = np.diff(self.distance)
        if spacing[0] <= 0 or not np.allclose(spacing, spacing[0],
                                              rtol=1e-6, atol=1e-9):
            raise ValueError("Track.distance has to rise in even steps.")

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

    x, y: positions in FastF1's units of 1/10 m (see scale)
    time: the time of each sample in seconds, as plain numbers. A
        FastF1 time column has to be converted first, with
        .dt.total_seconds().
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
    known = np.isfinite(x) & np.isfinite(y)
    x, y = x[known], y[known]
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

def track_from_laps(positions, name="unknown", spacing=1.0, lam=None,
                    scale=0.1, max_offset=8.0, bin_width=0.5,
                    speed_line=None, cutoff=1.5, fault=0.15):
    """Build a Track from the position data of many laps.

    FastF1's positions all lie on one fixed map of the circuit (see
    GEOMETRY in the notes at the top of this file). One lap gives a
    point on that map only every 15m or so at speed. Each lap is sampled
    at different places, so pooling laps fills the map in.

    Method:
      1. Use the first lap as a reference line, and give every sample
         from every lap a distance along it
      2. Read the fastest lap's clock at each point of that line
      3. Average the samples into short bins along the track, each bin
         weighted by how many samples it holds
      4. Fit a smoothing spline through the bin averages, x and y each
         as a function of the clock, as stiff as `cutoff` calls for
      5. Leave out the bins that lie sharply off the fitted line (map
         faults), and fit again
      6. Differentiate the spline for curvature

    Why the clock and not distance. Curvature is a second derivative, so
    it magnifies every small error, and how much that matters depends on
    speed: lateral acceleration is speed squared times curvature, so a
    small curvature error at 250 km/h is a large error in g. Smoothing
    against time reaches over more metres where the car is fast and
    fewer where it is slow, which is what both need. It needs
    speed_line; without one the fit falls back to distance and to
    scipy's own choice of stiffness.

    How stiff. The data cannot say, because every lap traces the same
    map. A car can: it cannot follow features of a path that come and go
    more than once or twice a second. A smoothing spline of stiffness
    lam, through points carrying `density` weight per unit of clock,
    irons out anything shorter than about 2 pi (lam / density)^(1/4).
    Turned round, that gives the stiffness for a chosen cut-off.

    positions: list of (x, y) raw position arrays, one per lap. Every
        other lap is lined up against the first, so put a lap with no
        holes in its data there.
    lam: stiffness of the spline. None works it out from `cutoff`.
    max_offset: metres. Samples further than this from the reference line
        are dropped, which removes off-track moments.
    bin_width: metres of track per averaging bin.
    speed_line: optional (x, y, speed_kph) from the fastest lap. Sets the
        clock, and is projected onto the fitted line so its speed is
        aligned with the geometry.
    cutoff: features of the path that come and go more often than this,
        in times a second, are smoothed away.
    fault: metres. Bins further than this from the fitted line are
        treated as faults in the map and left out. None keeps them all.

    The track's channels hold x and y, the fitted line itself in metres
    at each point of track.distance, and speed_kph when speed_line is
    given.
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
    # so a whole lap of clock equals the lap length: one unit of clock
    # is then the time the car takes to cover a metre at its average
    # speed.
    mean_speed = None
    if speed_line is None:
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
        mean_speed = (ref_s[-1] - ref_s[0]) / ref_clock[-1]
        ref_clock = ref_clock * mean_speed

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

    pooled_s, pooled_x, pooled_y = [], [], []
    for lap_x, lap_y in positions:
        lap_x = np.asarray(lap_x, dtype=float) * scale
        lap_y = np.asarray(lap_y, dtype=float) * scale
        known = np.isfinite(lap_x) & np.isfinite(lap_y)
        lap_x, lap_y = lap_x[known], lap_y[known]
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

    # 3. Average into bins along the track, weighted by sample count.
    # The smoothing spline puts a knot at every data point, so samples
    # from different laps landing millimetres apart would make it
    # numerically singular. Bins exactly bin_width apart avoid that.
    start = s.min()
    bins = np.floor((s - start) / bin_width).astype(int)
    counts = np.bincount(bins)
    occupied = counts > 0
    centre = start + (np.flatnonzero(occupied) + 0.5) * bin_width
    clock = clock_at(centre)
    bin_x = np.bincount(bins, weights=x)[occupied] / counts[occupied]
    bin_y = np.bincount(bins, weights=y)[occupied] / counts[occupied]
    weight = counts[occupied].astype(float)

    # 4. How stiff to make the spline
    if lam is not None:
        stiffness = lam
    elif mean_speed is None:
        stiffness = None
    else:
        density = weight.sum() / (clock[-1] - clock[0])
        reach = mean_speed / (2.0 * math.pi * cutoff)   # units of clock
        stiffness = density * reach ** 4

    # 5. Fit, leave out the bins that lie off the line, and fit again
    # until no new ones turn up.
    #
    # A corner makes the fitted line sit a little inside the map over
    # tens of metres. That is the smoothing at work, not a fault. A
    # fault is sharp. So each bin's sideways distance from the line is
    # judged against the typical distance over the 10m either side of
    # it. Each stretch left out is widened by 2m either side, so the
    # shoulders of a fault go with it.
    keep = np.ones(len(clock), dtype=bool)
    too_rough = False
    if fault is not None:
        nearby = 2 * int(round(10.0 / bin_width)) + 1
        widen = np.ones(2 * int(round(2.0 / bin_width)) + 1)
        for _ in range(6):
            spline_x, spline_y = _fit_line(clock[keep], bin_x[keep],
                                           bin_y[keep], weight[keep],
                                           lam=stiffness)
            along_x = spline_x.derivative(1)(clock)
            along_y = spline_y.derivative(1)(clock)
            sideways = (((bin_y - spline_y(clock)) * along_x
                         - (bin_x - spline_x(clock)) * along_y)
                        / np.maximum(np.hypot(along_x, along_y), 1e-9))
            sharp = np.abs(sideways - medfilt(sideways, nearby)) > fault
            off = np.convolve(sharp, widen, mode='same') > 0
            if off.mean() > 0.1:
                # A tenth of the lap is not a handful of faults. The
                # data is too rough for this to mean anything.
                keep[:] = True
                too_rough = True
                break
            if np.array_equal(~off, keep):
                break
            keep = ~off

    spline_x, spline_y = _fit_line(clock[keep], bin_x[keep], bin_y[keep],
                                   weight[keep], lam=stiffness)
    left_out = float((~keep).sum() * bin_width)

    # 6. Curvature from the spline's own derivatives, on a fine grid
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

    # The fitted line itself, in metres, so that anything else can be
    # placed along it later
    channels = {'x': np.interp(distance, arc, fx),
                'y': np.interp(distance, arc, fy)}
    if speed_line is not None:
        fitted_tree = cKDTree(np.column_stack([fx, fy]))
        _, index = fitted_tree.query(np.column_stack([line_x, line_y]))
        along = arc[index]
        order = np.argsort(along, kind='stable')
        channels['speed_kph'] = np.interp(distance, along[order],
                                          line_v[order])

    if lam is not None:
        smoothing = f"smoothed with lam {lam:g}"
    elif stiffness is None:
        smoothing = "smoothed against distance, automatic"
    else:
        smoothing = (f"features quicker than {cutoff:g} a second "
                     f"smoothed away")
    if fault is None:
        faults = "map faults kept"
    elif too_rough:
        faults = "too rough to pick out map faults"
    else:
        faults = f"{left_out:.0f}m of map faults left out"
    return Track(name=name, distance=distance, curvature=curvature,
                 source=f"pooled from {len(positions)} laps, {smoothing}, "
                        f"{faults}",
                 channels=channels)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _terminal_speed(car):
    """Top speed, where all the power goes into beating drag."""
    return (car.power / (0.5 * AIR_DENSITY * car.cda)) ** (1 / 3)

def _time_along(step, speed):
    """Seconds taken to cover a line, given the speed in m/s at each of
    its evenly spaced points.

    Time is distance over speed. Each point stands for one step of
    track, half of it either side, covered at that point's speed.
    """
    return float(np.sum(step / np.maximum(speed, 1.0)))

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
        # allows for them not all reaching their limit together, and
        # for a driver who is not on the limit all the way.
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
      2. periodic=True: a flying lap that joins up with itself. The lap
         is run twice. The second time the car starts at the speed it
         finished the first, and has to cross the line slowly enough
         for the corners that follow it. This is the one to use for
         comparing setups, because the start speed then changes with
         the car as it should. Twice the cost.
      3. the real speed at the line, if the track carries a measured
         speed channel (geometry built from driven laps does)
      4. otherwise the cornering limit at the start
    """
    step = track.step
    points = len(track.distance)

    corner_speed = cornering_limit(track, car)

    if initial_speed is not None:
        periodic = False
    elif not periodic and 'speed_kph' in track.channels:
        initial_speed = float(track.channels['speed_kph'][0]) / 3.6

    # The sweeps below step through the lap one point at a time, so they
    # read from plain Python lists rather than numpy arrays (see SPEED in
    # the notes at the top of this file).
    corner_cap = corner_speed.tolist()

    # Curvature half-way between each point and the next
    halfway_curvature = (0.5 * (track.curvature[1:]
                                + track.curvature[:-1])).tolist()

    def accelerate(start):
        # Forwards from the first point, as hard as the car can go.
        #
        # Each step uses the acceleration available half-way along it
        # (see STEPPING in the notes at the top of this file). The
        # speed there is not known yet, so it is predicted: v^2 changes
        # by 2 x acceleration x distance, and over half a step the car
        # is assumed to keep the acceleration it had on the last one.
        # previous holds speed squared.
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
        return forward

    def brake(beyond=None):
        # The same thing backwards from the last point, under braking:
        # how fast could the car have been going one step earlier and
        # still be down to this speed here?
        #
        # beyond is the most the car may be doing one step past the last
        # point, where the next lap begins. None means the track simply
        # ends there.
        backward = [0.0] * points
        if beyond is None:
            backward[-1] = corner_cap[-1]
        else:
            slowing = available_longitudinal(car, beyond, join_curvature,
                                             braking=True)
            backward[-1] = min(math.sqrt(beyond ** 2 + 2 * slowing * step),
                               corner_cap[-1])
        deceleration = 0.0
        for i in range(points - 2, -1, -1):
            ahead = backward[i + 1] ** 2
            halfway = math.sqrt(max(ahead + deceleration * step, 1.0))
            deceleration = available_longitudinal(
                car, halfway, halfway_curvature[i], braking=True)
            squared = ahead + 2 * deceleration * step
            backward[i] = min(math.sqrt(max(squared, 1.0)), corner_cap[i])
        return backward

    start = corner_speed[0] if initial_speed is None else initial_speed
    forward = accelerate(start)
    backward = brake()

    if periodic:
        # The lap joins up with itself, one step after its last point.
        # The braking sweep has just said how fast the car may be going
        # at the first point and still make the corners after it. Brake
        # again with the end of the lap knowing that, then start the lap
        # one step on from the speed the car finishes at.
        join_curvature = 0.5 * (track.curvature[-1] + track.curvature[0])
        most = backward[0]
        backward = brake(beyond=most)
        finish = min(forward[-1], backward[-1])
        gain = available_longitudinal(car, finish, join_curvature,
                                      braking=False)
        start = math.sqrt(max(finish ** 2 + 2 * gain * step, 1.0))
        forward = accelerate(min(start, most))

    forward, backward = np.array(forward), np.array(backward)

    speed = np.minimum(np.minimum(forward, backward), corner_speed)

    # What sets the speed at each point. The corner, if nine tenths or
    # more of the grip is going sideways. (A car holding its speed
    # through a long corner sits a little under the corner's own limit,
    # because some grip has to go on beating drag.) Otherwise braking,
    # if it is slowing for something ahead, or else how hard it can
    # accelerate.
    sideways = (car.mass * speed ** 2 * track.curvature
                / np.maximum(car.grip_force(speed), 1e-9))
    limit = np.full(points, 'power', dtype=object)
    limit[backward < forward] = 'brake'
    limit[sideways >= 0.9] = 'corner'

    return LapResult(track=track, car=car, speed=speed,
                     lap_time=_time_along(step, speed), limit=limit)

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class Reference:
    """One real lap to fit against.

    lap_time is the time the real car took along the fitted line, worked
    out from its speed at each point. That is the fair target for a
    simulation along the same line. The fitted line stops some metres
    short of the timing line at each end, so lap_time usually comes out
    a few tenths under the timed lap, which is kept in official_time.
    """
    track: Track
    speed_kph: np.ndarray    # real speed at each of track.distance
    lap_time: float          # seconds along this line at the real speed
    drag_limited: bool = True    # does the car reach terminal speed here?
    official_time: float = None  # the timed lap, seconds
    stream_offset: float = None  # seconds the speed clock was corrected by
    clock: str = ""              # what was found about the two clocks
    lap_choice: str = "lap"      # which lap this is, and why that one
    driver: str = ""             # whose lap it is, as a three-letter code
    eased: str = ""              # where the map was eased, if anywhere
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
      - the relative speed error at every point, each weighted by the
        share of the lap's time the real car spends there, so that
        together they add up to the mean squared speed error over the
        lap's time
      - the relative lap-time error, times time_weight
      - the relative top-speed error, times top_weight, only where the
        car reaches terminal speed (see lap_error)

    The speed trace carries most of the weight. The top-speed term is
    weighted up because top speed is what pins the drag.

    Why by time and not by distance. A lap is won and lost in seconds,
    and a metre of slow corner takes several times longer to cover than
    a metre of straight. Counting every metre the same lets the long
    fast stretches outvote the corners. Tested on made-up laps with a
    known car: weighting by time brought the fitted grip a third closer
    to the truth and cut the lap-time error by a third.
    """
    actual = reference.speed_kph / 3.6
    try:
        result = simulate(track, car)
    except Exception:
        return np.ones(len(actual) + 2)

    # Time spent at each point is step / speed, so the shares are
    # proportional to 1 / speed
    share = 1.0 / np.maximum(actual, 1.0)
    share = share / share.sum()

    speed = ((result.speed - actual) / np.maximum(actual, 1.0)
             * np.sqrt(share))
    timing = (time_weight * (result.lap_time - reference.lap_time)
              / reference.lap_time)
    top = 0.0
    if reference.drag_limited:
        top = (top_weight * (result.speed.max() - actual.max())
               / actual.max())

    return np.concatenate([speed, [timing, top]])

SHARED_BOUNDS = {
    'mu': (0.5, 2.5),
    'load_sensitivity': (0.0, 0.4),
    'drive_fraction': (0.2, 1.0),
    'brake_limit': (3.0, 9.0),
    'brake_fraction': (0.2, 1.5),
}
CIRCUIT_BOUNDS = {
    'cla': (2.5, 7.5),
    'cda': (0.8, 2.5),
    'line': (0.3, 1.5),
}

def straightened(track, line):
    """The same track with every bend scaled by `line`.

    line is a factor on the map's curvature: 1 is the map itself, 0.8 a
    track that bends 0.8 times as much everywhere. The real laps ask for
    about 0.8. What that stands for is not settled: real cars taking
    wider arcs than the map by using the kerbs, or more cornering grip
    than the tyre figure allows, or some of each (see GEOMETRY and
    FITTING in the notes at the top of this file).

    One number for a whole circuit is a simplification. On real laps
    the corners do not all ask for the same.
    """
    return replace(track, curvature=track.curvature * line)

def fit_multi(references, base_car,
              shared=('brake_fraction', 'drive_fraction'),
              per_circuit=('cla', 'cda', 'line'),
              verbose=True):
    """Fit one car across several circuits at once.

    What is fitted, and what is not:
      - The tyre's grip (mu) is NOT fitted. It is taken from base_car,
        like load_sensitivity, as something known about the tyre.
      - brake_fraction and drive_fraction are shared by every circuit:
        the share of that grip the car uses under braking and under
        power. They belong to the car and its driver.
      - cla and cda are fitted per circuit, because teams genuinely run
        different wing levels.
      - line is fitted per circuit: the factor on the map's curvature
        that the laps ask for (see straightened()). Every circuit has
        its own map.

    Why the grip is held. Grip, line and the two shares cannot all four
    be fitted: the laps pin down only three things about them (see
    FITTING in the notes at the top of this file). So the grip is held
    at a believable figure for the tyre. If the real figure is higher,
    line comes out lower and both shares higher by the same factor, and
    every lap stays the same.

    Solved as a least-squares problem on lap_residuals(), with a
    trust-region method. The choice matters. A general-purpose minimiser
    working on one error number stalled part-way, stopped at a different
    place for every starting point, and used thousands of laps doing it.
    This one reaches the same answer from any start in a few hundred
    laps. To prove it on the day, the fit is run from two very different
    starting points and the two answers are compared.

    Only fit what the laps can pin down. Tested on made-up laps with a
    known car: load_sensitivity trades off against downforce and comes
    out badly wrong, and brake_limit does nothing for a car like this
    (its tyres give up before its brakes do), so both are better held
    at an assumed value than fitted.

    Returns (shared_values, per_circuit_cars, tracks, outcome). tracks
    are the references' tracks with each circuit's line applied.
    Simulate on those.
    """
    n = len(references)

    # Catch a mistyped name here, with a message that says what is
    # allowed. ('mu') without a comma is the text 'mu', not a list.
    if isinstance(shared, str) or isinstance(per_circuit, str):
        raise ValueError("shared and per_circuit are lists of names. One "
                         "name on its own needs a comma: ('mu',)")
    unknown = ([name for name in shared if name not in SHARED_BOUNDS]
               + [name for name in per_circuit
                  if name not in CIRCUIT_BOUNDS])
    if unknown:
        raise ValueError(
            f"Cannot fit {', '.join(unknown)}. shared can hold "
            f"{', '.join(SHARED_BOUNDS)}; per_circuit can hold "
            f"{', '.join(CIRCUIT_BOUNDS)}.")
    if ('mu' in shared and 'brake_fraction' in shared
            and 'drive_fraction' in shared and 'line' in per_circuit):
        raise ValueError("mu, brake_fraction, drive_fraction and line "
                         "cannot all four be fitted: the laps pin down "
                         "only three things about them. See FITTING in "
                         "the notes at the top of this file.")

    lower = np.array([SHARED_BOUNDS[name][0] for name in shared]
                     + [CIRCUIT_BOUNDS[name][0] for _ in range(n)
                        for name in per_circuit])
    upper = np.array([SHARED_BOUNDS[name][1] for name in shared]
                     + [CIRCUIT_BOUNDS[name][1] for _ in range(n)
                        for name in per_circuit])

    def unpack(values):
        shared_values = {name: float(value) for name, value
                         in zip(shared, values[:len(shared)])}
        cars, tracks = [], []
        cursor = len(shared)
        for reference in references:
            own = {name: float(value) for name, value
                   in zip(per_circuit,
                          values[cursor:cursor + len(per_circuit)])}
            cursor += len(per_circuit)
            # line belongs to the track, everything else to the car
            line = own.pop('line', 1.0)
            cars.append(replace(base_car, **shared_values, **own))
            tracks.append(straightened(reference.track, line))
        return shared_values, cars, tracks

    # A circuit only needs simulating again when one of its own numbers
    # changes. The solver nudges one number at a time, so remembering
    # each circuit's laps saves most of the work. Only the latest ones
    # are ever asked for again, so the store is emptied when it gets
    # big.
    seen = {}
    simulated = 0

    def residuals(values):
        nonlocal simulated
        _, cars, tracks = unpack(values)
        parts = []
        cursor = len(shared)
        for index, reference in enumerate(references):
            key = ((index,) + tuple(values[:len(shared)])
                   + tuple(values[cursor:cursor + len(per_circuit)]))
            cursor += len(per_circuit)
            if key not in seen:
                if len(seen) > 2000:
                    seen.clear()
                seen[key] = lap_residuals(tracks[index], cars[index],
                                          reference)
                simulated += 1
            parts.append(seen[key])
        return np.concatenate(parts)

    def solve(start):
        # The solver needs to start strictly inside the bounds. A fit
        # that has not settled within 200 steps is not going to: the
        # ones that work take 10 to 30.
        margin = 1e-6 * (upper - lower)
        start = np.clip(start, lower + margin, upper - margin)
        return least_squares(residuals, start, bounds=(lower, upper),
                             x_scale='jac', diff_step=1e-3, max_nfev=200)

    # Start from the base car, on the map as it is
    own_start = [1.0 if name == 'line' else getattr(base_car, name)
                 for name in per_circuit]
    outcome = solve(np.array(
        [getattr(base_car, name) for name in shared] + own_start * n))

    # The same fit again from somewhere very different: the shared
    # numbers near the top of their range, little downforce and a much
    # lower line. Agreement shows the answer comes from the data and
    # not from where the search began.
    span = upper - lower
    far = lower + 0.2 * span
    far[:len(shared)] = (lower + 0.8 * span)[:len(shared)]
    second = solve(far)

    gap = float(np.max(np.abs(second.x - outcome.x) / span))
    # status 0 is the solver saying it ran out of steps
    ran_out = outcome.status == 0 or second.status == 0
    if second.cost < outcome.cost:
        outcome = second

    shared_values, cars, tracks = unpack(outcome.x)

    if verbose:
        mean_error = np.mean([lap_error(track, car, reference)[0]
                              for track, car, reference
                              in zip(tracks, cars, references)])
        print(f"\nMulti-circuit fit ({simulated} laps simulated, "
              f"mean error {mean_error:.4f})")
        if ran_out:
            print("  WARNING: the fit ran out of steps before it settled. "
                  "Treat the numbers below with care.")
        if gap < 0.01:
            print("  A second starting point gave the same answer.")
        else:
            print(f"  WARNING: a second starting point gave a different "
                  f"answer, up to {gap:.0%} of a parameter's range away. "
                  f"The better of the two is shown. Treat it with care.")

        def at_bound(value, bounds):
            # Within a hundredth of the allowed range of either end
            low, high = bounds
            return ("  <- at bound"
                    if min(value - low, high - value) < 0.01 * (high - low)
                    else "")

        if 'mu' not in shared:
            print(f"\n  Tyre grip taken as {base_car.mu:.2f} (assumed, not "
                  f"fitted).")
        print("  Shared by every circuit:")
        for name, value in shared_values.items():
            print(f"    {name}: {getattr(base_car, name):.3f} "
                  f"-> {value:.3f}{at_bound(value, SHARED_BOUNDS[name])}")
        print("\n  Per circuit (aero, and line: the factor on the map's "
              "curvature that the laps ask for):")
        cursor = len(shared)
        for reference in references:
            own = outcome.x[cursor:cursor + len(per_circuit)]
            cursor += len(per_circuit)
            print(f"    {reference.track.name}: "
                  + ", ".join(f"{name} {value:.3f}"
                              + at_bound(value, CIRCUIT_BOUNDS[name])
                              for name, value in zip(per_circuit, own)))

    return shared_values, cars, tracks, outcome

def _stream_offset(pos_time, x, y, car_time, speed_kph, scale=0.1,
                   reach=1.0, window=2.0, longest_gap=1.0, jump=2.0):
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

    The positions also jump (see WHERE THE CAR IS in the notes at the
    top of this file): several times a lap, one sample lands 2 to 10m
    from where the speed says the car should be. A stretch with a jump
    in it looks like a clock error when it is not, and on real laps that
    moved the answer by up to 0.03s. So the job is done twice. The first
    answer is good enough to spot the jumps: any hop from one sample to
    the next that differs from the speed's version by more than `jump`
    metres. The second leaves those hops out of both sides.

    reach: seconds, the largest shift looked for either way
    longest_gap: seconds. Stretches with a longer hole in either stream
        are left out.
    jump: metres. Hops further than this from what the speed says are
        left out of the second pass.
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

    # The hop from each position sample to the next, and the distance
    # the speed says the car has covered since its own first sample
    hop = np.hypot(np.diff(x), np.diff(y))
    covered = np.concatenate(
        [[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1])
                          * np.diff(car_time))])

    def by_speed(shift):
        """Every hop again, this time from the speed trace slid in
        time by `shift`."""
        return np.diff(np.interp(pos_time + shift, car_time, covered))

    def running(values):
        """Running total, so the sum over any stretch is one
        subtraction."""
        return np.concatenate([[0.0], np.cumsum(values)])

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

    shifts = np.arange(-reach, reach + 1e-9, 0.01)

    def best_shift(sound):
        """The shift at which the two streams agree best over every
        stretch, counting only the hops marked sound."""
        total = running(hop * sound)
        travelled = total[last] - total[first]

        miss = []
        for shift in shifts:
            total = running(by_speed(shift) * sound)
            predicted = total[last] - total[first]
            # A speed sensor reading slightly high or low would look
            # like a clock offset, so the speed trace is allowed one
            # scale factor
            factor = (np.sum(predicted * travelled)
                      / max(np.sum(predicted ** 2), 1e-9))
            miss.append(np.mean((travelled - factor * predicted) ** 2))

        best = int(np.argmin(miss))
        if best in (0, len(shifts) - 1) or not np.isfinite(miss[best]):
            return None

        # Good data agrees to within a few percent of the distance
        # covered. If even the best shift leaves the two far apart, they
        # are not describing the same lap, and no shift means anything.
        if math.sqrt(miss[best]) > 0.15 * np.mean(travelled):
            return None

        # A parabola through the lowest three points finds the minimum
        # between the 0.01s steps
        before, lowest, after = miss[best - 1:best + 2]
        bend = before - 2.0 * lowest + after
        if bend <= 0:
            return float(shifts[best])
        return float(shifts[best] + 0.005 * (before - after) / bend)

    # First with every hop, then again without the ones that jump
    rough = best_shift(np.ones(len(hop)))
    if rough is None:
        return None
    sound = np.abs(hop - by_speed(rough)) <= jump
    better = best_shift(sound)
    return rough if better is None else better

def _distance_along(track, x, y, scale=0.1):
    """How far along a track's line each position sample lies, in metres.

    Takes the nearest point of the line, then adds the part of the gap
    to it that runs along the track.
    """
    line_x, line_y = track.channels['x'], track.channels['y']
    x = np.asarray(x, dtype=float) * scale
    y = np.asarray(y, dtype=float) * scale

    # Samples with no position stay unknown (nan)
    known = np.isfinite(x) & np.isfinite(y)
    distance = np.full(len(x), np.nan)
    x, y = x[known], y[known]

    _, index = cKDTree(np.column_stack([line_x, line_y])).query(
        np.column_stack([x, y]))

    heading_x, heading_y = np.gradient(line_x), np.gradient(line_y)
    length = np.maximum(np.hypot(heading_x, heading_y), 1e-9)
    along = ((x - line_x[index]) * heading_x[index]
             + (y - line_y[index]) * heading_y[index]) / length[index]
    distance[known] = track.distance[index] + along
    return distance

def _speed_by_distance(track, pos_time, x, y, car_time, speed_kph, offset,
                       longest_gap=1.0):
    """One lap's real speed at each point of a track.

    Every speed sample needs a place on the track. Reading it off the
    positions is the obvious way, and the positions are not good enough
    for it (see WHERE THE CAR IS in the notes at the top of this file):
    they jump by metres at the timing loops, and now and then they sit
    15m out for half a kilometre.

    The car's own speed has no jumps. Adding up speed x time gives how
    far the car had travelled at each speed sample, as smoothly as the
    speed itself changes. That is a distance from an unknown starting
    point, on a sensor that may read a shade high or low, so two numbers
    are still needed: where on the track the count starts, and how many
    metres of track one counted metre is worth. The positions give
    those, all several hundred of them together, in a way that takes no
    notice of the ones that are out:
      - the rate is the middle value of the slope between every pair of
        samples (a Theil-Sen fit)
      - the start is the middle value of what is then left over

    A hole in the speed data breaks the count, because nobody knows how
    far the car went while nothing was recorded. So each stretch between
    holes longer than `longest_gap` seconds gets a start of its own.

    offset: seconds the speed is stamped later than position, from
        _stream_offset(). The car's distance at the moment of a position
        sample stamped t is its distance at speed-stamp t + offset.

    Returns None if the two streams cannot be matched up.
    """
    car_time = np.asarray(car_time, dtype=float)
    speed = np.asarray(speed_kph, dtype=float)
    keep = np.isfinite(car_time) & np.isfinite(speed)
    order = np.argsort(car_time[keep], kind='stable')
    car_time, speed = car_time[keep][order], speed[keep][order]
    if len(car_time) < 50:
        return None

    # How far the car had gone at each speed sample, by its own speed
    travelled = np.concatenate(
        [[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / 3.6
                          * np.diff(car_time))])

    # The same for the moment of each position sample, next to where
    # that sample lies on the track
    moment = np.asarray(pos_time, dtype=float) + offset
    on_track = _distance_along(track, x, y)
    usable = ((moment >= car_time[0]) & (moment <= car_time[-1])
              & np.isfinite(on_track))
    if usable.sum() < 50:
        return None
    moment, on_track = moment[usable], on_track[usable]
    by_speed = np.interp(moment, car_time, travelled)

    rate = theilslopes(on_track, by_speed)[0]
    if not 0.95 < rate < 1.05:
        # A speed sensor is never 5% out. The streams do not match.
        return None
    left_over = on_track - rate * by_speed

    # Which stretch between holes each speed sample belongs to, and
    # each position sample. A stretch with too few position samples to
    # judge by keeps the start found from the whole lap.
    stretch = np.concatenate(
        [[0], np.cumsum(np.diff(car_time) > longest_gap)])
    belongs = stretch[np.searchsorted(car_time, moment, side='right') - 1]
    start = np.full(stretch[-1] + 1, float(np.median(left_over)))
    for number in range(stretch[-1] + 1):
        own = belongs == number
        if own.sum() >= 10:
            start[number] = np.median(left_over[own])

    place = start[stretch] + rate * travelled

    # Interpolating needs places that only ever go forwards
    furthest = np.maximum.accumulate(place)
    forward = np.concatenate([[True], place[1:] > furthest[:-1]])
    if forward.sum() < 50:
        return None
    return np.interp(track.distance, place[forward], speed[forward])

def _eased(track, speed_kph, most_g, longest=50.0):
    """Ease the kinks that the real car's speed proves the map wrong on.

    The real car went through every point at a known speed. Speed
    squared times curvature is the sideways acceleration that takes,
    and no car of this kind has more than most_g. So wherever the map
    asks for more, it is the map that is wrong, and the bend is eased to
    the tightest one that speed allows: most_g * g / speed^2.

    Only short stretches are eased. A kink in the map is short: on real
    races the stretches over 6g are 9 to 36m long. A stretch of more
    than `longest` metres is a whole corner the map has too tight, and
    easing whole corners hides them from the fit (tested on made-up
    laps: it put the downforce figure 34% out where it had been 4%). So
    those are left as they are and reported.

    This only catches a bend that is impossible outright. A bend the map
    has a little too tight passes.

    Returns the track, a sentence on what was eased and a sentence on
    what was left alone. A sentence is empty if there is nothing to
    say, and with nothing eased the track is the one handed in. The
    curvature as the map had it is kept in
    track.channels['map_curvature'].
    """
    speed = np.maximum(np.asarray(speed_kph, dtype=float) / 3.6, 1.0)
    needs = speed ** 2 * track.curvature / GRAVITY
    over = needs > most_g
    if not over.any():
        return track, "", ""

    # Each unbroken stretch over the limit runs from a start up to, but
    # not including, an end
    switches = np.flatnonzero(np.diff(np.concatenate(
        [[False], over, [False]]).astype(int)))
    ease = np.zeros(len(over), dtype=bool)
    long_ones = []
    for start, end in zip(switches[::2], switches[1::2]):
        if (end - start) * track.step <= longest:
            ease[start:end] = True
        else:
            long_ones.append(f"{(end - start) * track.step:.0f}m at "
                             f"{track.distance[start]:.0f}m")

    left = ""
    if long_ones:
        left = (f"the map asks for more than {most_g:g}g for "
                f"{', '.join(long_ones)}. That is too long to be a kink, "
                f"so it was left alone: the map may have whole corners "
                f"too tight here")
    if not ease.any():
        return track, "", left

    curvature = np.where(ease, most_g * GRAVITY / speed ** 2,
                         track.curvature)
    worst = int(np.argmax(np.where(ease, needs, 0.0)))
    channels = dict(track.channels, map_curvature=track.curvature)
    said = (f"map eased over {ease.sum() * track.step:.0f}m, where it "
            f"asked for more than {most_g:g}g at the real speed (worst "
            f"{needs[worst]:.1f}g, at {track.distance[worst]:.0f}m)")
    return (replace(track, curvature=curvature, channels=channels), said,
            left)

def _fastest_driver(laps):
    """The three-letter code of whoever set the fastest lap in `laps`.

    FastF1's pick_fastest() takes the quickest lap that counted. A lap
    time that was deleted, for leaving the track say, does not.
    """
    lap = laps.pick_fastest()
    if lap is None:
        raise ValueError("No lap in this session is marked as a fastest "
                         "lap, so there is none to take. Name the driver "
                         "instead.")
    return str(lap['Driver'])

def build_reference(year, race, driver=None, spacing=1.0, lam=None,
                    max_laps=60, drag_limited=True,
                    most_g=MOST_LATERAL_G):
    """Load a driver's laps and prepare a reference for fitting.

    driver is a three-letter code such as 'NOR'. Left out, it is whoever
    set the fastest lap of the race. The reference's `driver` says who
    that was.

    Geometry is pooled from up to max_laps clean laps. Every lap lies on
    the same map path whatever the tyres or the fuel load, so any clean
    lap will do. Speed comes from one lap alone: the driver's quickest
    with DRS shut. Its clock is lined up with the position data's first
    (see _stream_offset), and each speed sample is then placed on the
    track by how far the car had travelled (see _speed_by_distance).

    Last, kinks in the map are eased: short stretches where it asks for
    more than most_g of sideways acceleration at the speed the real car
    went (see _eased). The reference's `eased` says where, if anywhere.
    """
    import telemetry

    session = telemetry.load_session(year, race)
    if driver is None:
        driver = _fastest_driver(session.laps)
    laps = session.laps.pick_drivers(driver).pick_quicklaps().pick_wo_box()

    def seconds(column):
        return column.dt.total_seconds().to_numpy()

    # Where a sample is missing, FastF1 fills in zeros and carries on:
    # the car at X = Y = 0, or doing 0 km/h in the middle of a lap. The
    # next two leave those made-up samples out.
    def measured_positions(lap):
        pos = lap.get_pos_data()
        if 'X' not in pos:
            return pos          # nothing at all was recorded
        return pos[(pos['X'] != 0) | (pos['Y'] != 0)]

    def measured_speed(lap, **padding):
        car = lap.get_car_data(**padding)
        if 'Speed' not in car:
            return car
        return car[car['Speed'] > 0]

    def clock_offset(lap, pos):
        """The speed clock's offset measured on one lap, or None."""
        try:
            car = measured_speed(lap)
            return _stream_offset(seconds(pos['SessionTime']), pos['X'],
                                  pos['Y'], seconds(car['SessionTime']),
                                  car['Speed'])
        except Exception:
            return None

    def longest_gap(times):
        """Longest wait between two samples, in seconds."""
        return float(np.max(np.diff(times))) if len(times) > 1 else 0.0

    def drs_share(lap):
        """Share of a lap driven with DRS open. FastF1's DRS channel
        reads 10 or more while the flap is open. 0 if there is no
        telling."""
        try:
            return float(np.mean(lap.get_car_data()['DRS'].to_numpy() >= 10))
        except Exception:
            return 0.0

    # Every clean lap that has a time and position data, in the order
    # they were driven
    usable = []
    for _, lap in laps.iterlaps():
        try:
            lap_time = lap['LapTime'].total_seconds()
            pos = measured_positions(lap)
        except Exception:
            continue
        if math.isfinite(lap_time) and len(pos) >= 50:
            usable.append((lap_time, lap, pos))
    if not usable:
        raise ValueError(f"{driver} has no clean laps with position data "
                         f"at {race} {year}.")

    # The lap to fit is the quickest one with DRS shut. In a race DRS
    # comes with a tow from the car ahead, and both cut drag in a way
    # the model knows nothing about. If DRS was open on every lap, the
    # quickest of all has to do, and a warning further down says so.
    quickest_first = sorted(usable, key=lambda entry: entry[0])
    official_time, fastest, fastest_pos = quickest_first[0]
    lap_choice = f"lap {int(fastest['LapNumber'])}, the quickest"
    drs_open = drs_share(fastest)
    if drs_open >= 0.01:
        for lap_time, lap, pos in quickest_first[1:]:
            if drs_share(lap) < 0.01:
                lap_choice = (f"lap {int(lap['LapNumber'])}, the quickest "
                              f"with DRS shut (lap "
                              f"{int(fastest['LapNumber'])} was "
                              f"{lap_time - official_time:.2f}s quicker "
                              f"with it open)")
                official_time, fastest, fastest_pos = lap_time, lap, pos
                drs_open = 0.0
                break

    pos_time = seconds(fastest_pos['SessionTime'])
    positions = [(fastest_pos['X'].to_numpy(), fastest_pos['Y'].to_numpy())]
    holes = [longest_gap(pos_time) > 1.0]
    own_offset = clock_offset(fastest, fastest_pos)
    offsets = [own_offset]

    for lap_time, lap, pos in usable:
        if len(positions) >= max_laps:
            break
        if lap is fastest:
            continue
        positions.append((pos['X'].to_numpy(), pos['Y'].to_numpy()))
        holes.append(longest_gap(seconds(pos['SessionTime'])) > 1.0)
        offsets.append(clock_offset(lap, pos))

    # track_from_laps() lines every lap up against the first one in the
    # list, so the first should have no hole of over a second in it
    # (samples normally arrive four or five times a second). That is the
    # fastest lap unless it has a hole and another lap has none.
    notes = []
    if drs_open >= 0.01:
        notes.append(f"DRS was open for {drs_open:.0%} of this lap, and no "
                     f"clean lap had it shut. The model has no DRS, so the "
                     f"fitted drag is a blend of open and shut")
    fastest_has_hole = holes[0]
    if holes[0] and not all(holes):
        positions.insert(0, positions.pop(holes.index(False)))
    elif holes[0]:
        notes.append("every lap has a hole of over a second in its "
                     "position data, so the line may be wrong there")

    # Every lap gives its own measurement of the clock offset, and on
    # real data they scatter by about 0.03s either way. The middle value
    # is used, so one odd lap cannot move it. How well that middle value
    # is pinned down improves with the number of laps: roughly the
    # width of the middle half of the laps over the square root of how
    # many there are. The correction is made only when that is small.
    offsets = np.array([value for value in offsets if value is not None])
    stream_offset = None
    if len(offsets) < 3:
        clock = ("speed and position clocks could not be compared: "
                 "not corrected")
    else:
        quarter, three_quarters = np.percentile(offsets, [25, 75])
        doubt = (three_quarters - quarter) / math.sqrt(len(offsets))
        if doubt >= 0.02:
            own = ("not measured" if own_offset is None
                   else f"{own_offset:+.2f}s")
            clock = (f"speed and position clocks are {offsets.min():+.2f}s "
                     f"to {offsets.max():+.2f}s apart depending on the lap "
                     f"(this lap {own}): not corrected")
        else:
            stream_offset = float(np.median(offsets))
            if abs(stream_offset) < 0.005:
                clock = "speed and position clocks agree"
            else:
                side = "later" if stream_offset > 0 else "earlier"
                clock = (f"speed is time-stamped {abs(stream_offset):.2f}s "
                         f"{side} than position (measured on "
                         f"{len(offsets)} laps, give or take "
                         f"{doubt:.3f}s): corrected")

    # A first placing of the speed samples: each at the position the car
    # had at that moment, read between the position samples either side.
    # The positions are a few metres out in places, but this is good
    # enough for track_from_laps() to set its clock by. With no clock
    # correction to make, the two streams are taken as they come.
    shift = 0.0 if stream_offset is None else stream_offset
    speed_line = car_stamps = car_speed = None
    try:
        car = measured_speed(fastest, pad=5, pad_side='both')
        car_stamps = seconds(car['SessionTime'])
        car_speed = car['Speed'].to_numpy()
        car_time = car_stamps - shift
        inside = (car_time >= pos_time[0]) & (car_time <= pos_time[-1])
        if inside.sum() < 50:
            raise ValueError("too little speed data")
        speed_line = (
            np.interp(car_time[inside], pos_time, fastest_pos['X']),
            np.interp(car_time[inside], pos_time, fastest_pos['Y']),
            car_speed[inside])
        if longest_gap(car_time[inside]) > 1.0:
            notes.append(f"this lap has a "
                         f"{longest_gap(car_time[inside]):.1f}s hole in "
                         f"its speed data, so the real speed there is a "
                         f"guess")
    except Exception:
        speed_line = car_stamps = None
        if stream_offset is not None:
            stream_offset = None
            clock = ("this lap's speed data could not be read on its "
                     "own: clocks not corrected")

    if speed_line is None:
        # The last resort: FastF1's own merge of the two streams. It
        # reads positions between samples, so a made-up sample at 0, 0
        # drags its neighbours towards it. Keep only what lies within
        # 20m of a position that was really measured.
        tel = fastest.get_telemetry()
        real = cKDTree(np.column_stack([fastest_pos['X'], fastest_pos['Y']]))
        spot = np.column_stack([tel['X'], tel['Y']]).astype(float)
        known = np.isfinite(spot).all(axis=1)
        away = np.full(len(tel), np.inf)
        away[known] = real.query(spot[known])[0]
        tel = tel[(away < 200) & (tel['Speed'] > 0).to_numpy()]
        speed_line = (tel['X'], tel['Y'], tel['Speed'])

    track = track_from_laps(positions, name=race, spacing=spacing, lam=lam,
                            speed_line=speed_line)

    # Now the line exists, place each speed sample properly: by how far
    # the car had travelled, and not by where the positions put it
    placed = None
    if car_stamps is not None:
        try:
            placed = _speed_by_distance(track, pos_time, fastest_pos['X'],
                                        fastest_pos['Y'], car_stamps,
                                        car_speed, shift)
        except Exception:
            placed = None
    if placed is not None:
        track.channels['speed_kph'] = placed
    else:
        notes.append("this lap's speed could not be placed by the car's "
                     "own distance, so its positions were used: the real "
                     "speed may be a few metres out of place")
        if fastest_has_hole:
            notes.append("this lap has a hole of over a second in its "
                         "position data, so the real speed there is a "
                         "guess")

    speed_kph = track.channels['speed_kph']
    along_line = _time_along(track.step, speed_kph / 3.6)

    # Where the map asks for more than a car can do at the speed the
    # real one went, the map is wrong there
    track, eased, left_alone = _eased(track, speed_kph, most_g)
    if left_alone:
        notes.append(left_alone)

    # The line covers all but the last few metres of the lap, so the two
    # times should agree to well within 1%
    if not 0.99 < along_line / official_time < 1.01:
        notes.append(f"at the real speed this line takes {along_line:.1f}s "
                     f"but the lap was timed at {official_time:.1f}s, so "
                     f"the line or the speed data is wrong somewhere")

    return Reference(track=track,
                     speed_kph=speed_kph,
                     lap_time=along_line,
                     drag_limited=drag_limited,
                     official_time=official_time,
                     stream_offset=stream_offset,
                     clock=clock,
                     lap_choice=lap_choice,
                     driver=driver,
                     eased=eased,
                     notes=notes)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

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

# One change to the car at a time: (what to call it, which number, how,
# by how much). 'times' multiplies the number, 'plus' adds to it.
SETUP_CHANGES = [
    ('10% more downforce', 'cla', 'times', 1.10),
    ('10% more drag', 'cda', 'times', 1.10),
    ('10kg lighter', 'mass', 'plus', -10.0),
    ('5% more power', 'power', 'times', 1.05),
    ('5% more tyre grip', 'mu', 'times', 1.05),
]

def setup_effects(track, car, changes=SETUP_CHANGES):
    """What each change to the car is worth on a track, in seconds a lap.

    Makes one change at a time, runs the flying lap again and takes the
    difference from the unchanged car. Negative is quicker. A flying lap
    (periodic=True) is used so that the speed the car starts the lap at
    changes with the car, as it would.

    Each line is one number changed with everything else held, which is
    not how a real car changes: a bigger wing brings drag with its
    downforce. Changes this small add up closely enough, so the worth of
    a wing is roughly the downforce line plus the drag line, each scaled
    to what the wing really gives.

    How far to trust them. Tested on 80 made-up sessions with a known
    car, each fitted from data with the faults real data has. What
    matters most is whether the corners all ask for the same line
    number (see FITTING in the notes at the top of this file).
      - Where they did (54 sessions), every figure was within 10% of
        the truth nine times in ten.
      - Where each corner asked for its own (26 sessions), as on real
        laps, the drag, power and grip figures were within 9, 10 and
        13% nine times in ten. The downforce figure was typically 14%
        out and 46% at worst. The mass figure was typically 10% out and
        26% at worst.
    So on real circuits trust the drag and power figures most, and
    treat the downforce and mass figures as rough. There is one check
    from outside: teams reckon 10kg of fuel costs about 0.3s a lap, and
    on real fitted circuits this gives 0.23s at Monza, 0.22s at Bahrain
    and 0.31s at Spa.

    Returns a dict of {what the change is called: seconds}.
    """
    base = simulate(track, car, periodic=True).lap_time

    effects = {}
    for label, name, how, amount in changes:
        now = getattr(car, name)
        changed = replace(car, **{name: now * amount if how == 'times'
                                  else now + amount})
        effects[label] = simulate(track, changed,
                                  periodic=True).lap_time - base
    return effects

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import style

    style.apply()

    # Geometry check only. Set to False once every circuit passes.
    CHECK_ONLY = False

    # (year, race, driver, spline stiffness - None to work it out,
    #  does the car reach terminal speed?)
    # A driver of None means whoever set the fastest lap of the race.
    # Put a three-letter code such as 'HAM' there for that driver's
    # quickest lap instead.
    # Monaco is left out: its map asked for 11g in the two Swimming Pool
    # chicanes. Kinks like that are now eased (see _eased), so it may be
    # worth trying again.
    setups = [
        (2024, 'Monza', None, None, True),
        # (2024, 'Monaco', None, None, False),
        (2024, 'Silverstone', None, None, True),
        (2024, 'Barcelona', None, None, True),
    ]

    references = []
    for year, race, driver, lam, drag_limited in setups:
        who = driver or "the fastest lap of the race"
        print(f"Loading {race} {year} ({who})...")
        reference = build_reference(year, race, driver, lam=lam,
                                    drag_limited=drag_limited)
        print(f"  {reference.track.length:.0f}m, "
              f"tightest radius {reference.track.radius.min():.0f}m")
        print(f"  {reference.driver}, {reference.lap_choice}: "
              f"{reference.official_time:.3f}s, "
              f"{reference.lap_time:.3f}s along this line")
        print(f"  {reference.track.source}")
        print(f"  {reference.clock}")
        if reference.eased:
            print(f"  {reference.eased}")
        for note in reference.notes:
            print(f"  WARNING: {note}")
        references.append(reference)

    print("\nGeometry check (the map as it came, at the real speed; real "
          "F1 cars peak around 5-6g):")
    lateral = []
    for reference in references:
        track = reference.track
        speed_ms = reference.speed_kph / 3.6
        # The map before any easing, which is what is being checked
        on_map = track.channels.get('map_curvature', track.curvature)
        g = speed_ms ** 2 * on_map / GRAVITY
        lateral.append(g)
        over = int((g > MOST_LATERAL_G).sum())
        eased = int((track.curvature != on_map).sum())
        print(f"  {track.name}: peak {np.nanmax(g):.1f}g, points above "
              f"{MOST_LATERAL_G:g}g: {over}"
              + (f" ({eased} eased for the fit)" if eased else ""))

    if CHECK_ONLY:
        # squeeze=False gives a grid of plots even when there is only
        # one, so the loop below works for any number of circuits
        fig, axes = plt.subplots(len(references), 1, squeeze=False,
                                 figsize=(13, 2.6 * len(references)))
        axes = axes[:, 0]
        for axis, reference, g in zip(axes, references, lateral):
            axis.plot(reference.track.distance, g,
                      color=style.DRIVER_B, linewidth=1)
            axis.axhline(MOST_LATERAL_G, color=style.ACCENT,
                         linewidth=0.8, linestyle='--')
            axis.set_ylabel('lateral g')
            axis.set_ylim(0, max(8, min(np.nanmax(g), 15)))
            axis.set_title(f"{reference.track.name}  peak "
                           f"{np.nanmax(g):.1f}g",
                           color=style.TEXT, fontsize=11)
        axes[-1].set_xlabel('Distance (m)')
        style.title(fig, "Geometry check",
                    f"Dashed line is {MOST_LATERAL_G:g}g. Narrow spikes "
                    "above it are faults in the map, and are eased for "
                    "the fit; sustained plateaus are real corners.")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
        raise SystemExit("\nGeometry check only. "
                         "Set CHECK_ONLY = False to run the fit.")

    print("\nFitting...")
    shared_values, cars, tracks, outcome = fit_multi(references, F1_2024)

    # tracks are the references' tracks with each circuit's line applied
    results = [simulate(track, car) for track, car in zip(tracks, cars)]

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

    print("\nWhat a change to the car is worth (seconds a lap, flying "
          "lap, one change at a time):")
    for reference, track, car in zip(references, tracks, cars):
        print(f"  {reference.track.name}:")
        for label, seconds in setup_effects(track, car).items():
            print(f"    {label:20s} {seconds:+.2f}s")
    print("  Drag and power: good to about 10%. Tyre grip: about 15%. "
          "Downforce and mass are\n  rough: typically 10 to 15% out, "
          "sometimes 30%. See setup_effects() for the tests.")

    fig, axes = plt.subplots(len(references), 1, squeeze=False,
                             figsize=(13, 3 * len(references)))
    axes = axes[:, 0]
    for axis, reference, result in zip(axes, references, results):
        axis.plot(reference.track.distance, reference.speed_kph,
                  color=style.DRIVER_A, label='Actual', linewidth=1.2)
        axis.plot(reference.track.distance, result.speed_kph,
                  color=style.DRIVER_B, label='Simulated', linewidth=1.2)
        axis.set_ylabel('km/h')
        whose = f" ({reference.driver})" if reference.driver else ""
        axis.set_title(f"{reference.track.name}{whose}  "
                       f"{result.lap_time:.3f}s vs "
                       f"{reference.lap_time:.3f}s",
                       color=style.TEXT, fontsize=11)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel('Distance (m)')

    style.title(fig, "Multi-circuit fit",
                "Tyre grip held. Braking and drive shares fitted for all "
                "circuits, aero and line for each.")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()