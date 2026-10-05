"""Checks for Simulator.py. Run them with:  python simulator_checks.py

Each check compares the simulator with an answer known some other way:
a formula from a physics textbook, a rule every lap has to obey, or the
simulator itself run a different way. None of them needs any F1 data,
so the whole file runs in a few seconds.

Run it after every change to Simulator.py. A line that says BAD means
the change broke something that used to work. The last line says ALL OK,
or how many checks failed.

The checks were themselves checked: the simulator was broken on purpose
in 18 different places, one at a time, and every break made at least
one line here say BAD.

The last section runs the simulator at the size of a Formula Student
car and its events, to show the physics holds there too.
"""

import math
import sys

import numpy as np

import Simulator as sim
from Simulator import AIR_DENSITY, F1_2024, GRAVITY

failed = []
count = 0

def check(label, good, detail=""):
    """One line for a check that is simply right or wrong."""
    global count
    count += 1
    if not good:
        failed.append(label)
    print(f"  {'ok ' if good else 'BAD'} {label}"
          + (f"  ({detail})" if detail else ""))

def close(label, got, want, tolerance):
    """One line for a check against a number known some other way."""
    check(label, abs(got - want) <= tolerance,
          f"got {got:.4f}, should be {want:.4f}")

# ---------------------------------------------------------------------------
# Tracks and cars to test with
# ---------------------------------------------------------------------------

def straight(length, spacing=1.0):
    distance = np.arange(0, length, spacing)
    return sim.Track(name='straight', distance=distance,
                     curvature=np.zeros_like(distance), source='made up')

def circle(radius, length=2000.0, spacing=1.0):
    distance = np.arange(0, length, spacing)
    return sim.Track(name='circle', distance=distance,
                     curvature=np.full_like(distance, 1.0 / radius),
                     source='made up')

# A made-up 4.2km circuit: where each corner starts, how long it is and
# its radius, all in metres. A hairpin, a chicane, medium corners and
# two fast sweeps.
CORNERS = [(400, 120, 40), (800, 50, 18), (880, 50, 25), (1500, 300, 150),
           (2200, 80, 30), (2600, 200, 90), (3300, 90, 22),
           (3600, 250, 120)]

def circuit(spacing=1.0, corners=CORNERS, length=4200.0):
    """The made-up circuit. Each corner tightens over 20m and opens out
    over 20m, as a real one does."""
    distance = np.arange(0, length, spacing)
    curvature = np.zeros_like(distance)
    for start, how_long, radius in corners:
        curvature += np.interp(
            distance,
            [start - 20, start, start + how_long, start + how_long + 20],
            [0, 1 / radius, 1 / radius, 0], left=0, right=0)
    return sim.Track(name='made-up circuit', distance=distance,
                     curvature=curvature, source='made up')

# A car with no wings and tyres whose grip does not change with load,
# so that the textbook formulas apply exactly
NO_WINGS = sim.Car(name='no wings', mass=800.0, power=300_000.0, cla=0.0,
                   cda=1.0, mu=1.5, brake_limit=20.0)

# ---------------------------------------------------------------------------
print("1. Corners with an exact answer")
# ---------------------------------------------------------------------------

# No downforce: the tyres give mu x weight, the corner needs m v^2 / R,
# so v = sqrt(mu g R)
for radius in (20.0, 80.0, 200.0):
    speed = sim.cornering_limit(circle(radius), NO_WINGS)[0]
    close(f"no downforce, {radius:.0f}m radius (m/s)", speed,
          math.sqrt(1.5 * GRAVITY * radius), 1e-5)

# With downforce the load grows with speed squared, and the same balance
# gives v^2 = mu g R / (1 - mu q R / m), where q = half x air density x ClA
winged = sim.replace(NO_WINGS, power=600_000.0, cla=4.0, cda=1.2)
q = 0.5 * AIR_DENSITY * winged.cla
for radius in (20.0, 80.0, 150.0):
    exact = math.sqrt(1.5 * GRAVITY * radius
                      / (1 - 1.5 * q * radius / winged.mass))
    speed = sim.cornering_limit(circle(radius), winged)[0]
    close(f"with downforce, {radius:.0f}m radius (m/s)", speed, exact, 1e-5)

# The F1 car's tyres lose grip as the load on them rises:
# mu = mu at the reference load x (load per tyre / reference load) ^ -k
for kph in (100.0, 250.0):
    speed = kph / 3.6
    load = (F1_2024.mass * GRAVITY
            + 0.5 * AIR_DENSITY * F1_2024.cla * speed ** 2)
    exact = F1_2024.mu * ((load / 4) / F1_2024.reference_load) ** (
        -F1_2024.load_sensitivity)
    close(f"F1 car, tyre grip at {kph:.0f} km/h", F1_2024.effective_mu(speed),
          exact, 1e-9)
check("and has less of it at 250 km/h than at 100",
      F1_2024.effective_mu(250 / 3.6) < F1_2024.effective_mu(100 / 3.6))

# With tyres like that there is no neat formula for the corner speed,
# but at the answer the grip must still equal what the corner needs
for radius in (15.0, 60.0, 120.0):
    speed = sim.cornering_limit(circle(radius), F1_2024)[0]
    needs = F1_2024.mass * speed ** 2 / radius
    close(f"F1 car, {radius:.0f}m radius: grip over what the corner needs",
          F1_2024.grip_force(speed) / needs, 1.0, 1e-6)

# ---------------------------------------------------------------------------
print("2. Straights with an exact answer")
# ---------------------------------------------------------------------------

# Top speed is where drag x speed uses all the power
top = sim._terminal_speed(F1_2024)
run = sim.simulate(straight(6000.0), F1_2024, initial_speed=30.0)
close("F1 car after 6km flat out reaches its top speed (m/s)",
      run.speed[-1], top, 0.05)
close("at that speed, drag x speed over power", F1_2024.drag(top) * top
      / F1_2024.power, 1.0, 1e-9)

# All the power going into speed, with next to no drag:
# v^3 = v0^3 + 3 P s / m
no_drag = sim.replace(NO_WINGS, cda=1e-6, mu=5.0)
run = sim.simulate(straight(800.0), no_drag, initial_speed=40.0)
for metres in (100, 400, 799):
    exact = (40.0 ** 3 + 3 * 300_000.0 * metres / 800.0) ** (1 / 3)
    close(f"power alone, speed after {metres}m (m/s)", run.speed[metres],
          exact, 0.01)

# More power than the tyres can use: acceleration is mu g x the share
# the driven wheels have, so v^2 = v0^2 + 2 a s
wheelspin = sim.replace(NO_WINGS, power=5_000_000.0, cda=1e-6, mu=1.2,
                        drive_fraction=0.5)
run = sim.simulate(straight(300.0), wheelspin, initial_speed=5.0)
for metres in (50, 150, 299):
    exact = math.sqrt(25.0 + 2 * 1.2 * GRAVITY * 0.5 * metres)
    close(f"grip alone, speed after {metres}m (m/s)", run.speed[metres],
          exact, 0.01)

# A car with plenty of grip and power, but a limit on the force its
# wheels can give: acceleration is that force over the mass
capped = sim.replace(NO_WINGS, power=5_000_000.0, cda=1e-6, mu=5.0,
                     max_tractive_force=4000.0)
run = sim.simulate(straight(300.0), capped, initial_speed=5.0)
for metres in (100, 299):
    exact = math.sqrt(25.0 + 2 * (4000.0 / 800.0) * metres)
    close(f"capped force, speed after {metres}m (m/s)", run.speed[metres],
          exact, 0.01)

# Braking for a tight corner at the end of a straight, no wings and next
# to no drag: the car slows at mu g
distance = np.arange(0, 600.0, 1.0)
curvature = np.zeros_like(distance)
curvature[500:] = 1.0 / 20.0
to_a_corner = sim.Track(name='straight then corner', distance=distance,
                        curvature=curvature, source='made up')
stopper = sim.replace(NO_WINGS, power=2_000_000.0, cda=1e-6)

def slowing(result, first, last):
    """Deceleration between two points of the braking zone, in m/s2."""
    braking = np.flatnonzero(result.limit == 'brake')
    a, b = braking[first], braking[last]
    return ((result.speed[a] ** 2 - result.speed[b] ** 2)
            / (2 * (distance[b] - distance[a])))

run = sim.simulate(to_a_corner, stopper, initial_speed=90.0)
close("braking, deceleration (m/s2)", slowing(run, 5, -10),
      1.5 * GRAVITY, 0.02)
close("and the corner speed after it (m/s)", run.speed[550],
      math.sqrt(1.5 * GRAVITY * 20.0), 0.2)

# With drag, the air slows the car as well as the tyres. Take one step
# early in the braking zone, where the car is still quick.
draggy = sim.replace(stopper, cda=1.5)
run = sim.simulate(to_a_corner, draggy, initial_speed=90.0)
at = np.flatnonzero(run.limit == 'brake')[5]
one_step = (run.speed[at] ** 2 - run.speed[at + 1] ** 2) / 2.0
between = 0.5 * (run.speed[at] + run.speed[at + 1])
close("braking with drag: the tyres plus the air (m/s2)", one_step,
      1.5 * GRAVITY + draggy.drag(between) / draggy.mass, 0.02)

# Brakes that give out before the tyres do
run = sim.simulate(to_a_corner, sim.replace(stopper, brake_limit=0.8),
                   initial_speed=90.0)
close("weak brakes hold the deceleration to their own limit (m/s2)",
      slowing(run, 5, -40), 0.8 * GRAVITY, 0.02)

# ---------------------------------------------------------------------------
print("3. Rules every lap has to obey (F1 car, made-up circuit)")
# ---------------------------------------------------------------------------

track = circuit()
car = F1_2024
lap = sim.simulate(track, car, initial_speed=70.0)

# Work out the force at the tyres on every step, from the speeds alone
speed = lap.speed
between = 0.5 * (speed[1:] + speed[:-1])
bend = 0.5 * (track.curvature[1:] + track.curvature[:-1])
gaining = (speed[1:] ** 2 - speed[:-1] ** 2) / (2 * track.step)
grip = car.grip_force(between)
sideways = car.mass * between ** 2 * bend
along = car.mass * gaining + car.drag(between)   # + driving, - braking
driving = np.maximum(along, 0.0)

most_grip = float((np.hypot(sideways, along) / grip).max())
most_power = float((driving * between / car.power).max())
most_force = float((driving / car.max_tractive_force).max())
check("never more grip than the tyres have", most_grip < 1.02,
      f"most used {most_grip:.3f} of it")
check("never more power than the engine has", most_power < 1.02,
      f"most used {most_power:.3f} of it")
check("never more force than the cap at low speed", most_force < 1.02,
      f"most used {most_force:.3f} of it")

# The same lap whatever the spacing of the points
coarse = sim.simulate(circuit(1.0), car, initial_speed=70.0).lap_time
fine = sim.simulate(circuit(0.25), car, initial_speed=70.0).lap_time
check("the lap time does not depend on the spacing",
      abs(coarse - fine) < 0.05,
      f"{coarse:.3f}s at 1m, {fine:.3f}s at 0.25m")

# Anything that should help does help
better = [('5% more grip', dict(mu=car.mu * 1.05)),
          ('5% more power', dict(power=car.power * 1.05)),
          ('5% less mass', dict(mass=car.mass * 0.95)),
          ('5% less drag', dict(cda=car.cda * 0.95)),
          ('5% more downforce', dict(cla=car.cla * 1.05)),
          ('10% more drive', dict(drive_fraction=car.drive_fraction * 1.1))]
for label, change in better:
    changed = sim.simulate(track, sim.replace(car, **change),
                           initial_speed=70.0).lap_time
    check(f"{label} makes the lap quicker", changed < lap.lap_time,
          f"{changed - lap.lap_time:+.3f}s")

# ---------------------------------------------------------------------------
print("4. A flying lap joins up with itself")
# ---------------------------------------------------------------------------

# A lap with a hairpin 40m after the line. Run three in a row: the
# middle one is a true flying lap, and periodic=True should match it.
distance = np.arange(0, 3000.0, 1.0)
curvature = np.zeros_like(distance)
curvature[40:90] = 1 / 20.0
curvature[1500:1700] = 1 / 80.0
loop = sim.Track(name='loop', distance=distance, curvature=curvature,
                 source='made up')
flying = sim.simulate(loop, car, periodic=True)
three = sim.Track(name='three laps', distance=np.arange(0, 9000.0, 1.0),
                  curvature=np.tile(curvature, 3), source='made up')
middle = sim.simulate(three, car, initial_speed=50.0).speed[3000:6000]
close("lap time against the middle lap of three (s)", flying.lap_time,
      float(np.sum(1.0 / middle)), 0.01)
check("speed all round the lap against that middle lap",
      np.abs(flying.speed - middle).max() < 0.2,
      f"worst {np.abs(flying.speed - middle).max() * 3.6:.2f} km/h")
check("it starts at the speed it finishes at",
      abs(flying.speed[0] - flying.speed[-1]) < 2.0,
      f"{flying.speed[0] * 3.6:.1f} and {flying.speed[-1] * 3.6:.1f} km/h")

# ---------------------------------------------------------------------------
print("5. What the simulator says is limiting the car")
# ---------------------------------------------------------------------------

def long_corner(radius):
    distance = np.arange(0, 1500.0, 1.0)
    curvature = np.zeros_like(distance)
    curvature[300:1300] = 1 / radius
    return sim.Track(name='long corner', distance=distance,
                     curvature=curvature, source='made up')

for radius in (50.0, 150.0):
    run = sim.simulate(long_corner(radius), car, initial_speed=40.0)
    check(f"round a long {radius:.0f}m corner it is the corner",
          np.mean(run.limit[700:1200] == 'corner') > 0.99)
run = sim.simulate(long_corner(250.0), car, initial_speed=40.0)
check("round one it can take flat out it is the engine",
      np.all(run.limit[900:1200] == 'power'))
run = sim.simulate(straight(2000.0), car, initial_speed=40.0)
check("down a straight it is the engine", np.all(run.limit == 'power'))

# ---------------------------------------------------------------------------
print("6. Mistakes are refused with a plain message")
# ---------------------------------------------------------------------------

def refused(make):
    """Does this raise a ValueError? Returns its message, or None."""
    try:
        make()
    except ValueError as error:
        return str(error)
    return None

for label, given in (
        ('uneven spacing', dict(distance=[0, 1, 2.5, 3],
                                curvature=[0, 0, 0, 0])),
        ('only one point', dict(distance=[0], curvature=[0])),
        ('distance going backwards', dict(distance=[3, 2, 1],
                                          curvature=[0, 0, 0]))):
    message = refused(lambda: sim.Track(name='bad', source='made up',
                                        **given))
    check(f"a track with {label}", message is not None, message or "")

flat = sim.Track(name='signed', distance=[0, 1, 2, 3],
                 curvature=[0, -0.025, 0.025, 0], source='made up')
check("left and right turns count the same", np.all(flat.curvature >= 0))

# A reference for the fit checks: the F1 car's own lap of the circuit
real = sim.simulate(track, car, initial_speed=70.0)
lap_of_track = sim.Reference(
    track=sim.Track(name='made-up circuit', distance=track.distance,
                    curvature=track.curvature, source='made up',
                    channels={'speed_kph': real.speed * 3.6}),
    speed_kph=real.speed * 3.6, lap_time=real.lap_time)

for label, names in (
        ("a name that is not a car number", dict(shared=('mu', 'wings'))),
        ("one name without its comma", dict(shared=('mu'))),
        ("grip, both shares and line all at once",
         dict(shared=('mu', 'brake_fraction', 'drive_fraction')))):
    message = refused(lambda: sim.fit_multi([lap_of_track], car,
                                            verbose=False, **names))
    check(f"a fit asked for {label}", message is not None,
          (message or "")[:60] + "...")

# ---------------------------------------------------------------------------
print("7. Kinks in the map are eased, whole corners are not")
# ---------------------------------------------------------------------------

# 250 km/h round a 250m radius is 2.0g. A 50m radius there is 9.8g.
distance = np.arange(0, 400.0)
at_speed = np.full(400, 250.0)
gentle = np.full(400, 0.004)
kinked = gentle.copy()
kinked[190:210] = 0.02          # a 20m kink
kinked[260:380] = 0.02          # and 120m more: a whole corner

def needs_g(track):
    return (at_speed / 3.6) ** 2 * track.curvature / GRAVITY

same, eased, left = sim._eased(
    sim.Track(name='gentle', distance=distance, curvature=gentle,
              source='made up'), at_speed, 6.0)
check("a map that never asks for more than 6g is left as it is",
      eased == "" and left == "" and np.array_equal(same.curvature, gentle))

before = sim.Track(name='kinked', distance=distance, curvature=kinked,
                   source='made up')
after, eased, left = sim._eased(before, at_speed, 6.0)
close("the 20m kink is eased to exactly 6g", needs_g(after)[190:210].max(),
      6.0, 1e-9)
check("the 120m stretch is left alone and reported",
      np.array_equal(after.curvature[260:380], kinked[260:380])
      and '120m at 260m' in left, left[:44] + "...")
check("nothing else is touched, and the map as it came is kept",
      np.array_equal(after.curvature[:190], kinked[:190])
      and np.array_equal(after.channels['map_curvature'], kinked)
      and np.array_equal(before.curvature, kinked), eased)

# ---------------------------------------------------------------------------
print("8. The fit finds a car it was not told about")
# ---------------------------------------------------------------------------

# Two made-up circuits, each with its own wings, and a car that brakes
# with 0.7 of its grip, drives with 0.5 and takes every corner 0.8 times
# as tight as the map. The fit sees only the two speed traces.
second = circuit(corners=[(300, 150, 60), (900, 60, 20), (1300, 250, 110),
                          (2000, 120, 45), (2500, 60, 16), (2900, 300, 200),
                          (3500, 150, 70)], length=3900.0)
truth = [(track, dict(cla=4.0, cda=1.30)), (second, dict(cla=5.5, cda=1.60))]
laps_seen = []
for the_map, wings in truth:
    true_car = sim.replace(car, brake_fraction=0.7, drive_fraction=0.5,
                           **wings)
    driven = sim.simulate(sim.straightened(the_map, 0.8), true_car,
                          initial_speed=70.0)
    laps_seen.append(sim.Reference(
        track=sim.Track(name=the_map.name, distance=the_map.distance,
                        curvature=the_map.curvature, source='made up',
                        channels={'speed_kph': driven.speed * 3.6}),
        speed_kph=driven.speed * 3.6, lap_time=driven.lap_time))

shared, cars, tracks, outcome = sim.fit_multi(laps_seen, car, verbose=False)
close("braking share", shared['brake_fraction'], 0.7, 0.02)
close("drive share", shared['drive_fraction'], 0.5, 0.02)
for found, fitted_track, (the_map, wings) in zip(cars, tracks, truth):
    line = fitted_track.curvature.max() / the_map.curvature.max()
    close(f"downforce at the circuit with {wings['cla']}", found.cla,
          wings['cla'], 0.1)
    close(f"drag at the circuit with {wings['cda']}", found.cda,
          wings['cda'], 0.03)
    close("line there", line, 0.8, 0.02)

# ---------------------------------------------------------------------------
print("9. The table of what a change is worth")
# ---------------------------------------------------------------------------

worth = sim.setup_effects(track, car)
check("downforce, less mass, power and grip gain time, drag loses it",
      worth['10% more downforce'] < 0 and worth['10kg lighter'] < 0
      and worth['5% more power'] < 0 and worth['5% more tyre grip'] < 0
      and worth['10% more drag'] > 0,
      ", ".join(f"{seconds:+.2f}s" for seconds in worth.values()))
by_hand = (sim.simulate(track, sim.replace(car, mass=car.mass - 10.0),
                        periodic=True).lap_time
           - sim.simulate(track, car, periodic=True).lap_time)
close("10kg lighter is the flying lap with it minus the one without (s)",
      worth['10kg lighter'], by_hand, 1e-12)

# ---------------------------------------------------------------------------
print("10. At the size of a Formula Student car")
# ---------------------------------------------------------------------------

# Something like a Formula Student car: 280kg with its driver, 60kW at
# the wheels, modest wings, rear-wheel drive. Not any real car's numbers.
FS_CAR = sim.Car(name='like a Formula Student car', mass=280.0,
                 power=60_000.0, cla=3.0, cda=1.3, mu=1.5, brake_limit=3.0,
                 drive_fraction=0.6, max_tractive_force=3500.0)

# Skidpad: the lane is 3m wide round a circle 15.25m across, so its
# middle is a circle of 9.125m radius. The same balance as section 1.
radius = 15.25 / 2 + 1.5
points = 229
skidpad = sim.Track(name='skidpad',
                    distance=np.arange(points) * 2 * math.pi * radius / points,
                    curvature=np.full(points, 1 / radius), source='made up')
q = 0.5 * AIR_DENSITY * FS_CAR.cla
exact = math.sqrt(FS_CAR.mu * GRAVITY * radius
                  / (1 - FS_CAR.mu * q * radius / FS_CAR.mass))
run = sim.simulate(skidpad, FS_CAR, periodic=True)
close("skidpad lap (s)", run.lap_time, 2 * math.pi * radius / exact, 0.01)

# A hairpin of 4.5m radius between two 30m straights. Corners this tight
# need points closer together than the 1m used for F1 circuits.
def hairpin_time(spacing):
    distance = np.arange(0, 60 + math.pi * 4.5, spacing)
    turning = (distance >= 30) & (distance < 30 + math.pi * 4.5)
    hairpin = sim.Track(name='hairpin', distance=distance,
                        curvature=np.where(turning, 1 / 4.5, 0.0),
                        source='made up')
    return sim.simulate(hairpin, FS_CAR, initial_speed=15.0).lap_time

check("a tight hairpin is the same at 0.25m spacing as at 0.1m",
      abs(hairpin_time(0.25) - hairpin_time(0.1)) < 0.005,
      f"{hairpin_time(0.25):.3f}s and {hairpin_time(0.1):.3f}s; "
      f"at 1m it is {hairpin_time(1.0):.3f}s")

# The acceleration event: 75m from a standing start. The exact answer
# comes from stepping through the same forces in tiny steps of time.
def standing_start(car, length, tick=1e-4):
    speed = covered = seconds = 0.0
    while covered < length:
        push = min(car.power / max(speed, 1e-9), car.max_tractive_force,
                   car.grip_force(speed) * car.drive_fraction)
        speed += (push - car.drag(speed)) / car.mass * tick
        covered += speed * tick
        seconds += tick
    return seconds, speed

exact_time, exact_speed = standing_start(FS_CAR, 75.0)
distance = np.arange(0, 75.0 + 0.125, 0.25)
run = sim.simulate(sim.Track(name='75m', distance=distance,
                             curvature=np.zeros_like(distance),
                             source='made up'), FS_CAR, initial_speed=0.0)
close("75m from rest, speed at the line (m/s)", run.speed[-1], exact_speed,
      0.02)

# From rest the time has to be added up step by step, each step at the
# average of the speeds at its two ends. lap_time is not that: it is for
# a lap already at speed, and from rest it comes out wrong.
by_steps = float(np.sum(2 * 0.25 / (run.speed[1:] + run.speed[:-1])))
close("75m from rest, time added up step by step (s)", by_steps,
      exact_time, 0.005)

# ---------------------------------------------------------------------------

print()
if failed:
    print(f"{len(failed)} of {count} checks FAILED:")
    for label in failed:
        print(f"  {label}")
    sys.exit(1)
print(f"ALL OK ({count} checks)")
