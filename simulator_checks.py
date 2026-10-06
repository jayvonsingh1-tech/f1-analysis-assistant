"""Checks for Simulator.py. Run them with:  python simulator_checks.py

Each check compares the simulator with an answer known some other way:
a formula from a physics textbook, a rule every lap has to obey, or the
simulator itself run a different way. None of them needs any F1 data,
so the whole file runs in well under a minute.

Run it after every change to Simulator.py. A line that says BAD means
the change broke something that used to work. The last line says ALL OK,
or how many checks failed.

Section 13 checks the part that reads FastF1's data, build_reference().
It makes up a race with a known car, writes that car's laps out the way
FastF1 records them, and compares what build_reference() makes of them
with the truth. It needs pandas, which FastF1 needs too.

The checks were themselves checked: the simulator was broken on
purpose, one place at a time, to see whether a line here says BAD. For
sections 1 to 12 that was 69 places, and every one was caught. For the
part that reads FastF1's data it was about 100 places. About a dozen of
those change nothing that can be measured on the made-up race (each is
a second safeguard behind a first one, or a refinement too small to
see), and the rest were caught.

The last section runs the simulator at the size of a Formula Student
car and its events, to show the physics holds there too.
"""

import math
import sys
import types

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
distance_600 = distance             # kept for section 12
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
# What holds the car back without being asked to: the air, and the drag
# that comes with cornering (see section 12)
share = np.minimum(sideways / grip, 1.0)
held_back = car.drag(between) + car.corner_drag * share ** 2 * grip
along = car.mass * gaining + held_back   # + driving, - braking
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
         dict(shared=('mu', 'brake_fraction', 'drive_fraction'))),
        ("power both shared and per circuit",
         dict(per_circuit=('cla', 'cda', 'line', 'power')))):
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

# Two made-up circuits. At each one the car has its own wings, weighs
# something different and runs in different air. It brakes with 0.7 of
# its grip, drives with 0.5 and takes every corner 0.8 times as tight as
# the map. The fit is told the weight and the air, as it is for a real
# lap, and sees the two speed traces. The rest it has to find. (The
# points are 2m apart here, which halves the time the fits take.)
first = circuit(2.0)
second = circuit(2.0, corners=[(300, 150, 60), (900, 60, 20),
                               (1300, 250, 110), (2000, 120, 45),
                               (2500, 60, 16), (2900, 300, 200),
                               (3500, 150, 70)], length=3900.0)
truth = [(first, dict(cla=4.0, cda=1.30), dict(mass=790.0, air_density=1.15)),
         (second, dict(cla=5.5, cda=1.60), dict(mass=800.0, air_density=1.19))]

def laps_seen(powers):
    """One lap of each circuit by the true car, with this power at
    each, as references for the fit."""
    seen = []
    for (the_map, wings, day), power in zip(truth, powers):
        true_car = sim.replace(car, brake_fraction=0.7, drive_fraction=0.5,
                               power=power, **wings, **day)
        driven = sim.simulate(sim.straightened(the_map, 0.8), true_car,
                              initial_speed=70.0)
        seen.append(sim.Reference(
            track=sim.Track(name=the_map.name, distance=the_map.distance,
                            curvature=the_map.curvature, source='made up',
                            channels={'speed_kph': driven.speed * 3.6}),
            speed_kph=driven.speed * 3.6, lap_time=driven.lap_time, **day))
    return seen

# The same 640kW at both circuits, as laps from one season have
shared, cars, tracks, outcome = sim.fit_multi(laps_seen([640e3, 640e3]), car,
                                              verbose=False)
outcome_same = outcome              # kept for section 12
close("braking share", shared['brake_fraction'], 0.7, 0.02)
close("drive share", shared['drive_fraction'], 0.5, 0.02)
close("power (kW)", shared['power'] / 1000, 640.0, 10.0)
for found, fitted_track, (the_map, wings, day) in zip(cars, tracks, truth):
    line = fitted_track.curvature.max() / the_map.curvature.max()
    check(f"it runs the {day['mass']:.0f}kg car in the air it was told",
          found.mass == day['mass']
          and found.air_density == day['air_density'])
    close(f"downforce at the circuit with {wings['cla']}", found.cla,
          wings['cla'], 0.1)
    close(f"drag at the circuit with {wings['cda']}", found.cda,
          wings['cda'], 0.03)
    close("line there", line, 0.8, 0.02)

# A different power at each circuit, as laps from different seasons
# have, with the fit asked for a power for each
shared, cars, tracks, outcome = sim.fit_multi(
    laps_seen([640e3, 700e3]), car,
    shared=('brake_fraction', 'drive_fraction'),
    per_circuit=('cla', 'cda', 'line', 'power'), verbose=False)
for found, power in zip(cars, (640.0, 700.0)):
    close(f"a power for each circuit: the one with {power:.0f}kW",
          found.power / 1000, power, 10.0)

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
print("10. Weight, air and power")
# ---------------------------------------------------------------------------

# The standard day at sea level, 15C and 1013.25 millibar, is 1.225 kg/m3
close("air at 15C and 1013.25mbar (kg/m3)", sim.density_of_air(15.0, 1013.25),
      1.225, 0.0005)

# Damp air, against the weather forecasters' formula: (pressure - 0.378 x
# vapour pressure) / (287.05 x kelvin). Steam tables give water's vapour
# pressure at 30C as 4246 pascals, and the air here holds 80% of that.
exact = (100_000.0 - 0.378 * 0.8 * 4246.0) / (287.05 * 303.15)
close("air at 30C, 1000mbar and 80% humidity (kg/m3)",
      sim.density_of_air(30.0, 1000.0, 80.0), exact, 0.0005)

# Downforce and drag are in proportion to the air's density
thin = sim.replace(car, air_density=AIR_DENSITY / 2)
close("half the air gives half the downforce",
      thin.downforce(80.0) / car.downforce(80.0), 0.5, 1e-12)
close("and half the drag", thin.drag(80.0) / car.drag(80.0), 0.5, 1e-12)
close("so the top speed is higher by the cube root of 2",
      sim._terminal_speed(thin) / sim._terminal_speed(car), 2 ** (1 / 3),
      1e-12)
half_wings = sim.replace(car, cla=car.cla / 2, cda=car.cda / 2)
close("a lap in half the air is a lap with half the wings (s)",
      sim.simulate(track, thin, periodic=True).lap_time,
      sim.simulate(track, half_wings, periodic=True).lap_time, 1e-9)

# Reading the air off a lap's weather. This stands in for a FastF1 lap.
class LapWith:
    def __init__(self, **weather):
        self.weather = weather

    def get_weather_data(self):
        return self.weather

density, said = sim._air_for(LapWith(AirTemp=25.0, Pressure=780.0,
                                     Humidity=30.0))
check("the thin air of Mexico City is accepted",
      density is not None and 0.88 < density < 0.93, said)
density, said = sim._air_for(LapWith(AirTemp=25.0, Pressure=78.0,
                                     Humidity=30.0))
check("a pressure ten times too low is not", density is None, said)
density, said = sim._air_for(LapWith())
check("a lap with no weather gives no density", density is None, said)
density, said = sim._air_for(LapWith(AirTemp=20.0, Pressure=1000.0,
                                     Humidity=float('nan')))
check("a missing humidity is done without",
      density is not None
      and abs(density - sim.density_of_air(20.0, 1000.0)) < 1e-12, said)

# Weight: the least the rules allow plus the fuel still on board
close("2024, first lap of 50: 798kg and all 100kg of fuel",
      sim.race_weight(2024, 1, 50), 898.0, 1e-9)
close("2024, last lap of 50: 2kg of fuel left",
      sim.race_weight(2024, 50, 50), 800.0, 1e-9)
close("2020, lap 34 of 53: 746kg and 20/53 of the fuel",
      sim.race_weight(2020, 34, 53), 746.0 + 100.0 * 20 / 53, 1e-9)
close("2026, first lap of 50: 768kg and the 70kg those cars start with",
      sim.race_weight(2026, 1, 50), 838.0, 1e-9)
check("a season with no weight on record gives none",
      sim.race_weight(1990, 10, 50) is None)

# Why the weight has to be told and cannot be fitted: a car 10% bigger
# in every way laps exactly the same. Every force on it is 10% bigger,
# and so is the mass those forces have to move.
bigger = sim.replace(car, mass=car.mass * 1.1, power=car.power * 1.1,
                     cla=car.cla * 1.1, cda=car.cda * 1.1,
                     max_tractive_force=car.max_tractive_force * 1.1,
                     reference_load=car.reference_load * 1.1)
close("a car 10% bigger in every way laps the same (s)",
      sim.simulate(track, bigger, periodic=True).lap_time,
      sim.simulate(track, car, periodic=True).lap_time, 1e-9)
worth_bigger = sim.setup_effects(track, bigger)
check("and its table is the same, but for 10kg being a smaller share of it",
      all(abs(worth_bigger[label] - worth[label]) < 1e-9
          for label in worth if label != '10kg lighter')
      and abs(worth_bigger['10kg lighter'] / worth['10kg lighter']
              - 1 / 1.1) < 0.01,
      f"10kg lighter: {worth_bigger['10kg lighter']:+.3f}s, was "
      f"{worth['10kg lighter']:+.3f}s")

# ---------------------------------------------------------------------------
print("11. Hills")
# ---------------------------------------------------------------------------

def hill(slope, length=3000.0):
    """A straight on one steady slope: 0.05 climbs 1m in every 20."""
    distance = np.arange(0, length, 1.0)
    return sim.Track(name='hill', distance=distance,
                     curvature=np.zeros_like(distance), source='made up',
                     slope=np.full_like(distance, slope))

level = sim.simulate(straight(3000.0), car, initial_speed=40.0)
check("a slope of nothing is a flat track",
      np.array_equal(sim.simulate(hill(0.0), car, initial_speed=40.0).speed,
                     level.speed))

# With next to no power and no drag a car is a ball on a slope: it
# gains or loses speed only by the height it drops or climbs.
# v^2 = v0^2 - 2 g h
ball = sim.replace(NO_WINGS, power=1e-3, cda=1e-12)
for slope, way in ((-0.08, "down"), (0.03, "up")):
    run = sim.simulate(hill(slope, 500.0), ball, initial_speed=30.0)
    exact = math.sqrt(30.0 ** 2 - 2 * GRAVITY * slope * 499.0)
    close(f"coasting {way} a slope of {abs(slope):.0%} for 499m (m/s)",
          run.speed[-1], exact, 1e-6)

# The same on a slope that changes: flat for 100m, then a climb that
# is 1 in 10 from the next point on. At every point the ball's speed
# has to match the height it has climbed so far.
slope = np.where(np.arange(300) < 100, 0.0, 0.1)
climbed = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]))])
run = sim.simulate(sim.replace(hill(0.0, 300.0), slope=slope), ball,
                   initial_speed=30.0)
worst = float(np.abs(run.speed ** 2
                     - (30.0 ** 2 - 2 * GRAVITY * climbed)).max())
check("where the slope changes, its speed follows the height point by "
      "point", worst < 1e-3, f"speed squared within {worst:.1e}")

# Top speed is where the power is all used: on drag, and on lifting the
# car up the hill (or with the hill's help, going down)
for slope in (0.05, -0.05, 0.15):
    top = sim._terminal_speed(car, slope)
    used = (car.drag(top) + car.mass * GRAVITY * slope) * top
    close(f"top speed on a slope of {slope:+.0%}: power used over power",
          used / car.power, 1.0, 1e-9)
    run = sim.simulate(hill(slope, 8000.0), car, initial_speed=40.0)
    close("and the car reaches it after 8km (m/s)", run.speed[-1], top, 0.05)
run = sim.simulate(hill(-0.05, 8000.0), car, initial_speed=40.0)
check("downhill it goes faster than its top speed on the level",
      run.speed[-1] > sim._terminal_speed(car) + 1.0,
      f"{run.speed[-1] * 3.6:.0f} against "
      f"{sim._terminal_speed(car) * 3.6:.0f} km/h")

# Braking for the corner of section 2, on a hill. No wings and next to
# no drag, so the car slows at mu g, plus g x the slope going up and
# less g x the slope going down.
for slope, way in ((0.06, "uphill"), (-0.06, "downhill")):
    on_a_hill = sim.replace(to_a_corner, slope=np.full(600, slope))
    run = sim.simulate(on_a_hill, stopper, initial_speed=90.0)
    braking = np.flatnonzero(run.limit == 'brake')
    a, b = braking[5], braking[-10]
    lost = (run.speed[a] ** 2 - run.speed[b] ** 2) / (2.0 * (b - a))
    close(f"braking {way}, deceleration (m/s2)", lost,
          (1.5 + slope) * GRAVITY, 0.02)

# Braking again, on a road that is flat and then climbs 1 in 10 from
# 400m on, part-way through the braking zone. Each metre of braking
# takes off mu g plus g x the slope half-way along that metre.
slope = np.where(np.arange(600) < 400, 0.0, 0.1)
run = sim.simulate(sim.replace(to_a_corner, slope=slope), stopper,
                   initial_speed=90.0)
braking = np.flatnonzero(run.limit == 'brake')[5:-10]
lost = (run.speed[braking] ** 2 - run.speed[braking + 1] ** 2) / 2.0
should = (1.5 + 0.5 * (slope[braking] + slope[braking + 1])) * GRAVITY
worst = float(np.abs(lost - should).max())
check("braking across a change of slope, metre by metre",
      braking[0] < 390 and braking[-1] > 410 and worst < 0.02,
      f"within {worst:.3f} m/s2 on every metre")

# The loop of section 4, now climbing 80m and coming back down. Three
# laps in a row again: the middle one is a true flying lap.
height = 40.0 * np.sin(2 * math.pi * np.arange(3000) / 3000.0)
hilly = sim.replace(loop, slope=np.gradient(height))
flying = sim.simulate(hilly, car, periodic=True)
three = sim.Track(name='three hilly laps', distance=np.arange(0, 9000.0, 1.0),
                  curvature=np.tile(hilly.curvature, 3), source='made up',
                  slope=np.tile(hilly.slope, 3))
middle = sim.simulate(three, car, initial_speed=50.0).speed[3000:6000]
close("a hilly flying lap against the middle lap of three (s)",
      flying.lap_time, float(np.sum(1.0 / middle)), 0.01)
# The car is braking for the hairpin as it crosses the line, on an 8%
# climb, so the one step of braking that joins the lap up is on a slope
worst = float(np.abs(flying.speed - middle)[-100:].max())
check("braking across the line on a climb, the last 100m match that "
      "middle lap", worst < 0.015, f"within {worst:.4f} m/s")
on_the_flat = sim.simulate(loop, car, periodic=True)
moved = float(np.abs(flying.speed - on_the_flat.speed).max())
check("and the hills change its speed", moved > 1.0,
      f"by up to {moved * 3.6:.1f} km/h, and the lap by "
      f"{flying.lap_time - on_the_flat.lap_time:+.3f}s")

# That loop is braking as it crosses the line. This one crosses it flat
# out on the steepest part of the climb, so the one step that joins the
# end of the lap to its start is on a slope too.
curvature = np.zeros(3000)
curvature[1000:1050] = 1 / 20.0
curvature[2200:2400] = 1 / 80.0
climbing = sim.Track(name='line on a climb', distance=np.arange(0, 3000.0),
                     curvature=curvature, source='made up',
                     slope=np.gradient(height))
flying = sim.simulate(climbing, car, periodic=True)
three = sim.Track(name='three laps', distance=np.arange(0, 9000.0, 1.0),
                  curvature=np.tile(curvature, 3), source='made up',
                  slope=np.tile(climbing.slope, 3))
middle = sim.simulate(three, car, initial_speed=50.0).speed[3000:6000]
worst = float(np.abs(flying.speed - middle)[:500].max())
check("crossing the line on a climb, the first 500m match that middle lap",
      worst < 0.002, f"within {worst:.5f} m/s")

message = refused(lambda: sim.Track(name='bad', distance=[0, 1, 2],
                                    curvature=[0, 0, 0], source='made up',
                                    slope=[0.0, 0.1]))
check("a track with a slope for only some of its points",
      message is not None, message or "")

# Heights in the position data become the slope. A made-up circuit: a
# circle 2km round that rises and falls 10m either side of its middle
# height, twice a lap. Six laps of positions as FastF1 gives them: a
# sample every 15m or so, in tenths of a metre, each lap sampled at
# different places.
round_trip = 2000.0
def on_circle(along):
    angle = 2 * math.pi * along / round_trip
    radius = round_trip / (2 * math.pi)
    return (radius * np.cos(angle), radius * np.sin(angle),
            10.0 * np.sin(2 * angle))

positions = []
for lap_number in range(6):
    along = np.arange(2.5 * lap_number, round_trip, 15.0)
    positions.append(tuple(np.round(10 * np.asarray(values))
                           for values in on_circle(along)))
along = np.arange(0.0, round_trip, 12.0)
x, y, _ = on_circle(along)
speed_line = (10 * x, 10 * y, np.full(len(along), 200.0))
built = sim.track_from_laps(positions, speed_line=speed_line)
true_slope = (10.0 * 4 * math.pi / round_trip
              * np.cos(4 * math.pi * built.distance / round_trip))
worst = float(np.abs(built.slope - true_slope)[50:-50].max())
check("heights in the position data: the slope comes out right",
      worst < 0.004, f"within {worst:.2%}, on slopes of up to "
      f"{np.abs(true_slope).max():.1%}")
close("and the height from lowest to highest (m)",
      np.ptp(built.channels['z']), 20.0, 0.2)
without = sim.track_from_laps([lap[:2] for lap in positions],
                              speed_line=speed_line)
check("no heights in the position data: a flat track",
      not without.slope.any() and 'z' not in without.channels)
some = sim.track_from_laps(positions[:5] + [positions[5][:2]],
                           speed_line=speed_line)
check("heights on some laps and not others: a flat track too",
      not some.slope.any() and 'z' not in some.channels)

# Heights that cannot be right are not believed
same, said = sim._hills(built)
check("believable heights are kept, and described",
      same is built and said.startswith('20m from the lowest'), said)
flat, said = sim._hills(without)
check("no heights: said so", 'no heights' in said, said)
level = sim.replace(built, slope=np.zeros(len(built.distance)),
                    channels=dict(built.channels,
                                  z=np.full(len(built.distance), 85.2)))
flat, said = sim._hills(level)
check("heights that are all the same: a flat track, said so",
      flat is level and 'a flat track' in said, said)
steep = sim.replace(built, slope=built.slope * 5)
flat, said = sim._hills(steep)
check("a slope steeper than 1 in 4 is not believed",
      not flat.slope.any() and 'cannot be right' in said, said)
channels = dict(built.channels, z=built.channels['z']
                + np.linspace(0.0, 20.0, len(built.distance)))
broken = sim.replace(built, channels=channels)
flat, said = sim._hills(broken)
check("nor a lap that ends 20m higher than it began",
      not flat.slope.any() and 'cannot be right' in said, said)

# ---------------------------------------------------------------------------
print("12. The drag that comes with cornering")
# ---------------------------------------------------------------------------

# A tyre gripping sideways drags: with all its grip in use, by
# corner_drag x that grip, and with a share of it in use, by corner_drag
# x the share squared x the grip. The same car with it and without it:
DRAGGY = sim.replace(NO_WINGS, corner_drag=0.07)

check("a car made without a corner drag has none",
      NO_WINGS.corner_drag == 0.0 and F1_2024.corner_drag > 0)
check("on a straight it changes nothing",
      np.array_equal(
          sim.simulate(straight(2000.0), DRAGGY, initial_speed=40.0).speed,
          sim.simulate(straight(2000.0), NO_WINGS, initial_speed=40.0).speed))

# Coasting round a circle with next to no power and no air to push
# through, only the corner drag slows the car. With no wings the grip is
# mu m g, the corner uses v^2 / (R mu g) of it, and the drag works out
# at corner_drag x m v^4 / (R^2 mu g). That gives
# 1 / v^2 = 1 / v0^2 + 2 x corner_drag x distance / (R^2 mu g)
coaster = sim.replace(DRAGGY, power=1e-3, cda=1e-12)
for radius in (200.0, 400.0):
    run = sim.simulate(circle(radius, 600.0), coaster, initial_speed=40.0)
    exact = (1 / 40.0 ** 2 + 2 * 0.07 * 599.0
             / (radius ** 2 * 1.5 * GRAVITY)) ** -0.5
    close(f"coasting 599m round a {radius:.0f}m circle from 40 m/s (m/s)",
          run.speed[-1], exact, 1e-4)

# Flat out round a big circle, the car settles at the speed where its
# power is all used: on the air, and on the corner drag
def used_up(the_car, speed, radius):
    """Power the air and the corner drag take at a steady speed, as a
    share of what the car has."""
    grip = the_car.grip_force(speed)
    share = the_car.mass * speed ** 2 / radius / grip
    return ((the_car.drag(speed) + the_car.corner_drag * share ** 2 * grip)
            * speed / the_car.power)

run = sim.simulate(circle(600.0, 12000.0), DRAGGY, initial_speed=40.0)
close("flat out round a 600m circle: power used over power, at the end",
      used_up(DRAGGY, run.speed[-1], 600.0), 1.0, 1e-3)
without = sim.simulate(circle(600.0, 12000.0), NO_WINGS, initial_speed=40.0)
check("and that is slower than the same car with no corner drag",
      run.speed[-1] < without.speed[-1] - 0.5,
      f"{run.speed[-1] * 3.6:.1f} against {without.speed[-1] * 3.6:.1f} km/h")
run = sim.simulate(circle(250.0, 9000.0), F1_2024, initial_speed=40.0)
close("the F1 car flat out round a 250m circle: power used over power",
      used_up(F1_2024, run.speed[-1], 250.0), 1.0, 1e-3)

# Braking while turning: a gentle 300m bend into the tight corner of
# section 2. Each metre takes off what the tyres have left for braking
# (the grip not being used to turn) plus the corner drag.
bending = sim.Track(name='bend then corner', distance=distance_600,
                    curvature=np.where(distance_600 < 500, 1 / 300.0,
                                       1 / 20.0), source='made up')
turner = sim.replace(stopper, corner_drag=0.07)
run = sim.simulate(bending, turner, initial_speed=60.0)
at = np.flatnonzero(run.limit == 'brake')[5]
between = 0.5 * (run.speed[at] + run.speed[at + 1])
share = between ** 2 / 300.0 / (1.5 * GRAVITY)
exact = 1.5 * GRAVITY * (math.sqrt(1 - share ** 2) + 0.07 * share ** 2)
close("braking in a bend: the grip left over plus the corner drag (m/s2)",
      (run.speed[at] ** 2 - run.speed[at + 1] ** 2) / 2.0, exact, 0.02)
check("and the corner drag is part of it",
      0.07 * share ** 2 * 1.5 * GRAVITY > 0.05,
      f"{0.07 * share ** 2 * 1.5 * GRAVITY:.2f} of {exact:.2f} m/s2")

# On a whole lap it can only cost time, and more of it costs more
laps_with = [sim.simulate(track, sim.replace(car, corner_drag=amount),
                          periodic=True).lap_time
             for amount in (0.0, 0.035, 0.07, 0.14)]
check("more corner drag, slower lap",
      all(later > earlier + 0.01
          for earlier, later in zip(laps_with, laps_with[1:])),
      ", ".join(f"{seconds:.2f}s" for seconds in laps_with))

# The fit holds the corner drag at what the base car has. The laps of
# section 8 were driven by a car with the base car's 0.07.
check("a fit keeps the corner drag the base car has",
      all(found.corner_drag == car.corner_drag for found in cars))
# Told there is none, it has to explain the same laps another way, and
# cannot do it as well.
shared_none, cars_none, tracks_none, outcome_none = sim.fit_multi(
    laps_seen([640e3, 640e3]), sim.replace(car, corner_drag=0.0),
    verbose=False)
check("fitted without it, laps driven with it fit worse",
      outcome_none.cost > 10 * outcome_same.cost + 1e-9,
      f"squared error {outcome_none.cost:.2e} against "
      f"{outcome_same.cost:.2e}")
# Asked to, the fit can find the corner drag as well, on laps as clean
# as these. On real laps it cannot be trusted to (see CORNER DRAG in the
# notes at the top of Simulator.py), which is why it is held.
shared_asked, cars_asked, tracks_asked, outcome_asked = sim.fit_multi(
    laps_seen([640e3, 640e3]), sim.replace(car, corner_drag=0.03),
    shared=('brake_fraction', 'drive_fraction', 'power', 'corner_drag'),
    verbose=False)
close("asked for it, the fit finds the corner drag on clean laps",
      shared_asked['corner_drag'], 0.07, 0.005)

# ---------------------------------------------------------------------------
print("13. A made-up race, read the way a real one is")
# ---------------------------------------------------------------------------

# build_reference() turns the raw samples of a race into a lap the fit
# can use. It cannot be checked on a real race, where nobody knows the
# true car. So this section makes a race up: a circuit and a car that
# are known exactly, and that car's laps written out the way FastF1
# records them. Positions come about 4.5 times a second, in tenths of a
# metre. Speed comes about 4 times a second, in whole km/h, stamped by
# a clock of its own that runs late. build_reference() is then handed
# the race as if it were a real one, and what it makes of it is
# compared with the truth.

# Each bend of the made-up circuit, in the order they are driven: the
# straight before it in metres, the angle it turns through in degrees
# (+ to the left, - to the right) and its radius in metres. The two odd
# lengths are what it takes for the lap to end where it began.
BENDS = [(300.0, 90, 60), (150.0, -45, 30), (60.0, 45, 40),
         (100.0, 90, 100), (782.24, 120, 20), (100.0, -30, 200),
         (134.96, 90, 150)]
HOME_STRAIGHT = 250.0   # from the last bend back to the line, metres
FINE = 0.25             # metres between the points the circuit is drawn at

def made_up_circuit():
    """The circuit at every FINE metres: how far round each point is,
    the bend there (1 / radius, + to the left and - to the right), the
    way the track points, and where the point is (x, y and height).

    Each bend tightens over 20m and opens out over 20m, or over less if
    it is a short one. The lap climbs and drops 30m."""
    length = HOME_STRAIGHT + sum(
        straight + math.radians(abs(angle)) * radius
        for straight, angle, radius in BENDS)
    along = np.linspace(0.0, length, int(round(length / FINE)) + 1)
    bend = np.zeros_like(along)
    start = 0.0
    for straight, angle, radius in BENDS:
        start += straight
        how_long = math.radians(abs(angle)) * radius
        ramp = min(20.0, 0.4 * how_long)
        bend += math.copysign(1 / radius, angle) * np.interp(
            along, [start - ramp / 2, start + ramp / 2,
                    start + how_long - ramp / 2, start + how_long + ramp / 2],
            [0, 1, 1, 0])
        start += how_long

    # Add up the bends for the way the track points, and that for where
    # each point is
    step = np.diff(along)
    heading = np.concatenate(
        [[0.0], np.cumsum(0.5 * (bend[1:] + bend[:-1]) * step)])
    halfway = 0.5 * (heading[1:] + heading[:-1])
    # The line is put at x = 1200m, y = -800m, so that x = y = 0 is
    # nowhere near the track
    x = 1200.0 + np.concatenate([[0.0], np.cumsum(np.cos(halfway) * step)])
    y = -800.0 + np.concatenate([[0.0], np.cumsum(np.sin(halfway) * step)])
    height = 15.0 * np.sin(2 * math.pi * along / length)
    return along, bend, heading, x, y, height

along, bend, heading, map_x, map_y, height = made_up_circuit()
LAP = float(along[-1])
true_slope = np.gradient(height, along)

# The made-up car and its quickest lap, which is lap 30 of a 40 lap
# race in 2024 on a warm day. It brakes with 0.7 of its grip, drives
# with 0.5, and takes every bend 0.8 times as tight as the map.
RACE_LAPS, QUICKEST = 40, 30
WEATHER = dict(AirTemp=25.0, Pressure=1000.0, Humidity=40.0)
TRUE = dict(power=640e3, cla=4.5, cda=1.40, brake_fraction=0.7,
            drive_fraction=0.5)
TRUE_LINE = 0.8
true_car = sim.replace(
    F1_2024, mass=sim.race_weight(2024, QUICKEST, RACE_LAPS),
    air_density=sim.density_of_air(25.0, 1000.0, 40.0), **TRUE)

points = int(round(LAP))
metres = np.linspace(0.0, LAP, points, endpoint=False)
true_map = sim.Track(name='made-up race', distance=metres,
                     curvature=np.interp(metres, along, bend),
                     source='made up',
                     slope=np.interp(metres, along, true_slope))
flying = sim.simulate(sim.straightened(true_map, TRUE_LINE), true_car,
                      periodic=True)

# Its speed in m/s at every point of the circuit, and the time it has
# taken to get there from the line
true_speed = np.interp(along, metres, flying.speed, period=LAP)
clock = np.concatenate(
    [[0.0], np.cumsum(np.diff(along)
                      / (0.5 * (true_speed[1:] + true_speed[:-1])))])
LAP_TIME = float(clock[-1])

LATE = 0.085    # seconds the speed is stamped later than the positions

# Where the timing loops are, in metres round the lap: a dozen of them,
# none in a slow corner
LOOPS = np.array([0.0, 250.0, 480.0, 720.0, 950.0, 1150.0, 1350.0, 1550.0,
                  1800.0, 2000.0, 2250.0, 2450.0])

# What real data does wrong, each in a known amount (see WHERE THE CAR
# IS and GEOMETRY in the notes at the top of Simulator.py).
# made_up_stint() says what each of these is.
ROUGH = ('loops', 'slip', 'stamps', 'map faults', 'strays', 'off track',
         'zeros', 'position hole', 'short lap')

def seconds(column):
    """A column of FastF1 times as plain numbers of seconds."""
    return column.dt.total_seconds().to_numpy()

class MadeUpLap(dict):
    """One lap as FastF1 hands it over: its number, driver and time,
    and the samples recorded during it.

    What build_reference() asks a lap for is here under FastF1's own
    names: 'LapNumber', 'Driver', 'LapTime' and the three get_...()
    calls. The rest is kept under names of its own, with a _ in front:
      _positions  this lap's position samples
      _car        the speed samples of the driver's whole stint
      _rows       which rows of those are this lap's (first, one past
                  the last)
      _weather    the weather on the lap, or None
      _pits       whether the lap starts or ends in the pits
      _unreadable whether asking for its speed samples fails
    """

    def get_pos_data(self):
        return self['_positions'].copy()

    def get_car_data(self, pad=0, pad_side='both'):
        # pad asks for that many samples from the laps either side too
        if self['_unreadable']:
            raise KeyError("no car data")
        first, last = self['_rows']
        return self['_car'].iloc[max(first - pad, 0):last + pad].copy()

    def get_weather_data(self):
        return self['_weather']

class MadeUpLaps:
    """A race's laps, and the ways of picking from them that
    build_reference() uses."""

    def __init__(self, laps):
        self.laps = list(laps)

    def pick_drivers(self, driver):
        return MadeUpLaps(lap for lap in self.laps
                          if lap['Driver'] == driver)

    def pick_quicklaps(self):
        # FastF1 keeps the laps within 107% of the quickest
        if not self.laps:
            return self
        most = 1.07 * min(lap['LapTime'] for lap in self.laps)
        return MadeUpLaps(lap for lap in self.laps if lap['LapTime'] < most)

    def pick_wo_box(self):
        # ... and the laps that neither start nor end in the pits
        return MadeUpLaps(lap for lap in self.laps if not lap['_pits'])

    def pick_fastest(self):
        if not self.laps:
            return None
        return min(self.laps, key=lambda lap: lap['LapTime'])

    def iterlaps(self):
        return enumerate(self.laps)

    def __getitem__(self, column):
        # laps['LapNumber'] is every lap's number, as it is in FastF1
        return np.array([lap[column] for lap in self.laps])

def bump(place, length):
    """A smooth bump along the circuit: 1 at `place`, falling to 0 half
    of `length` either side of it."""
    reach = (along - place) / length
    return np.where(np.abs(reach) < 0.5, np.cos(math.pi * reach) ** 2, 0.0)

def made_up_stint(driver, numbers, slower, pits, begins, rng, faults=(),
                  heights=True, weather=WEATHER, drs=(), late=LATE,
                  drift=0.0, reads=1.0, unreadable=()):
    """One driver's laps, driven one after the other without stopping.

    numbers: the number of each lap in the race
    slower: how many times as long as the quickest lap each one takes
    pits: for each lap, whether it starts or ends in the pits
    begins: the time on the session's clock at which the first starts
    faults: the names of the faults the samples are to have. Each is
        explained where it is made, below.
    heights: whether the positions carry a height
    drs: the numbers of the laps driven with DRS open on the long
        straight
    late: seconds the speed is stamped later than the positions
    drift: seconds later still for every second of the stint
    reads: what the speed sensor reads, as a share of the true speed
    unreadable: the numbers of the laps whose speed cannot be read
    """
    numbers, slower = np.asarray(numbers), np.asarray(slower)
    starts = begins + np.concatenate([[0.0], np.cumsum(LAP_TIME * slower)])
    most = int((starts[-1] - begins) / 0.18) + 1

    def car_at(moment):
        """Which of the stint's laps the car is on at each moment (0 for
        the first), how many seconds into it, and how far round it has
        got."""
        which = np.clip(np.searchsorted(starts, moment, side='right') - 1,
                        0, len(slower) - 1)
        into = moment - starts[which]
        # clock holds the time at each distance on the quickest lap.
        # Read the other way round, it gives the distance at each time.
        return which, into, np.interp(into / slower[which], clock, along)

    # ---- positions, about 4.5 times a second ----
    moment = begins + np.cumsum(rng.uniform(0.18, 0.26, most))
    moment = moment[moment < starts[-1]]
    which, into, at = car_at(moment)
    stamp = moment.copy()               # the time each one is stamped
    fed = at.copy()                     # how far round it says the car is
    sideways = np.zeros(len(at))        # and how far left of the map
    passed = np.searchsorted(LOOPS, at, side='right') - 1   # last loop
    on_quickest = numbers[which] == QUICKEST    # taken on that lap

    if 'loops' in faults:
        # Between one timing loop and the next the positions follow the
        # car's speed, and run up to 3% fast or slow. At each loop they
        # are put right in one jump.
        rate = rng.uniform(-0.03, 0.03, (len(numbers), len(LOOPS)))
        fed += rate[which, passed] * (at - LOOPS[passed])
    if 'jumps' in faults:
        # Nothing wrong between the loops, but at four of them the
        # positions jump 8m forwards, and at the next they jump back
        fed += 8.0 * np.isin(passed, (1, 4, 6, 9))
    if 'slip' in faults:
        # On the quickest lap they sit 15m behind for 500m
        fed -= 15.0 * (on_quickest & (at > 1000) & (at < 1500))
    if 'stamps' in faults:
        # Each time stamp is up to 0.03s out
        stamp += rng.uniform(-0.03, 0.03, len(stamp))
    if 'strays' in faults:
        # One sample in every 700 is thrown 2m to one side
        sideways += 2.0 * (np.arange(len(at)) % 700 == 350)
    if 'off track' in faults:
        # Lap 9 runs up to 12m wide for 100m
        wide = (numbers[which] == 9) & (at > 900) & (at < 1000)
        sideways += wide * 12.0 * np.sin(math.pi * (at - 900) / 100.0) ** 2

    # The map every lap is drawn on, with its own faults
    left_x, left_y = -np.sin(heading), np.cos(heading)
    out = np.zeros(len(along))
    if 'map faults' in faults:
        # In three places it steps 0.3m to one side and back within 5m
        for place in (450.0, 1400.0, 2300.0):
            out += 0.3 * bump(place, 5.0)
    fed = fed % LAP
    columns = {
        'X': np.interp(fed, along, map_x + out * left_x)
        + sideways * np.interp(fed, along, left_x),
        'Y': np.interp(fed, along, map_y + out * left_y)
        + sideways * np.interp(fed, along, left_y)}
    if heights:
        columns['Z'] = np.interp(fed, along, height)
    positions = pd.DataFrame({name: np.round(10 * values)
                              for name, values in columns.items()})
    positions['SessionTime'] = pd.to_timedelta(stamp, unit='s')
    keep = np.ones(len(positions), dtype=bool)
    if 'zeros' in faults:
        # One sample in every 150 is missing, and FastF1 fills it with
        # zeros
        missing = np.arange(len(positions)) % 150 == 75
        positions.loc[missing, list(columns)] = 0.0
    if 'position hole' in faults:
        # The quickest lap has no positions for 2s, half-way round the
        # fourth bend
        keep &= ~(on_quickest & (into > 15.0) & (into < 17.0))
    if 'gaps' in faults:
        # Every lap has none for 1.2s on the long straight
        keep &= ~((into > 20.0) & (into < 21.2))
    if 'short lap' in faults:
        # Lap 15 has none after its first ten seconds
        keep &= ~((numbers[which] == 15) & (into > 10.0))
    positions, stamp = positions[keep], stamp[keep]

    # ---- speed, about 4 times a second, on a clock that runs late ----
    # (moment, which, into, at and keep start again here, for the
    # moments at which the speed is sampled)
    moment = begins + np.cumsum(rng.uniform(0.20, 0.28, most))
    moment = moment[moment < starts[-1]]
    which, into, at = car_at(moment)
    on_quickest = numbers[which] == QUICKEST
    open_flap = np.isin(numbers[which], drs) & (at > 1000) & (at < 1600)
    car = pd.DataFrame({
        'Speed': np.round(3.6 * reads * np.interp(at, along, true_speed)
                          / slower[which]),
        'DRS': np.where(open_flap, 12, 0),
        'SessionTime': pd.to_timedelta(
            moment + late + drift * (moment - begins), unit='s')})
    keep = np.ones(len(car), dtype=bool)
    if 'zeros' in faults:
        car.loc[np.arange(len(car)) % 150 == 75, 'Speed'] = 0.0
    if 'speed hole' in faults:
        # The quickest lap has no speed for 2.3s, from the end of the
        # long straight and through most of the braking for the hairpin
        keep &= ~(on_quickest & (into > 25.0) & (into < 27.3))
    # Three ways for the quickest lap to have too little speed to use:
    # none after its first 20 seconds, none in its first 4 seconds, and
    # only every fourth sample
    if 'speed stops' in faults:
        keep &= ~(on_quickest & (into > 20.0))
    if 'speed starts late' in faults:
        keep &= ~(on_quickest & (into < 4.0))
    if 'sparse speed' in faults:
        keep &= ~(on_quickest & (np.arange(len(car)) % 4 > 0))
    car = car[keep].reset_index(drop=True)
    ticked = seconds(car['SessionTime'])

    # ---- cut into laps by the time each sample is stamped ----
    laps = []
    for index, number in enumerate(numbers):
        begin, end = starts[index], starts[index + 1]
        laps.append(MadeUpLap(
            LapNumber=float(number), Driver=driver,
            LapTime=pd.Timedelta(seconds=end - begin),
            _positions=positions[(stamp >= begin) & (stamp < end)],
            _car=car, _rows=(int(np.searchsorted(ticked, begin)),
                             int(np.searchsorted(ticked, end))),
            _weather=weather, _pits=bool(pits[index]),
            _unreadable=number in unreadable))
    return laps, float(starts[-1])

def made_up_race(seed=1, **options):
    """The made-up race. Driver AAA runs laps 4 to 39: lap 4 slowly,
    lap 30 the quickest, lap 39 into the pits. Driver BBB runs laps 5
    to 8, all of them slower, and is first in the list of laps. The
    options are made_up_stint()'s."""
    rng = np.random.default_rng(seed)
    numbers = np.arange(4, 40)
    slower = 1.0 + rng.uniform(0.004, 0.02, len(numbers))
    slower[numbers == 4] = 1.12
    slower[numbers == QUICKEST] = 1.0
    slower[numbers == 39] = 1.05
    laps, ends = made_up_stint('AAA', numbers, slower, numbers == 39,
                               3000.0, rng, **options)
    others, ends = made_up_stint('BBB', [5, 6, 7, 8],
                                 [1.03, 1.031, 1.029, 1.032], [False] * 4,
                                 ends + 100.0, rng, **options)
    return types.SimpleNamespace(
        laps=MadeUpLaps(others + laps), total_laps=RACE_LAPS,
        slower=dict(zip(numbers.tolist(), slower.tolist())))

def read(race, driver=None, year=2024, **options):
    """build_reference() on a made-up race.

    build_reference() gets its race by `import telemetry` and a call to
    telemetry.load_session(). Python keeps every module it has imported
    in sys.modules, and an import looks there first. So for as long as
    build_reference() runs, sys.modules holds a stand-in under that
    name whose load_session() hands over the made-up race. Whatever was
    there before is put back afterwards. No F1 data is touched."""
    real = sys.modules.get('telemetry')
    sys.modules['telemetry'] = types.SimpleNamespace(
        load_session=lambda year, name, session_type='R': race)
    try:
        return sim.build_reference(year, 'made-up race', driver, **options)
    finally:
        if real is None:
            del sys.modules['telemetry']
        else:
            sys.modules['telemetry'] = real

circuit_points = cKDTree(np.column_stack([map_x, map_y]))

def typical(values):
    """The size of a typical one of these: the root of their mean
    square."""
    return float(np.sqrt(np.mean(np.square(values))))

def against_the_truth(reference, slower=1.0, without=None):
    """How far a reference is from the made-up truth.

    Each point of its track is matched to the nearest point of the
    circuit, and compared with what the car was really doing there, on
    a lap that took `slower` times as long as its quickest. Returns the
    typical and the worst error in its speed (km/h) and in its bends
    (as sideways g at the true speed), the worst error in its slope,
    and which point of the circuit each point of the track is. The
    first and last 20 points are left out, and so is the stretch of the
    circuit between the two distances in `without`, if given.
    """
    track = reference.track
    _, nearest = circuit_points.query(
        np.column_stack([track.channels['x'], track.channels['y']]))
    judged = np.ones(len(nearest), dtype=bool)
    judged[:20] = judged[-20:] = False
    if without is not None:
        judged &= ((along[nearest] < without[0])
                   | (along[nearest] > without[1]))
    speed = (reference.speed_kph
             - 3.6 * true_speed[nearest] / slower)[judged]
    sideways = (true_speed[nearest] ** 2 / GRAVITY
                * (track.curvature - np.abs(bend[nearest])))[judged]
    slope = (track.slope - true_slope[nearest])[judged]
    return dict(speed=typical(speed), worst_speed=float(np.abs(speed).max()),
                bends=typical(sideways),
                worst_bend=float(np.abs(sideways).max()),
                slope=float(np.abs(slope).max()), nearest=nearest)

check("the made-up circuit ends where it began",
      math.hypot(map_x[-1] - map_x[0], map_y[-1] - map_y[0]) < 0.05
      and abs(heading[-1] - 2 * math.pi) < 1e-4,
      f"{LAP:.0f}m round, and a lap by the made-up car takes "
      f"{LAP_TIME:.3f}s")

# ---- The race as it should be: nothing wrong with the samples but
# their rounding and the late clock ----
race = made_up_race()
reference = read(race)
miss = against_the_truth(reference)
built = reference.track

check("with no driver named, it takes whoever set the fastest lap",
      reference.driver == 'AAA', reference.driver)
check("and that driver's quickest lap, with the time it was given",
      reference.lap_choice == 'lap 30, the quickest'
      and abs(reference.official_time - LAP_TIME) < 1e-5,
      f"{reference.lap_choice}, {reference.official_time:.3f}s")
check("the slow lap and the lap into the pits are not used, and a clean "
      "map has no faults",
      built.source == ('pooled from 34 laps, features quicker than 1.5 a '
                       'second smoothed away, 0m of map faults left out'),
      built.source)
close("the speed's clock is found to run 0.085s late (s)",
      reference.stream_offset, LATE, 0.003)
check("and is put right",
      'later than position' in reference.clock
      and reference.clock.endswith('): corrected'), reference.clock)
check("the line covers the lap, bar a few metres at the ends",
      LAP - 35.0 < built.length < LAP + 3.0,
      f"{built.length:.0f}m of {LAP:.0f}m")
check("the speed at each point is the true car's there",
      miss['speed'] < 1.5 and miss['worst_speed'] < 18.0,
      f"typically within {miss['speed']:.2f} km/h, at worst "
      f"{miss['worst_speed']:.1f}")
own_time = float(np.sum(built.step / true_speed[miss['nearest']]))
close("so the time along the line is the true car's (s)",
      reference.lap_time, own_time, 0.04)
check("the bends are the circuit's",
      miss['bends'] < 0.15 and miss['worst_bend'] < 1.3,
      f"typically within {miss['bends']:.2f}g at the true speed, at "
      f"worst {miss['worst_bend']:.2f}g")
close("the tightest of them, radius (m)", 1 / built.curvature.max(), 20.0,
      1.5)
check("the slope is the circuit's", miss['slope'] < 0.008,
      f"within {miss['slope']:.2%}, on slopes of up to "
      f"{np.abs(true_slope).max():.1%}")
check("and the climb is described", reference.hills.startswith(
      '30m from the lowest point of the lap to the highest'),
      reference.hills[:51])
check("the car is given the weight it had on lap 30 of 40",
      reference.mass == sim.race_weight(2024, 30, 40)
      and 'lap 30 of 40' in reference.conditions,
      "no weight" if reference.mass is None else f"{reference.mass:.1f}kg")
close("and the air of the day (kg/m3)", reference.air_density,
      true_car.air_density, 1e-12)
check("nothing is eased and no problems are noted",
      reference.eased == "" and reference.notes == [],
      f"{reference.eased} {reference.notes}")

# The reference hangs together: the true car, run along the track as it
# was built, does the speeds that were read
rerun = sim.simulate(sim.straightened(built, TRUE_LINE), true_car)
apart = typical(rerun.speed * 3.6 - reference.speed_kph)
check("the true car, run along the track as built, does the speeds as read",
      apart < 3.0, f"typically within {apart:.2f} km/h")
close("and takes the same time (s)", rerun.lap_time, reference.lap_time, 0.1)

# From the raw samples to a car. One lap pins a car down only roughly
# (see FITTING in the notes at the top of Simulator.py), so the power,
# downforce and drag are not asked to be as close as the rest.
shared, cars, tracks, outcome = sim.fit_multi([reference], F1_2024,
                                              verbose=False)
found = cars[0]
close("the fit, from this lap: braking share", found.brake_fraction, 0.7,
      0.03)
close("drive share", found.drive_fraction, 0.5, 0.02)
close("line", tracks[0].curvature.max() / built.curvature.max(), TRUE_LINE,
      0.03)
for label, name, tolerance in (("power", 'power', 0.05),
                               ("downforce", 'cla', 0.08),
                               ("drag", 'cda', 0.06)):
    close(f"{label}, as a share of the true car's",
          getattr(found, name) / getattr(true_car, name), 1.0, tolerance)
worth_true = sim.setup_effects(sim.straightened(true_map, TRUE_LINE),
                               true_car)
worth_found = sim.setup_effects(tracks[0], found)
furthest = max(abs(worth_found[label] / worth_true[label] - 1)
               for label in worth_true)
check("and its table of what a change is worth is the true car's",
      furthest < 0.08, f"every row within {furthest:.0%}")

# ---- Three faults in the map, and nothing else wrong ----
def metres_left_out(reference):
    """How many metres of the map were left out as faults, or None if
    its source does not say."""
    said = reference.track.source.rsplit(', ', 1)[-1]
    if not said.endswith('m of map faults left out'):
        return None
    return int(said.split('m')[0])

faulty = read(made_up_race(faults=('map faults',)))
found_faults = metres_left_out(faulty)
check("three faults in the map are found and left out",
      found_faults is not None and 5 <= found_faults <= 40,
      faulty.track.source.rsplit(', ', 1)[-1])

# ---- The same race as real data has it: everything in ROUGH ----
rough = read(made_up_race(faults=ROUGH))
miss = against_the_truth(rough)
found_faults = metres_left_out(rough)
check("rough data: the same lap, and the lap with too few samples is "
      "left out", rough.lap_choice == 'lap 30, the quickest'
      and rough.track.source.startswith('pooled from 33 laps,'),
      rough.track.source[:19])
check("the stray samples and the lap that ran wide are left out with "
      "the map's faults", found_faults is not None
      and 40 <= found_faults <= 150, rough.track.source.rsplit(', ', 1)[-1])
close("the clock is still found, near enough (s)", rough.stream_offset,
      LATE, 0.03)
check("the speed at each point is still the true car's",
      miss['speed'] < 3.0 and miss['worst_speed'] < 18.0,
      f"typically within {miss['speed']:.2f} km/h, at worst "
      f"{miss['worst_speed']:.1f}")
check("the bends are still the circuit's",
      miss['bends'] < 0.16 and miss['worst_bend'] < 1.6,
      f"typically within {miss['bends']:.2f}g, at worst "
      f"{miss['worst_bend']:.2f}g")
check("and so is the slope", miss['slope'] < 0.008,
      f"within {miss['slope']:.2%}")
check("none of it is a problem worth a note", rough.notes == [],
      "; ".join(rough.notes))
rerun = sim.simulate(sim.straightened(rough.track, TRUE_LINE), true_car)
apart = typical(rerun.speed * 3.6 - rough.speed_kph)
check("the true car, run along that track, still does the speeds as read",
      apart < 4.0, f"typically within {apart:.2f} km/h")

# ---- Faults that are checked one at a time ----
QUICK = dict(max_laps=8)        # eight laps are enough for most of these

def kilos(mass):
    """A weight, for the end of a check's line."""
    return "no weight" if mass is None else f"{mass:.1f}kg"

slipped = against_the_truth(read(made_up_race(faults=('slip',))))
miss = against_the_truth(reference)
check("positions that sit 15m behind for 500m do not move the speed",
      abs(slipped['speed'] - miss['speed']) < 0.1
      and abs(slipped['worst_speed'] - miss['worst_speed']) < 1.0,
      f"typically within {slipped['speed']:.2f} km/h of the truth with "
      f"them, {miss['speed']:.2f} without")

holed = read(made_up_race(faults=('speed hole',)), **QUICK)
miss = against_the_truth(holed, without=(1540.0, 1760.0))
check("a 2.3s hole in the speed: the lap is still used, and the hole is "
      "noted", holed.lap_choice == 'lap 30, the quickest'
      and len(holed.notes) > 0
      and 'hole in its speed data' in holed.notes[0], "; ".join(holed.notes))
check("and the speed either side of it is still in the right place",
      miss['speed'] < 1.5 and miss['worst_speed'] < 18.0,
      f"typically within {miss['speed']:.2f} km/h, at worst "
      f"{miss['worst_speed']:.1f}")

high = read(made_up_race(reads=1.015), **QUICK)
miss = against_the_truth(high, slower=1 / 1.015)
check("a speed sensor that reads 1.5% high: each reading is still put in "
      "the right place", miss['speed'] < 1.5 and miss['worst_speed'] < 18.0,
      f"typically within {miss['speed']:.2f} km/h of what it read")
check("and the lap time it adds up to is seen not to match",
      len(high.notes) == 1
      and 'the line or the speed data is wrong' in high.notes[0],
      "; ".join(high.notes))

# A sensor 10% high cannot be this lap's: the distance it adds up to is
# not believed, and each reading goes where the positions put it. The
# lap is also given a hole in its positions, where that is a guess.
wild = read(made_up_race(reads=1.1, faults=('position hole',)), **QUICK)
miss = against_the_truth(wild, slower=1 / 1.1, without=(760.0, 930.0))
check("a sensor 10% high: the speed is placed by the positions, and that "
      "is said", any('positions were used' in note for note in wild.notes)
      and miss['speed'] < 3.0,
      f"typically within {miss['speed']:.2f} km/h of what it read")
check("and so is the hole in the positions",
      any('hole of over a second in its position data' in note
          for note in wild.notes), f"{len(wild.notes)} notes")
close("the clock is still found, whatever the sensor reads (s)",
      wild.stream_offset, LATE, 0.005)

jumpy = read(made_up_race(faults=('jumps',)), **QUICK)
close("positions that jump 8m at the timing loops: the clock is still "
      "found (s)", jumpy.stream_offset, LATE, 0.01)

gappy = read(made_up_race(faults=('gaps',)), **QUICK)
check("a hole in every lap's positions is noted",
      len(gappy.notes) == 1 and gappy.notes[0].startswith(
          'every lap has a hole of over a second in its position data'),
      "; ".join(gappy.notes))

# ---- Which lap, and whose ----
# The second and third quickest of the clean laps: the number of each,
# and how many times as long as lap 30 it took
(second_slower, second), (third_slower, third) = sorted(
    (slower, number) for number, slower in race.slower.items()
    if number not in (4, 30, 39))[:2]
with_drs = read(made_up_race(drs=[QUICKEST]), **QUICK)
miss = against_the_truth(with_drs, slower=second_slower)
from_lap_30 = against_the_truth(with_drs)['speed']
check("DRS open on the quickest lap: the quickest with it shut is taken",
      with_drs.lap_choice.startswith(
          f"lap {second}, the quickest with DRS shut (lap 30 was ")
      and abs(with_drs.official_time - LAP_TIME * second_slower) < 1e-5
      and with_drs.notes == [], with_drs.lap_choice)
check("with that lap's speed, not lap 30's, and that lap's weight",
      miss['speed'] < 1.5 and miss['speed'] < from_lap_30
      and with_drs.mass == sim.race_weight(2024, second, 40),
      f"within {miss['speed']:.2f} km/h of it and {from_lap_30:.2f} of "
      f"lap 30's, {kilos(with_drs.mass)}")
all_open = read(made_up_race(drs=range(1, 41)), **QUICK)
check("DRS open on every lap: the quickest is taken, with a warning",
      all_open.lap_choice == 'lap 30, the quickest'
      and len(all_open.notes) == 1
      and all_open.notes[0].startswith('DRS was open for'),
      "; ".join(all_open.notes)[:40] + "...")

# A lap whose speed cannot be used is passed over for the next quickest
for label, missing in (
        ("no speed for the quickest lap", dict(unreadable=[QUICKEST])),
        ("speed that stops half-way round it",
         dict(faults=('speed stops',))),
        ("speed that starts 4s into it",
         dict(faults=('speed starts late',))),
        ("only every fourth speed sample of it",
         dict(faults=('sparse speed',)))):
    next_best = read(made_up_race(**missing), **QUICK)
    check(f"{label}: the next quickest is taken, and it is said why",
          next_best.lap_choice == (f"lap {second}, the quickest with speed "
                                   f"data that can be used")
          and abs(next_best.official_time - LAP_TIME * second_slower) < 1e-5
          and next_best.mass == sim.race_weight(2024, second, 40)
          and next_best.notes == ['lap 30 was quicker, but had no speed '
                                  'data that could be used'],
          "; ".join(next_best.notes))
skipped = read(made_up_race(drs=[QUICKEST], unreadable=[second]), **QUICK)
check("DRS open on the quickest and no speed for the next: the one after "
      "that, and it is said why", skipped.lap_choice.startswith(
          f"lap {third}, the quickest with DRS shut (lap 30 was ")
      and abs(skipped.official_time - LAP_TIME * third_slower) < 1e-5
      and skipped.notes == [f"lap {second} was quicker, but had no speed "
                            f"data that could be used"],
      "; ".join(skipped.notes))
message = refused(lambda: read(made_up_race(unreadable=range(1, 41))))
check("no speed for any lap", message is not None, message or "")

named = read(race, 'BBB')
miss = against_the_truth(named, slower=1.029)
check("a driver who is named: that driver's quickest lap",
      named.driver == 'BBB' and named.lap_choice == 'lap 7, the quickest'
      and named.track.source.startswith('pooled from 4 laps,')
      and miss['speed'] < 1.5,
      f"{named.lap_choice}, speed within {miss['speed']:.2f} km/h")
message = refused(lambda: read(race, 'CCC'))
check("a driver with no laps", message is not None, message or "")
message = refused(lambda: sim._fastest_driver(MadeUpLaps([])))
check("a race with no fastest lap", message is not None,
      (message or "")[:52] + "...")
asked = read(race, max_laps=5, spacing=2.0, lam=1000.0, drag_limited=False)
check("what it is asked for it does: five laps, points 2m apart, a "
      "stiffness, no top speed",
      asked.track.source.startswith('pooled from 5 laps, smoothed with '
                                    'lam 1000,')
      and asked.track.step == 2.0 and asked.drag_limited is False
      and reference.drag_limited is True, asked.track.source[:44])

# ---- What it does when something is missing ----
no_heights = read(made_up_race(heights=False), **QUICK)
check("no heights in the positions: a flat track, said so",
      not no_heights.track.slope.any() and 'no heights' in no_heights.hills,
      no_heights.hills)
no_weather = read(made_up_race(weather=None), **QUICK)
check("no weather: no air density, said so",
      no_weather.air_density is None
      and 'no weather data' in no_weather.conditions,
      no_weather.conditions.split('; ')[1])
old = read(race, year=1990, **QUICK)
check("a season with no weight on record: no weight, said so",
      old.mass is None and 'not known' in old.conditions,
      old.conditions.split('; ')[0])
length = race.total_laps
del race.total_laps
unplanned = read(race, **QUICK)
race.total_laps = length
check("no race distance on record: the last lap anyone drove stands in",
      unplanned.mass == sim.race_weight(2024, 30, 39)
      and 'lap 30 of 39' in unplanned.conditions, kilos(unplanned.mass))

# ---- The two clocks ----
agreeing = read(made_up_race(late=0.0), **QUICK)
check("clocks that agree are said to", agreeing.clock
      == 'speed and position clocks agree', agreeing.clock)
drifting = read(made_up_race(drift=0.0005), **QUICK)
check("clocks that drift apart through the race are not corrected",
      drifting.stream_offset is None
      and drifting.clock.endswith('not corrected'), drifting.clock)
pair = made_up_race()
pair.laps = MadeUpLaps(lap for lap in pair.laps.laps
                       if lap['LapNumber'] in (29, 30))
two = read(pair)
check("nor are clocks measured on only two laps",
      two.stream_offset is None and 'could not be compared' in two.clock,
      two.clock)

# One lap's clock, measured on its own
quickest = race.laps.pick_fastest()
samples, readings = quickest.get_pos_data(), quickest.get_car_data()
stamped = seconds(samples['SessionTime'])
x, y = samples['X'].to_numpy(), samples['Y'].to_numpy()
ticked = seconds(readings['SessionTime'])
shown = readings['Speed'].to_numpy()
close("one lap's clock, measured on its own (s)",
      sim._stream_offset(stamped, x, y, ticked, shown), LATE, 0.005)
check("a clock 1.5s late is beyond its reach, and it says it cannot tell",
      sim._stream_offset(stamped, x, y, ticked + 1.5, shown) is None)
check("nor can it with the speed from half a lap away",
      sim._stream_offset(stamped, x, y, ticked,
                         np.roll(shown, len(shown) // 2)) is None)
check("nor with only every fifth speed sample",
      sim._stream_offset(stamped, x, y, ticked[::5], shown[::5]) is None)

# One lap's speed, placed on the track by how far the car had gone
placed = sim._speed_by_distance(built, stamped, x, y, ticked, shown, LATE)
check("one lap's speed, placed by the distance the car had gone",
      placed is not None
      and typical(placed - 3.6 * true_speed[against_the_truth(
          reference)['nearest']]) < 1.5)
check("speed that adds up to a lap a tenth too long is not this lap's",
      sim._speed_by_distance(built, stamped, x, y, ticked, shown * 1.1,
                             LATE) is None)
check("and every fifth speed sample is too few to place",
      sim._speed_by_distance(built, stamped, x, y, ticked[::5], shown[::5],
                             LATE) is None)
# FastF1's positions are in tenths of a metre, the line in metres
between = [5 * (built.channels[name][100] + built.channels[name][101])
           for name in ('x', 'y')]
close("a sample half-way between two points of the line is placed "
      "half-way (m)", sim._distance_along(built, between[:1], between[1:])[0],
      0.5 * (built.distance[100] + built.distance[101]), 0.01)

# ---- Told that a car of this kind has no more than 2g, it has to
# call the short bends kinks and ease them, and report the long ones ----
gentle = read(race, most_g=2.0, **QUICK)
asks = (gentle.speed_kph / 3.6) ** 2 * gentle.track.curvature / GRAVITY
as_it_came = gentle.track.channels.get('map_curvature',
                                       gentle.track.curvature)
eased = gentle.track.curvature < as_it_came
check("bends too tight for the speed are eased, and said to be",
      eased.any() and abs(asks[eased].max() - 2.0) < 1e-9
      and gentle.eased.startswith('map eased over'), gentle.eased[:53])
check("and the long ones are left alone and noted",
      asks.max() > 2.5
      and any('too long to be a kink' in note for note in gentle.notes),
      f"still asks for {asks.max():.1f}g somewhere")

# ---------------------------------------------------------------------------
print("14. At the size of a Formula Student car")
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
