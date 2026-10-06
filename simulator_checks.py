"""Checks for Simulator.py. Run them with:  python simulator_checks.py

Each check compares the simulator with an answer known some other way:
a formula from a physics textbook, a rule every lap has to obey, or the
simulator itself run a different way. None of them needs any F1 data,
so the whole file runs in well under a minute.

Run it after every change to Simulator.py. A line that says BAD means
the change broke something that used to work. The last line says ALL OK,
or how many checks failed.

The checks were themselves checked: the simulator was broken on purpose
in 69 different places, one at a time, and every break made at least
one line here say BAD.

What they do not reach is build_reference(), the part that reads
FastF1's data. That is tested outside this file, on made-up sessions
and on real ones.

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
print("13. At the size of a Formula Student car")
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
