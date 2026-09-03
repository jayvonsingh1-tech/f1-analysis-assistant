import logging

import fastf1
import numpy as np

fastf1.Cache.enable_cache('cache')
fastf1.set_log_level(logging.WARNING)

def format_time(seconds):
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}:{remainder:06.3f}"

def load_race(year, race):
    session = fastf1.get_session(year, race, 'R')
    session.load()
    return session

def clean_laps(session, driver):
    laps = session.laps.pick_drivers(driver).copy()
    laps['Stint'] = (laps['TyreLife'] == 1).cumsum()
    laps['Seconds'] = laps['LapTime'].dt.total_seconds()
    return laps

def analyse_stints(session, driver):
    laps = clean_laps(session, driver)

    print(f"\n{driver}")

    for stint, stint_laps in laps.groupby('Stint'):
        clean = stint_laps[stint_laps['PitInTime'].isna() & stint_laps['PitOutTime'].isna()]
        slope = np.polyfit(clean['TyreLife'], clean['Seconds'], 1)[0]
        print(f"  Stint {stint}: {clean['Compound'].iloc[0]}, "
              f"{len(stint_laps)} laps, avg {format_time(clean['Seconds'].mean())}, "
              f"trend {slope:+.3f}s/lap")

def compare(session, driver_a, driver_b):
    for driver in (driver_a, driver_b):
        analyse_stints(session, driver)

    print(f"\n{driver_a} vs {driver_b}")

    results = {}
    for driver in (driver_a, driver_b):
        laps = clean_laps(session, driver)
        clean = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]

        results[driver] = {
            'median': clean['Seconds'].median(),
            'best': clean['Seconds'].min(),
            'stops': laps['PitInTime'].notna().sum(),
        }
        r = results[driver]
        print(f"  {driver}: median {format_time(r['median'])}, "
              f"best {format_time(r['best'])}, {r['stops']} stops")

    gap = results[driver_a]['median'] - results[driver_b]['median']
    quicker = driver_b if gap > 0 else driver_a
    print(f"  {quicker} quicker by {abs(gap):.3f}s per lap on median pace")

while True:
    command = input("\n> ").strip()

    if command in ('quit', 'exit', ''):
        break

    parts = command.split()

    try:
        year, race = int(parts[0]), parts[1]
        session = load_race(year, race)
        print(f"\n{race.title()} {year}")

        if len(parts) == 3:
            analyse_stints(session, parts[2].upper())
        elif len(parts) == 4:
            compare(session, parts[2].upper(), parts[3].upper())
        else:
            print("Try: 2024 Monza NOR   or   2024 Monza NOR PIA")
    except Exception as error:
        print(f"Didn't work: {error}")